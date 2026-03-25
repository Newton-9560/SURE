"""
trainer/fixmatch.py
-------------------
FixMatch semi-supervised training for the MAMP-MLP safety classifier.

Pipeline per epoch
------------------
1. generate_pseudo_labels()  – run model on weak-augmented unlabeled hidden
                               states; keep (strong_aug_x, pseudo_label) pairs
                               whose confidence exceeds a threshold τ.
2. build_pseudo_loader()     – wrap collected pairs in a DataLoader whose items
                               are compatible with MAMP/trainer.train_epoch().
3. train_epoch()             – supervised step on labeled DataLoader  (imported
                               from MAMP.trainer, unchanged).
4. train_epoch()             – unsupervised step on pseudo-labeled DataLoader,
                               using a ScaledBCELoss(lambda_u) as criterion so
                               the gradient magnitude is weighted correctly.
5. evaluate()                – validation metrics, checkpointing best model.

Augmentation contract
---------------------
Both `weak_aug` and `strong_aug` receive a **batched** float32 tensor of shape
(B, hidden_dim) on the *same device as the model* and return a tensor of the
same shape.  Default implementations live in `utils.augmentations`:

    from utils.augmentations import weak_aug, strong_aug
    # or customise:
    from utils.augmentations import WeakAugmentation, StrongAugmentation
    weak_aug   = WeakAugmentation(noise_std=0.02)
    strong_aug = StrongAugmentation(dropout_p=0.2, noise_std=0.1)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from MAMP.trainer import train_epoch, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PseudoLabeledDataset(Dataset):
    """Minimal Dataset wrapping (x_strong, pseudo_label, weight) triples.

    Items are returned as (x, y, info) to match the interface expected by
    MAMP.trainer.train_epoch (which ignores info).  _fixmatch_epoch reads
    info["weight"] to scale the unsupervised BCE loss per sample.
    """

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, w = self.samples[idx]
        return x, y, {"weight": w}


class _ScaledBCELoss(nn.Module):
    """BCELoss scaled by a constant weight (used for the unsupervised term)."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self._bce = nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self._bce(input, target)


def _print_pseudo_stats(stats: dict) -> None:
    """Pretty-print global and per-dataset pseudo-label statistics."""
    def _fmt(v):
        return f"{v:.4f}" if (v == v) else "   nan"   # nan != nan

    def _pct(v):
        return f"{v:.1%}" if (v == v) else "  nan"

    tau_pos = stats.get("threshold_pos", float("nan"))
    tau_neg = stats.get("threshold_neg", float("nan"))
    print(
        f"  Pseudo │ "
        f"tau_pos(unsafe)={_fmt(tau_pos)}  tau_neg(safe)={_fmt(tau_neg)}  │  "
        f"total={stats['n_total']}  "
        f"selected={stats['n_pseudo']}({_pct(stats['coverage'])})  "
        f"pos(unsafe)={stats['n_pseudo_pos']}  neg(safe)={stats['n_pseudo_neg']}  │  "
        f"acc={_fmt(stats['pseudo_acc'])}  "
        f"acc_pos(unsafe)={_fmt(stats['pseudo_acc_pos'])}  "
        f"acc_neg(safe)={_fmt(stats['pseudo_acc_neg'])}"
    )
    has_ds_tau = any("threshold_pos" in s for s in stats["per_dataset"].values())

    def _tau(s, key):
        v = s.get(key, float("nan"))
        return f"{v:.3f}" if (v == v) else "  nan"

    tau_cols  = f" {'tau_p':>6} {'tau_n':>6}" if has_ds_tau else ""
    col = (f"  {'Dataset':<30} {'total':>6} {'sel':>6} {'cov':>7}"
           f"{tau_cols} {'pos':>5} {'neg':>5} {'acc':>7} {'acc_pos':>8} {'acc_neg':>8}")
    print(col)
    print("  " + "-" * (len(col) - 2))
    for ds, s in sorted(stats["per_dataset"].items()):
        tau_vals = (f" {_tau(s, 'threshold_pos'):>6} {_tau(s, 'threshold_neg'):>6}"
                    if has_ds_tau else "")
        print(
            f"  {ds:<30} {s['n_total']:>6} {s['n_pseudo']:>6} {_pct(s['coverage']):>7}"
            f"{tau_vals}"
            f" {s['n_pseudo_pos']:>5} {s['n_pseudo_neg']:>5}"
            f" {_fmt(s['pseudo_acc']):>7} {_fmt(s['pseudo_acc_pos']):>8} {_fmt(s['pseudo_acc_neg']):>8}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    model,
    unlabeled_dataset,
    device,
    threshold_pos: float,
    threshold_neg: float,
    weak_aug,
    strong_aug,
    batch_size: int = 32,
    ds_thresholds: "dict[str, tuple[float, float]] | None" = None,
    use_pseudo_weights: bool = True,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Generate pseudo-labeled training samples from unlabeled hidden states.

    For each unlabeled batch:
      - Apply weak_aug → get model probability p  (p = P(safe)).
      - A sample is pseudo-labeled unsafe  (label 0) if p <  0.5 AND (1-p) >= tau_pos.
      - A sample is pseudo-labeled safe    (label 1) if p >= 0.5 AND  p   >= tau_neg.
      - Per-class thresholds allow curriculum pseudo-labeling (FlexMatch style):
        the harder class gets a lower threshold so more of its samples qualify.

    Args:
        model:              MAMP_MLP (or compatible) with sigmoid output.
        unlabeled_dataset:  HiddenStateDataset for unlabeled split.
        device:             torch device.
        threshold_pos:      Global confidence threshold for unsafe pseudo-labels
                            (used as fallback when ds_thresholds is None or a
                            dataset is not present in ds_thresholds).
        threshold_neg:      Global confidence threshold for safe pseudo-labels.
        weak_aug:           Callable (B, D) → (B, D) – light augmentation.
        strong_aug:         Callable (B, D) → (B, D) – strong augmentation.
        batch_size:         Batch size used during inference.
        ds_thresholds:      Optional per-dataset threshold overrides:
                            {dataset_name: (tau_pos, tau_neg)}.
                            When provided, each sample uses its dataset's
                            thresholds; falls back to global threshold_pos /
                            threshold_neg for unseen datasets.

    Returns:
        (pseudo_samples, stats):
          pseudo_samples – list of (x_strong, pseudo_label) CPU-tensor tuples
                           for samples that exceeded the confidence threshold.
          stats          – dict with global and per-dataset keys:
              n_total, n_pseudo, coverage,
              n_pseudo_pos (predicted safe), n_pseudo_neg (predicted unsafe),
              pseudo_acc, pseudo_acc_pos (acc on true-safe), pseudo_acc_neg (acc on true-unsafe),
              per_dataset: {dataset_name: same keys above}
    """
    # ── Pseudo-label weighting ─────────────────────────────────────────────
    # Each pseudo-labeled sample is weighted by combining MLP confidence with
    # the alignment between the MLP prediction and the LLM's refusal decision,
    # modulated by the LLM's perplexity (normalised to [0,1]).
    #
    # Let conf = max(p, 1-p)  (MLP confidence, always in [0.5, 1])
    # Let agrees = (MLP predicts safe) == (LLM did NOT refuse)
    # Let unc = normalised Perplexity  (0 = LLM certain, 1 = LLM uncertain)
    #
    #   Agreement   : weight = conf × (1 − unc)
    #     Both sources agree AND the LLM was certain → highest trust.
    #   Disagreement: weight = conf × unc
    #     Sources disagree; low weight unless the LLM itself was uncertain,
    #     which weakens the significance of the disagreement.
    #
    # NOTE – The original TODO had the uncertainty factors swapped (agreement
    # used unc and disagreement used 1−unc).  That would reward uncertain LLM
    # agreement and penalise certain LLM disagreement, both backwards.
    #
    # Fallback: if refusal or perplexity is unavailable, weight = conf.
    model.eval()
    pseudo_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # Global counters
    g_total = g_pseudo = g_pos = g_neg = 0
    # "unsafe" = positive class (label=0), "safe" = negative class (label=1)
    g_pos_sel = g_pos_cor = g_neg_sel = g_neg_cor = 0  # pos=true-unsafe, neg=true-safe

    # Per-dataset counters
    # ds -> [total, pseudo, n_pseudo_pos, n_pseudo_neg, pos_sel, pos_cor, neg_sel, neg_cor]
    # pos=predicted/true unsafe (label=0), neg=predicted/true safe (label=1)
    ds_counters: dict[str, list[int]] = {}

    loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    with torch.no_grad():
        for x, y, info in loader:
            x = x.to(device)                              # (B, D)
            y = y.to(device)                              # (B,)  ground-truth labels
            dataset_names = info["dataset_name"]          # list[str], length B

            g_total += x.size(0)
            for ds in dataset_names:
                if ds not in ds_counters:
                    ds_counters[ds] = [0] * 8
                ds_counters[ds][0] += 1

            x_weak   = weak_aug(x)                        # (B, D)
            # x_strong = strong_aug(x)                      # (B, D)
            x_strong = x # keep the original hidden states

            prob = model(x_weak).squeeze(1)               # (B,)  in (0, 1)

            # Build per-sample threshold tensors (vectorised lookup)
            if ds_thresholds is not None:
                tp_vec = torch.tensor(
                    [ds_thresholds.get(ds, (threshold_pos, threshold_neg))[0]
                     for ds in dataset_names],
                    dtype=torch.float32, device=device,
                )
                tn_vec = torch.tensor(
                    [ds_thresholds.get(ds, (threshold_pos, threshold_neg))[1]
                     for ds in dataset_names],
                    dtype=torch.float32, device=device,
                )
            else:
                tp_vec = threshold_pos
                tn_vec = threshold_neg

            # Per-class thresholds: unsafe if (1-p) >= tau_pos, safe if p >= tau_neg
            mask_pos = (prob <  0.5) & ((1.0 - prob) >= tp_vec)   # predict unsafe
            mask_neg = (prob >= 0.5) & (prob           >= tn_vec)  # predict safe
            mask = mask_pos | mask_neg                              # (B,) bool
            if not mask.any():
                continue

            sel_indices   = mask.nonzero(as_tuple=True)[0]          # indices in batch
            pseudo_label  = (prob[mask] >= 0.5).float()             # (B',)
            true_label    = y[mask]                                  # (B',)
            x_strong_kept = x_strong[mask].cpu()                    # (B', D)

            g_pseudo += sel_indices.size(0)
            g_pos    += (pseudo_label <  0.5).sum().item()   # predicted unsafe
            g_neg    += (pseudo_label >= 0.5).sum().item()   # predicted safe

            # positive class = unsafe (label=0), negative class = safe (label=1)
            pos_mask = true_label <  0.5   # true unsafe
            neg_mask = ~pos_mask           # true safe
            g_pos_sel += pos_mask.sum().item()
            g_neg_sel += neg_mask.sum().item()
            if pos_mask.any():
                g_pos_cor += (pseudo_label[pos_mask] <  0.5).sum().item()
            if neg_mask.any():
                g_neg_cor += (pseudo_label[neg_mask] >= 0.5).sum().item()

            # Per-dataset accumulation for the selected samples
            for j, orig_idx in enumerate(sel_indices.tolist()):
                ds  = dataset_names[orig_idx]
                pl  = pseudo_label[j].item()
                tl  = true_label[j].item()
                c   = ds_counters[ds]
                c[1] += 1                           # n_pseudo
                c[2] += int(pl <  0.5)              # n_pseudo_pos (predicted unsafe)
                c[3] += int(pl >= 0.5)              # n_pseudo_neg (predicted safe)
                if tl < 0.5:                        # true unsafe = positive
                    c[4] += 1                       # pos_sel
                    c[5] += int(pl <  0.5)          # pos_cor
                else:                               # true safe = negative
                    c[6] += 1                       # neg_sel
                    c[7] += int(pl >= 0.5)          # neg_cor

            # ── Per-sample weights ──────────────────────────────────────────
            if not use_pseudo_weights:
                # Uniform weighting: all pseudo-labels contribute equally.
                weight_sel = torch.ones(x_strong_kept.size(0), dtype=torch.float32)
            else:
                prob_sel = prob[mask]                              # (B',) P(safe)
                conf_sel = torch.where(
                    prob_sel >= 0.5, prob_sel, 1.0 - prob_sel     # (B',) ∈ [0.5, 1]
                )

                # Try to extract normalised perplexity and refusal from batch info.
                perp_vec = refusal_vec = None
                unc_info = info.get("uncertainty")
                if isinstance(unc_info, dict):
                    perp_raw = unc_info.get("Perplexity")
                    if isinstance(perp_raw, torch.Tensor):
                        perp_vec = perp_raw.float().to(device)    # (B,)
                ref_raw = info.get("refusal")
                if isinstance(ref_raw, torch.Tensor):
                    refusal_vec = ref_raw.to(device).bool()       # (B,)

                if perp_vec is not None and refusal_vec is not None:
                    perp_sel    = perp_vec[mask]                  # (B',)
                    refusal_sel = refusal_vec[mask]               # (B',)
                    mlp_safe    = prob_sel >= 0.5                 # True = MLP predicts safe
                    llm_safe    = ~refusal_sel                    # True = LLM did not refuse
                    agrees      = mlp_safe == llm_safe            # (B',)
                    weight_sel  = torch.where(
                        agrees,
                        conf_sel * (1.0 - perp_sel),   # agreement: trust ∝ LLM certainty
                        conf_sel * perp_sel,            # disagreement: trust ∝ LLM uncertainty
                    )                                             # (B',) ∈ [0, 1]
                else:
                    weight_sel = conf_sel                         # fallback: confidence only

            weight_cpu = weight_sel.cpu()
            for i in range(x_strong_kept.size(0)):
                pseudo_samples.append((
                    x_strong_kept[i],
                    pseudo_label[i].cpu(),
                    weight_cpu[i],                            # scalar tensor
                ))

    _nan = float("nan")

    def _stats(n_tot, n_pse, n_pos, n_neg, pos_sel, pos_cor, neg_sel, neg_cor):
        # pos = unsafe (label=0, the positive/detected class)
        # neg = safe   (label=1, the negative class)
        return {
            "n_total":          n_tot,
            "n_pseudo":         n_pse,
            "coverage":         n_pse / n_tot   if n_tot   > 0 else _nan,
            "n_pseudo_pos":     n_pos,   # predicted unsafe
            "n_pseudo_neg":     n_neg,   # predicted safe
            "pseudo_acc":       (pos_cor + neg_cor) / n_pse if n_pse    > 0 else _nan,
            "pseudo_acc_pos":   pos_cor / pos_sel            if pos_sel  > 0 else _nan,
            "pseudo_acc_neg":   neg_cor / neg_sel            if neg_sel  > 0 else _nan,
        }

    global_stats = _stats(g_total, g_pseudo, g_pos, g_neg,
                          g_pos_sel, g_pos_cor, g_neg_sel, g_neg_cor)
    global_stats["per_dataset"] = {
        ds: _stats(*c) for ds, c in sorted(ds_counters.items())
    }

    return pseudo_samples, global_stats


def build_pseudo_loader(
    pseudo_samples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> DataLoader:
    """Wrap pseudo-labeled samples in a shuffled DataLoader.

    The DataLoader yields (x, y, info) batches compatible with
    MAMP.trainer.train_epoch.

    Args:
        pseudo_samples: Output of generate_pseudo_labels().
        batch_size:     Mini-batch size.

    Returns:
        DataLoader ready for training.
    """
    dataset = _PseudoLabeledDataset(pseudo_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def _fixmatch_epoch(
    model,
    labeled_loader: DataLoader,
    pseudo_loader,          # DataLoader | None
    optimizer,
    sup_criterion,
    unsup_criterion,
    device,
) -> tuple[float, float]:
    """One FixMatch epoch: combined sup + unsup loss in a single backward pass.

    Iterates over the longer of labeled_loader / pseudo_loader; the shorter
    one is cycled so every step sees both a labeled and a pseudo-labeled batch.
    The two losses are summed before backward so the optimizer sees a single
    combined gradient.

    Returns:
        (avg_sup_loss, avg_unsup_loss) – per-sample averages for logging.
    """
    model.train()
    sup_total = unsup_total = 0.0
    sup_n = unsup_n = 0

    # Use the longer loader as primary; cycle the shorter one
    n_pseudo = len(pseudo_loader) if pseudo_loader is not None else 0
    pseudo_is_primary = n_pseudo > len(labeled_loader)

    if pseudo_is_primary:
        primary_loader, secondary_loader = pseudo_loader, labeled_loader
    else:
        primary_loader, secondary_loader = labeled_loader, pseudo_loader

    secondary_iter = iter(secondary_loader) if secondary_loader is not None else None

    def _next_secondary():
        nonlocal secondary_iter
        try:
            return next(secondary_iter)
        except StopIteration:
            secondary_iter = iter(secondary_loader)
            return next(secondary_iter)

    for primary_batch in primary_loader:
        optimizer.zero_grad()
        loss  = None
        x_u   = None   # may stay None when pseudo_loader is absent
        info_u = {}

        if pseudo_is_primary:
            x_u, y_u, info_u = primary_batch
            x_l, y_l, _      = _next_secondary()
        else:
            x_l, y_l, _ = primary_batch
            if secondary_iter is not None:
                x_u, y_u, info_u = _next_secondary()

        # Supervised loss
        if x_l.size(0) >= 2:
            x_l = x_l.to(device)
            y_l = y_l.to(device).unsqueeze(1)
            loss_s = sup_criterion(model(x_l), y_l)
            loss = loss_s
            sup_total += loss_s.item() * x_l.size(0)
            sup_n += x_l.size(0)

        # Unsupervised loss (per-sample weighted when weights are available)
        if x_u is not None and x_u.size(0) >= 2:
            x_u   = x_u.to(device)
            y_u   = y_u.to(device).unsqueeze(1)
            preds_u = model(x_u)
            weight_u = info_u.get("weight") if isinstance(info_u, dict) else None
            if weight_u is not None:
                w = weight_u.to(device).unsqueeze(1)       # (B, 1)
                per_sample = F.binary_cross_entropy(
                    preds_u, y_u, reduction="none"
                )                                          # (B, 1)
                loss_u = unsup_criterion.weight * (per_sample * w).mean()
            else:
                loss_u = unsup_criterion(preds_u, y_u)
            loss = (loss + loss_u) if loss is not None else loss_u
            unsup_total += loss_u.item() * x_u.size(0)
            unsup_n += x_u.size(0)

        if loss is not None:
            loss.backward()
            optimizer.step()

    avg_sup   = sup_total   / sup_n   if sup_n   > 0 else 0.0
    avg_unsup = unsup_total / unsup_n if unsup_n > 0 else 0.0
    return avg_sup, avg_unsup


def fixmatch_train(
    model,
    labeled_loader: DataLoader,
    unlabeled_dataset,
    val_loader: DataLoader,
    optimizer,
    device,
    run_name: str,
    num_epochs: int = 30,
    threshold: float = 0.95,
    lambda_u: float = 1.0,
    batch_size: int = 32,
    scheduler=None,
    threshold_decision: float = 0.5,
    verbose: bool = True,
    output_dir: str = "outputs/fixmatch",
    weak_aug=None,
    strong_aug=None,
    per_dataset_threshold: bool = False,
    min_pseudo_tau: float = 0.5,
    ema_alpha: float = 0.7,
    use_pseudo_weights: bool = True,
    no_dynamic_threshold: bool = False,
    max_pseudo_coverage: float = 0.8,
) -> list[dict]:
    """Full FixMatch training loop with per-epoch validation and checkpointing.

    Each epoch:
      1. Generate pseudo-labels from unlabeled data (generate_pseudo_labels).
      2. Combined epoch – _fixmatch_epoch() pairs each labeled mini-batch with
         a pseudo-labeled mini-batch and sums sup_loss + λ·unsup_loss before
         a single backward pass (canonical FixMatch).
         Unsupervised term is skipped when no pseudo-labeled samples exist.
      3. evaluate() on val_loader; save best checkpoint by val accuracy.

    Args:
        model:              MAMP_MLP to train.
        labeled_loader:     DataLoader for labeled split.
        unlabeled_dataset:  HiddenStateDataset for unlabeled split.
        val_loader:         DataLoader for validation split.
        optimizer:          e.g. AdamW(model.parameters(), lr=1e-3).
        device:             torch device.
        run_name:           Prefix for saved files.
        num_epochs:         Total training epochs.
        threshold:          Confidence threshold τ for pseudo-labeling (0-1).
        lambda_u:           Weight applied to the unsupervised loss term.
        batch_size:         Batch size for pseudo-label generation & training.
        scheduler:          Optional LR scheduler (stepped once per epoch).
        threshold_decision: Decision boundary for binary prediction (default 0.5).
        verbose:            Print per-epoch metrics.
        output_dir:         Directory for saving the best model checkpoint.
        weak_aug:           Weak augmentation callable – required.
        strong_aug:         Strong augmentation callable – required.
        min_pseudo_tau:     Floor applied to all FlexMatch thresholds to prevent
                            tau → 0 runaway collapse (default 0.5).
        ema_alpha:          EMA weight on the previous epoch's pseudo-label
                            counts when updating FlexMatch state. Higher values
                            slow adaptation and stabilise thresholds (default 0.7).
        use_pseudo_weights: If True (default), scale each pseudo-label's
                            unsupervised loss by a per-sample weight derived
                            from MLP confidence, LLM refusal agreement, and
                            normalised perplexity.  If False, all pseudo-labels
                            are weighted uniformly (weight = 1).
        no_dynamic_threshold: If True, disable FlexMatch curriculum and use
                            the fixed `threshold` value for both tau_pos and
                            tau_neg every epoch (plain FixMatch behaviour).
                            Also disables per_dataset_threshold.
                            Default False (FlexMatch enabled).

    Returns:
        history: List of per-epoch dicts with keys:
                 epoch, sup_loss, unsup_loss, total_loss, n_pseudo,
                 and all keys from evaluate().
    """
    if weak_aug is None or strong_aug is None:
        raise ValueError("Both weak_aug and strong_aug must be provided.")

    sup_criterion   = nn.BCELoss()
    unsup_criterion = _ScaledBCELoss(weight=lambda_u)

    best_val_accuracy = 0.0
    os.makedirs(output_dir, exist_ok=True)
    history: list[dict] = []

    # FlexMatch curriculum: track pseudo-label counts per class across epochs.
    # Initialise to 1.0 (float) so both classes start at the base threshold on
    # epoch 1.  EMA smoothing is applied each epoch to avoid wild swings.
    # Global counts (used when per_dataset_threshold=False)
    fm_pos_count: float = 1.0
    fm_neg_count: float = 1.0
    # Per-dataset counts (used when per_dataset_threshold=True)
    # {dataset_name: (pos_count, neg_count)}  — floats for EMA
    fm_ds_counts: dict[str, tuple[float, float]] = {}

    def _flexmatch_tau(pos_c: float, neg_c: float) -> tuple[float, float]:
        """FlexMatch concave threshold mapping from class counts.

        Applies min_pseudo_tau as a floor so neither threshold can collapse
        towards zero, preventing the noisy pseudo-label feedback loop.
        """
        _m  = max(pos_c, neg_c)
        b_p = pos_c / _m
        b_n = neg_c / _m
        tau_p = threshold * b_p / (2.0 - b_p)
        tau_n = threshold * b_n / (2.0 - b_n)
        return (max(tau_p, min_pseudo_tau),
                max(tau_n, min_pseudo_tau))

    frozen_pseudo = None  # cached pseudo-labels once coverage >= max_pseudo_coverage

    for epoch in range(1, num_epochs + 1):

        # ── Compute per-class thresholds ──────────────────────────────────
        if no_dynamic_threshold:
            # Plain FixMatch: fixed threshold every epoch.
            tau_pos = tau_neg = threshold
            ds_thresholds = None
        else:
            # FlexMatch curriculum:
            # beta_c = count_c / max(count_pos, count_neg) ∈ [0, 1]
            # tau_c  = tau * beta_c / (2 - beta_c)  (concave; equals tau at beta=1)
            tau_pos, tau_neg = _flexmatch_tau(fm_pos_count, fm_neg_count)

            if per_dataset_threshold:
                # Each dataset gets its own (tau_pos, tau_neg) from its own counts;
                # unseen datasets fall back to the global tau_pos / tau_neg.
                ds_thresholds = {
                    ds: _flexmatch_tau(p, n)
                    for ds, (p, n) in fm_ds_counts.items()
                }
            else:
                ds_thresholds = None

        # ── Step 1: generate pseudo-labels from unlabeled data ────────────
        if frozen_pseudo is not None:
            # Coverage already hit max_pseudo_coverage; reuse frozen labels.
            pseudo_samples, pseudo_stats = frozen_pseudo
            if verbose:
                print(f"  [Frozen] Reusing {len(pseudo_samples)} pseudo-labels "
                      f"(coverage capped at {max_pseudo_coverage:.0%})")
        else:
            pseudo_samples, pseudo_stats = generate_pseudo_labels(
                model=model,
                unlabeled_dataset=unlabeled_dataset,
                device=device,
                threshold_pos=tau_pos,
                threshold_neg=tau_neg,
                weak_aug=weak_aug,
                strong_aug=strong_aug,
                batch_size=batch_size,
                ds_thresholds=ds_thresholds,
                use_pseudo_weights=use_pseudo_weights,
            )
            pseudo_stats["threshold_pos"] = tau_pos
            pseudo_stats["threshold_neg"] = tau_neg

            # Store per-dataset thresholds in stats for logging / history
            if per_dataset_threshold and ds_thresholds:
                for ds, (tp, tn) in ds_thresholds.items():
                    if ds in pseudo_stats["per_dataset"]:
                        pseudo_stats["per_dataset"][ds]["threshold_pos"] = tp
                        pseudo_stats["per_dataset"][ds]["threshold_neg"] = tn

            # Freeze pseudo-labels once coverage reaches the cap.
            if pseudo_stats["coverage"] >= max_pseudo_coverage:
                frozen_pseudo = (pseudo_samples, pseudo_stats)
                if verbose:
                    print(f"  [Frozen] Coverage {pseudo_stats['coverage']:.1%} >= "
                          f"{max_pseudo_coverage:.0%} — freezing pseudo-labels")

        # ── Update FlexMatch counts for next epoch (EMA-smoothed) ────────
        # Skipped entirely when no_dynamic_threshold=True (counts unused).
        if not no_dynamic_threshold:
            new_pos = float(max(pseudo_stats["n_pseudo_pos"], 1))
            new_neg = float(max(pseudo_stats["n_pseudo_neg"], 1))
            fm_pos_count = ema_alpha * fm_pos_count + (1.0 - ema_alpha) * new_pos
            fm_neg_count = ema_alpha * fm_neg_count + (1.0 - ema_alpha) * new_neg
            if per_dataset_threshold:
                for ds, s in pseudo_stats["per_dataset"].items():
                    old_p, old_n = fm_ds_counts.get(ds, (1.0, 1.0))
                    new_p = float(max(s["n_pseudo_pos"], 1))
                    new_n = float(max(s["n_pseudo_neg"], 1))
                    fm_ds_counts[ds] = (
                        ema_alpha * old_p + (1.0 - ema_alpha) * new_p,
                        ema_alpha * old_n + (1.0 - ema_alpha) * new_n,
                    )

        # ── Steps 2 & 3: combined supervised + unsupervised epoch ────────
        # Both losses are summed before backward so a single gradient update
        # is applied per mini-batch (canonical FixMatch behaviour).
        pseudo_loader = build_pseudo_loader(pseudo_samples, batch_size) if pseudo_samples else None
        sup_loss, unsup_loss = _fixmatch_epoch(
            model, labeled_loader, pseudo_loader,
            optimizer, sup_criterion, unsup_criterion, device,
        )

        total_loss = sup_loss + unsup_loss

        # ── Step 4: validate ──────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device, threshold=threshold_decision)

        if scheduler is not None:
            scheduler.step()

        # ── Checkpoint best model ─────────────────────────────────────────
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            ckpt_path = os.path.join(output_dir, f"{run_name}_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # print(f"Model saved to {ckpt_path}")
            # print("-" * 100)

        record = {
            "epoch":        epoch,
            "sup_loss":     sup_loss,
            "unsup_loss":   unsup_loss,
            "total_loss":   total_loss,
            "pseudo_stats": pseudo_stats,
            **val_metrics,
        }
        history.append(record)

        if verbose:
            print(
                f"Epoch {epoch:>3}/{num_epochs} | "
                f"tau_pos={tau_pos:.4f}  tau_neg={tau_neg:.4f} | "
                f"sup={sup_loss:.4f}  unsup={unsup_loss:.4f}  total={total_loss:.4f} | "
                f"pseudo={pseudo_stats['n_pseudo']:>5}({pseudo_stats['coverage']:.1%}) | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"f1={val_metrics['f1']:.4f} | "
                f"refusal_safe={val_metrics['refusal_rate_safe']:.4f} | "
                f"refusal_unsafe={val_metrics['refusal_rate_unsafe']:.4f}"
            )
            _print_pseudo_stats(pseudo_stats)

    return history
