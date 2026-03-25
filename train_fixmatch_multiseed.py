"""Run train_fixmatch pipeline over 5 seeds and report mean ± std."""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.main import SafetyDataset, _MODEL_TEMPLATE_MAP
from MAMP import MAMP_MLP, HiddenStateDataset, predict, train
from MAMP.fixmatch import fixmatch_train
from utils.augmentations import weak_aug, strong_aug
from utils.utils import args_to_str, parse_size_list
from utils.summary import summarize_file, summarize_files, summarize_latex, _DATASET_ORDER, _harmonic_mean


SEEDS = [789, 214, 239]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_seed(args, seed):
    """Run the full fixmatch pipeline for one seed. Returns val predictions list."""
    set_random_seed(seed)

    labeled_size = parse_size_list(args.labeled_size)
    val_size = parse_size_list(args.val_size)

    safety_dataset = SafetyDataset(
        labeled_size=labeled_size,
        val_size=val_size,
        dataset_name_list=["figtxt", "wildjailbreak_vanilla",
                           "wildjailbreak_adversarial", "jbb_behaviors"],
        seed=seed,
    )
    train_data = safety_dataset.get_dataset(
        split="labeled", dataset_balanced=True, label_balanced=True
    )[:args.labeled_train_size]
    unlabeled_data = safety_dataset.get_dataset(
        split="unlabeled", dataset_balanced=False
    )
    random.shuffle(unlabeled_data)
    unlabeled_data = unlabeled_data[:2000]
    val_data = safety_dataset.get_dataset(split="val")

    print(f"\n[Seed {seed}] Train: {len(train_data)}  Unlabeled: {len(unlabeled_data)}  Val: {len(val_data)}")

    train_dataset = HiddenStateDataset(train_data, args.model_name, args.layer)
    unlabeled_dataset = HiddenStateDataset(unlabeled_data, args.model_name, args.layer)
    val_dataset = HiddenStateDataset(val_data, args.model_name, args.layer)

    print(f"[Seed {seed}] Train HS: {len(train_dataset)}  Unlabeled HS: {len(unlabeled_dataset)}  Val HS: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x0, _, _ = train_dataset[0]
    input_dim = x0.shape[0]

    model = MAMP_MLP(input_dim=input_dim, dropout_p=args.dropout).to(device)

    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # ── Warm-up ──────────────────────────────────────────────────────────
    warmup_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(warmup_optimizer, T_max=args.warmup_epochs)

    print(f"[Seed {seed}] Warm-up: {args.warmup_epochs} epochs")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=warmup_optimizer,
        device=device,
        run_name="warmup",
        num_epochs=args.warmup_epochs,
        scheduler=warmup_scheduler,
        verbose=True,
        output_dir=seed_dir,
    )

    warmup_ckpt = os.path.join(seed_dir, "warmup_model.pt")
    model.load_state_dict(torch.load(warmup_ckpt, map_location=device))
    print(f"[Seed {seed}] Loaded warm-up checkpoint")

    # ── FixMatch ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = args_to_str(args)
    fixmatch_train(
        model=model,
        labeled_loader=train_loader,
        unlabeled_dataset=unlabeled_dataset,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        run_name=run_name,
        num_epochs=args.epochs,
        threshold=args.threshold,
        lambda_u=args.lambda_u,
        batch_size=args.batch_size,
        scheduler=scheduler,
        verbose=True,
        output_dir=seed_dir,
        weak_aug=weak_aug,
        strong_aug=strong_aug,
        per_dataset_threshold=True,
        min_pseudo_tau=args.min_pseudo_tau,
        ema_alpha=args.ema_alpha,
        use_pseudo_weights=not args.no_pseudo_weights,
        no_dynamic_threshold=args.no_dynamic_threshold,
        max_pseudo_coverage=args.max_pseudo_coverage,
    )

    # ── Evaluate best checkpoint ─────────────────────────────────────────
    best_model_path = os.path.join(seed_dir, f"{run_name}_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    results = predict(model, val_loader, device)

    results_path = os.path.join(seed_dir, f"{args.model_name}_{run_name}_val_predictions.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Seed {seed}] Val predictions saved to {results_path}")
    summarize_file(results_path)

    return results


def compute_metrics_from_predictions(data):
    """Compute per-dataset and overall metrics from a prediction list.
    Returns a dict: {(dataset, label): refusal_rate, ...} plus derived metrics.
    """
    by_dataset_label = defaultdict(list)
    for entry in data:
        ds = entry.get("dataset_name", "unknown")
        label = entry.get("label", "unknown")
        by_dataset_label[(ds, label)].append(entry)

    metrics = {}
    for (ds, label), entries in by_dataset_label.items():
        refusal_rate = sum(1 for e in entries if e["refusal"]) / len(entries) * 100
        metrics[(ds, label)] = refusal_rate

    return metrics


def print_mean_std_table(all_metrics):
    """Print a summary table with mean ± std across seeds."""
    # Collect all (ds, label) keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    # Build per-key arrays
    key_values = defaultdict(list)
    for m in all_metrics:
        for k in all_keys:
            if k in m:
                key_values[k].append(m[k])

    print(f"\n{'='*80}")
    print(f"MEAN ± STD ACROSS {len(all_metrics)} SEEDS")
    print(f"{'='*80}")

    # Per-dataset breakdown
    headers = ["Dataset", "Label", "Refusal Rate", "Correctness"]
    rows = []
    for ds in sorted(set(k[0] for k in all_keys)):
        for label in ["safe", "unsafe"]:
            key = (ds, label)
            if key in key_values:
                vals = np.array(key_values[key])
                ref_mean, ref_std = vals.mean(), vals.std()
                if label == "safe":
                    corr_mean, corr_std = 100 - ref_mean, ref_std
                else:
                    corr_mean, corr_std = ref_mean, ref_std
                rows.append([
                    ds, label,
                    f"{ref_mean:.1f} ± {ref_std:.1f}%",
                    f"{corr_mean:.1f} ± {corr_std:.1f}%",
                ])

    # Overall by label
    for label in ["safe", "unsafe"]:
        all_vals = []
        for ds in sorted(set(k[0] for k in all_keys)):
            key = (ds, label)
            if key in key_values:
                all_vals.append(key_values[key])
        if all_vals:
            # Average across datasets per seed, then mean/std across seeds
            per_seed = np.mean(all_vals, axis=0)
            ref_mean, ref_std = per_seed.mean(), per_seed.std()
            if label == "safe":
                corr_mean, corr_std = 100 - ref_mean, ref_std
            else:
                corr_mean, corr_std = ref_mean, ref_std
            rows.append([
                "OVERALL", label,
                f"{ref_mean:.1f} ± {ref_std:.1f}%",
                f"{corr_mean:.1f} ± {corr_std:.1f}%",
            ])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))

    # ── Mean ± std LaTeX row ─────────────────────────────────────────────
    print_mean_std_latex(all_metrics)


def print_mean_std_latex(all_metrics):
    """Print LaTeX row with mean ± std across seeds."""
    # Collect per-seed values for each cell
    all_safe_refs = defaultdict(list)    # ds -> [seed values]
    all_unsafe_refs = defaultdict(list)

    for m in all_metrics:
        for (ds, label), val in m.items():
            if label == "safe":
                all_safe_refs[ds].append(val)
            else:
                all_unsafe_refs[ds].append(val)

    cells = []
    # Accumulators for average
    avg_safe_per_seed = []
    avg_unsafe_per_seed = []
    avg_acc_per_seed = []
    avg_hm_per_seed = []

    for ds in _DATASET_ORDER:
        safe_vals = np.array(all_safe_refs.get(ds, []))
        unsafe_vals = np.array(all_unsafe_refs.get(ds, []))

        if ds == "jbb_behaviors":
            cells.append("--")
            if len(unsafe_vals) > 0:
                cells.append(f"{unsafe_vals.mean():.1f}$\\pm${unsafe_vals.std():.1f}")
            else:
                cells.append("-")
        else:
            if len(safe_vals) > 0:
                cells.append(f"{safe_vals.mean():.1f}$\\pm${safe_vals.std():.1f}")
            else:
                cells.append("-")
            if len(unsafe_vals) > 0:
                cells.append(f"{unsafe_vals.mean():.1f}$\\pm${unsafe_vals.std():.1f}")
            else:
                cells.append("-")

            if len(safe_vals) > 0 and len(unsafe_vals) > 0:
                safe_correct = 100 - safe_vals
                acc = (safe_correct + unsafe_vals) / 2
                hm = np.array([_harmonic_mean(s, u) for s, u in zip(safe_correct, unsafe_vals)])
                cells.append(f"{acc.mean():.1f}$\\pm${acc.std():.1f}")
                cells.append(f"{hm.mean():.1f}$\\pm${hm.std():.1f}")
            else:
                cells.append("-")
                cells.append("-")

    # Compute per-seed averages for the "Average" columns
    n_seeds = len(all_metrics)
    for i in range(n_seeds):
        m = all_metrics[i]
        safe_refs = []
        unsafe_refs = []
        accs = []
        hms = []
        for ds in _DATASET_ORDER:
            s = m.get((ds, "safe"))
            u = m.get((ds, "unsafe"))
            if ds == "jbb_behaviors":
                if u is not None:
                    unsafe_refs.append(u)
            else:
                if s is not None:
                    safe_refs.append(s)
                if u is not None:
                    unsafe_refs.append(u)
                if s is not None and u is not None:
                    sc = 100 - s
                    accs.append((sc + u) / 2)
                    hms.append(_harmonic_mean(sc, u))

        avg_safe_per_seed.append(np.mean(safe_refs) if safe_refs else 0)
        avg_unsafe_per_seed.append(np.mean(unsafe_refs) if unsafe_refs else 0)
        avg_acc_per_seed.append(np.mean(accs) if accs else 0)
        avg_hm_per_seed.append(np.mean(hms) if hms else 0)

    avg_safe = np.array(avg_safe_per_seed)
    avg_unsafe = np.array(avg_unsafe_per_seed)
    avg_acc = np.array(avg_acc_per_seed)
    avg_hm = np.array(avg_hm_per_seed)

    cells.append(f"{avg_safe.mean():.1f}$\\pm${avg_safe.std():.1f}")
    cells.append(f"{avg_unsafe.mean():.1f}$\\pm${avg_unsafe.std():.1f}")
    cells.append(f"{avg_acc.mean():.1f}$\\pm${avg_acc.std():.1f}")
    cells.append(f"{avg_hm.mean():.1f}$\\pm${avg_hm.std():.1f}")

    print(f"\nLaTeX row (mean±std):\n& " + " & ".join(cells) + " \\\\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_size", nargs="+", default=[59*3, 59*3, 59*3, 25*3],
                        help="Labeled samples per dataset (int) or ratio (float < 1)")
    parser.add_argument("--val_size", nargs="+", default=[0.5, 500, 500, 0.5],
                        help="Val samples per dataset (int) or ratio (float < 1)")
    parser.add_argument("--model_name", "-m", type=str, default="llama3",
                        choices=list(_MODEL_TEMPLATE_MAP.keys()))
    parser.add_argument("--layer", type=int, default=17,
                        help="Transformer layer index to use as feature vector")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42,
                        help="(Unused, seeds are fixed to SEEDS list)")
    parser.add_argument("--output_dir", type=str, default="outputs/fixmatch_multiseed")
    parser.add_argument("--labeled_train_size", "-ls", type=int, default=80)
    # FixMatch-specific
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lambda_u", type=float, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--per_dataset_threshold", action="store_true", default=False)
    parser.add_argument("--min_pseudo_tau", type=float, default=0.8)
    parser.add_argument("--ema_alpha", type=float, default=0)
    parser.add_argument("--no_pseudo_weights", action="store_true", default=False)
    parser.add_argument("--no_dynamic_threshold", action="store_true", default=False)
    parser.add_argument("--max_pseudo_coverage", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    all_predictions = {}
    result_files = []

    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        results = run_single_seed(args, seed)
        metrics = compute_metrics_from_predictions(results)
        all_metrics.append(metrics)
        all_predictions[seed] = results

        # Save per-seed result file
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        run_name = args_to_str(args)
        result_path = os.path.join(seed_dir, f"{args.model_name}_{run_name}_val_predictions.json")
        result_files.append(result_path)

    # Save all predictions
    agg_path = os.path.join(args.output_dir, "all_seeds_predictions.json")
    serializable = {str(k): v for k, v in all_predictions.items()}
    with open(agg_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll predictions saved to {agg_path}")

    # Print mean ± std summary (legacy)
    print_mean_std_table(all_metrics)

    # Print mean ± std summary using summarize_files
    print(f"\n{'='*60}")
    print("Summary (via summarize_files):")
    print(f"{'='*60}")
    summarize_files(result_files)


if __name__ == "__main__":
    main()
