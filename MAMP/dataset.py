import json
import math
import os

import torch
from torch.utils.data import Dataset
from safetensors.numpy import load_file

# Resolve project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dataset.main import DATASET_MAP, FORMATTED_DIR, _LABEL_NORM


# ---------------------------------------------------------------------------
# Perplexity normalisation helpers
# ---------------------------------------------------------------------------

# Supported normalisation modes for perplexity values.
#   "none"       – return the raw value unchanged.
#   "minmax"     – linear rescale to [0, 1] using global min / max.
#   "log_minmax" – apply log1p first, then linear rescale to [0, 1].
#   "sigmoid"    – sigmoid centred at the global mean with a configurable scale.
PERPLEXITY_NORM_MODES = ("none", "minmax", "log_minmax", "sigmoid")


def _compute_perplexity_stats(uncertainty_data: dict) -> dict:
    """Compute global perplexity statistics from the full uncertainty dict.

    Iterates over all datasets / samples to find the global min, max, mean,
    and their log1p counterparts so that any normalisation mode can be applied
    at item-retrieval time without an extra pass.
    """
    vals = []
    for ds_entries in uncertainty_data.values():
        for methods in ds_entries.values():
            v = methods.get("Perplexity")
            if v is not None:
                vals.append(float(v))

    if not vals:
        return {"min": 0.0, "max": 1.0, "mean": 0.5,
                "log_min": 0.0, "log_max": 1.0}

    p_min  = min(vals)
    p_max  = max(vals)
    p_mean = sum(vals) / len(vals)
    log_vals = [math.log1p(v) for v in vals]
    return {
        "min":     p_min,
        "max":     p_max,
        "mean":    p_mean,
        "log_min": min(log_vals),
        "log_max": max(log_vals),
    }


def normalise_perplexity(
    value: float,
    stats: dict,
    mode: str = "minmax",
    sigmoid_scale: float = 1.0,
) -> float:
    """Normalise a single perplexity value to [0, 1].

    Args:
        value:         Raw perplexity value.
        stats:         Dict returned by _compute_perplexity_stats().
        mode:          One of PERPLEXITY_NORM_MODES.
        sigmoid_scale: Steepness of the sigmoid (only used when mode="sigmoid").

    Returns:
        Float in [0, 1] (or the raw value for mode="none").
    """
    if mode == "none":
        return value

    if mode == "minmax":
        denom = stats["max"] - stats["min"]
        return (value - stats["min"]) / denom if denom > 0 else 0.0

    if mode == "log_minmax":
        lv    = math.log1p(value)
        denom = stats["log_max"] - stats["log_min"]
        return (lv - stats["log_min"]) / denom if denom > 0 else 0.0

    if mode == "sigmoid":
        # Sigmoid centred at global mean; scale controls steepness.
        return 1.0 / (1.0 + math.exp(-sigmoid_scale * (value - stats["mean"])))

    raise ValueError(
        f"Unknown perplexity_norm mode '{mode}'. "
        f"Choose from {PERPLEXITY_NORM_MODES}."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HiddenStateDataset(Dataset):
    """Dataset that serves pre-saved hidden states as (x, y) pairs.

    Args:
        data_list:         list of dicts identifying each sample.
        model_name:        str, e.g. "llama3" – selects the hidden-state dir.
        layer:             int – which transformer layer to use as features.
        uncertainty_file:  path to the uncertainty JSON
                           ({dataset_name: {str(id): {method: float}}}).
        response_file:     path to the response JSON (list of dicts with at
                           least 'dataset_name', 'id', and 'refusal' keys).
        perplexity_norm:   normalisation mode for Perplexity values.
                           One of: "none", "minmax", "log_minmax", "sigmoid".
        perplexity_sigmoid_scale:
                           Steepness of sigmoid (only used when
                           perplexity_norm="sigmoid").
    """

    def __init__(
        self,
        data_list,
        model_name,
        layer,
        uncertainty_file=os.path.join(
            _PROJECT_ROOT, "outputs", "uncertainty_safe_only_1_token.json"
        ),
        response_file=os.path.join(
            _PROJECT_ROOT, "outputs",
            "llama3_responses_figtxt_wildjailbreak_vanilla_wildjailbreak_adversarial_jbb_behaviors_8800_generations_10.json"
        ),
        perplexity_norm: str = "minmax",
        perplexity_sigmoid_scale: float = 1.0,
    ):
        if perplexity_norm not in PERPLEXITY_NORM_MODES:
            raise ValueError(
                f"perplexity_norm must be one of {PERPLEXITY_NORM_MODES}, "
                f"got '{perplexity_norm}'."
            )
        self.layer                   = layer
        self.perplexity_norm         = perplexity_norm
        self.perplexity_sigmoid_scale = perplexity_sigmoid_scale

        hs_dir = os.path.join(_PROJECT_ROOT, "outputs", "hidden_states", model_name)

        # ── Uncertainty ────────────────────────────────────────────────────
        with open(uncertainty_file, "r", encoding="utf-8") as f:
            self.uncertainty_data = json.load(f)

        # Pre-compute normalisation statistics from the full uncertainty file.
        self.perplexity_stats = _compute_perplexity_stats(self.uncertainty_data)

        # ── Response / refusal index ───────────────────────────────────────
        # Build a fast lookup: (dataset_name, str(id)) → refusal (bool | None)
        self._refusal_index: dict[tuple[str, str], bool | None] = {}
        if response_file and os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                responses = json.load(f)
            for r in responses:
                key = (r["dataset_name"], str(r["id"]))
                self._refusal_index[key] = r.get("refusal")

        # ── Label index ────────────────────────────────────────────────────
        needed_datasets = {entry["dataset_name"] for entry in data_list}
        label_index = {}
        for ds_name in needed_datasets:
            meta_path = os.path.join(FORMATTED_DIR, DATASET_MAP[ds_name])
            with open(meta_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for e in entries:
                norm = _LABEL_NORM.get(e["label"], e["label"])
                label_index[(ds_name, e["id"])] = (norm == "safe")

        # ── Sample list ────────────────────────────────────────────────────
        self.samples = []
        for entry in data_list:
            dataset_name = entry["dataset_name"]
            entry_id     = entry["id"]
            hs_path = os.path.join(hs_dir, f"{dataset_name}_{entry_id}.safetensors")
            if not os.path.exists(hs_path):
                continue
            label = label_index.get((dataset_name, entry_id))
            if label is None:
                continue
            self.samples.append({
                "hs_path":      hs_path,
                "label":        label,
                "id":           entry_id,
                "prompt":       entry["prompt"],
                "category":     entry["category"],
                "dataset_name": dataset_name,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample       = self.samples[idx]
        dataset_name = sample["dataset_name"]
        entry_id     = sample["id"]

        hs  = load_file(sample["hs_path"])["hidden_states"]
        x   = torch.tensor(hs[self.layer], dtype=torch.float32)
        y   = torch.tensor(float(sample["label"]), dtype=torch.float32)

        # ── Uncertainty ────────────────────────────────────────────────────
        raw_uncertainty = self.uncertainty_data.get(dataset_name, {}).get(
            str(entry_id), None
        )
        uncertainty = None
        if raw_uncertainty is not None:
            uncertainty = dict(raw_uncertainty)   # shallow copy
            perp = uncertainty.get("Perplexity")
            if perp is not None:
                uncertainty["Perplexity"] = normalise_perplexity(
                    perp,
                    self.perplexity_stats,
                    mode=self.perplexity_norm,
                    sigmoid_scale=self.perplexity_sigmoid_scale,
                )

        # ── Refusal ────────────────────────────────────────────────────────
        refusal = self._refusal_index.get((dataset_name, str(entry_id)), None)

        info = {
            "prompt":       sample["prompt"],
            "category":     sample["category"],
            "dataset_name": dataset_name,
            "uncertainty":  uncertainty,
            "refusal":      refusal,
        }
        return x, y, info
