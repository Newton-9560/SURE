import json
import os
import sys
import random
import copy
from fastchat.model import get_conversation_template
try:
    from ue.lookup import lookup_uncertainty, UNCERTAINTY_METHOD_LIST
except ImportError:
    lookup_uncertainty = None
    UNCERTAINTY_METHOD_LIST = []
from tqdm import tqdm

DATASET_LIST = ["figtxt", "xstest", "repe", "wildjailbreak_vanilla", "wildjailbreak_adversarial", "jbb_behaviors", "strongreject"]

DATASET_MAP = {
    "figtxt": "figtxt_all_eval.json",
    "xstest": "xstest_all_eval.json",
    "repe": "repe_test_all_eval.json",
    "wildjailbreak_vanilla": "wildjailbreak_vanilla_all_eval.json",
    "wildjailbreak_adversarial": "wildjailbreak_adversarial_all_eval.json",
    "jbb_behaviors": "jbb_behaviors_all_eval.json",
    "strongreject": "strongreject_all_eval.json",
}

FORMATTED_DIR = os.path.join(os.path.dirname(__file__), "formatted")

_MODEL_TEMPLATE_MAP = {
    "llama3": {
        "openai_model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "huggingface_model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "template_name": "llama-3",
        "provider": "deepinfra",
        "provider_api_key": os.getenv("DEEPINFRA_API_KEY"),
        "provider_base_url": "https://api.deepinfra.com/v1/openai",
    },
    "qwen_7b": {
        "openai_model_path": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "huggingface_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "template_name": "qwen-7b-chat",
        "provider": "together",
        "provider_api_key": os.getenv("TOGETHER_API_KEY"),
        "provider_base_url": None,
    },
    "mistral": {
        "openai_model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "huggingface_model_path": "mistralai/Mistral-Small-24B-Instruct-2501",
        "template_name": "mistral",
        "provider": "deepinfra",
        "provider_api_key": os.getenv("DEEPINFRA_API_KEY"),
        "provider_base_url": "https://api.deepinfra.com/v1/openai",
    },
    "deepseek-v3": {
        "openai_model_path": "deepseek-ai/DeepSeek-V3.2",
        "huggingface_model_path": "deepseek-ai/DeepSeek-V3.2",
        "template_name": "deepseek-v3",
        "provider": "deepinfra",
        "provider_api_key": os.getenv("DEEPINFRA_API_KEY"),
        "provider_base_url": "https://api.deepinfra.com/v1/openai",
    },
    "llama3_70b": {
        "openai_model_path": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "huggingface_model_path": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "template_name": "llama-3.1",
        "provider": "deepinfra",
        "provider_api_key": os.getenv("DEEPINFRA_API_KEY"),
        "provider_base_url": "https://api.deepinfra.com/v1/openai",
    },
}

# Normalize heterogeneous label strings to "safe" / "unsafe"
_LABEL_NORM = {
    "safe": "safe",
    "unsafe": "unsafe",
    "harmless": "safe",
    "harmful": "unsafe",
}


class SafetyDataset:
    def __init__(self, labeled_size, val_size, dataset_name_list=None, seed=42):
        """
        Load all datasets, shuffle each independently with the given seed, then
        split per-dataset into:
          - labeled:   first labeled_size[i] samples from dataset i
          - val:       last val_size[i] samples from dataset i
          - unlabeled: everything in between

        Args:
            labeled_size: list of ints, one per dataset in dataset_name_list,
                          specifying how many labeled samples to take from each.
            val_size:     list of ints, one per dataset in dataset_name_list,
                          specifying how many val samples to take from each.
            dataset_name_list: list of dataset names to load (default: DATASET_LIST).
            seed: random seed for shuffling.
        """
        self.seed = seed
        self.data = {}

        if dataset_name_list is None:
            self.dataset_name_list = DATASET_LIST
        else:
            self.dataset_name_list = dataset_name_list

        if len(labeled_size) != len(self.dataset_name_list):
            raise ValueError(
                f"labeled_size length ({len(labeled_size)}) must match "
                f"dataset_name_list length ({len(self.dataset_name_list)})"
            )
        if len(val_size) != len(self.dataset_name_list):
            raise ValueError(
                f"val_size length ({len(val_size)}) must match "
                f"dataset_name_list length ({len(self.dataset_name_list)})"
            )

        self.labeled_size = labeled_size
        self.val_size = val_size

        labeled_split, unlabeled_split, val_split = [], [], []
        for i, name in enumerate(self.dataset_name_list):
            path = os.path.join(FORMATTED_DIR, DATASET_MAP[name])
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            samples = [
                {
                    "id": entry["id"],
                    "prompt": entry["prompt"],
                    "label": _LABEL_NORM[entry["label"]],
                    "category": entry.get("category", entry.get("type", "")),
                    "dataset_name": name,
                }
                for entry in raw
            ]

            l, u, v = self._split_dataset(samples, labeled_size[i], val_size[i])
            labeled_split.extend(l)
            unlabeled_split.extend(u)
            val_split.extend(v)

        self.data = {"labeled": labeled_split, "unlabeled": unlabeled_split, "val": val_split}

    def _split_dataset(self, dataset, labeled_size, val_size):
        """
        Split a single dataset's samples into labeled, unlabeled, and val sets.
          - labeled:   first labeled_size samples (after shuffle)
          - val:       last val_size samples (after shuffle)
          - unlabeled: everything in between
        """
        rng = random.Random(self.seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        n = len(dataset)

        labeled_size = (
            max(1, int(labeled_size * n)) if labeled_size < 1 else int(labeled_size)
        )
        val_size = (
            max(1, int(val_size * n)) if val_size < 1 else int(val_size)
        )

        if labeled_size + val_size > n:
            raise ValueError(
                f"labeled_size ({labeled_size}) + val_size ({val_size}) "
                f"= {labeled_size + val_size} exceeds dataset size ({n})"
            )

        labeled_indices = indices[:labeled_size]
        val_indices = indices[n - val_size:]
        unlabeled_indices = indices[labeled_size:n - val_size]

        labeled_split = [dict(dataset[i], split="labeled") for i in labeled_indices]
        unlabeled_split = [dict(dataset[i], split="unlabeled") for i in unlabeled_indices]
        val_split = [dict(dataset[i], split="val") for i in val_indices]

        return labeled_split, unlabeled_split, val_split

    @staticmethod
    def format_prompt(prompt, model_name):
        """Given a raw prompt string, return the model-specific formatted message.

        Supported model_name: "llama3", "qwen", "mistral".
        """

        if model_name not in _MODEL_TEMPLATE_MAP:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: {list(_MODEL_TEMPLATE_MAP.keys())}"
            )

        template_name = _MODEL_TEMPLATE_MAP[model_name]["template_name"]

        # Build llama-3 prompt manually for correct formatting
        if template_name == "llama-3":
            formatted_prompt = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            return formatted_prompt
        else:
            conv = copy.deepcopy(get_conversation_template(template_name))
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()

    def get_dataset(self, dataset_name_list=None, split=None, label="all", dataset_balanced=False, label_balanced=False):
        """
        Get the dataset
        Args:
            dataset_name_list: list of dataset names to get
            split: split to get
            label: label to get
            dataset_balanced: if True, truncate each dataset to the size of the
                              smallest one so every dataset contributes equally.
            label_balanced:   if True, within each dataset equalise safe and
                              unsafe counts by truncating the larger class.
                              Datasets with no safe samples are left untouched.
        Returns:
            list of samples: {
                "id": int,
                "prompt": str,
                "label": str,
                "category": str,
                "dataset_name": str,
                "split": str,
                "uncertainty": dict,
            }
        """
        if dataset_name_list is None:
            dataset_name_list = DATASET_LIST

        splits = [split] if split in ("labeled", "unlabeled", "val") else ["labeled", "unlabeled", "val"]

        result = []
        for s in splits:
            for entry in self.data[s]:
                if entry["dataset_name"] in dataset_name_list:
                    result.append(entry)

        if label != "all":
            result = [entry for entry in result if entry["label"] == label]

        if label_balanced:
            # Within each dataset, equalise safe and unsafe counts.
            # Datasets with no safe samples are left untouched.
            by_dataset: dict[str, dict[str, list]] = {}
            for entry in result:
                ds = entry["dataset_name"]
                lbl = entry["label"]
                by_dataset.setdefault(ds, {"safe": [], "unsafe": []})
                by_dataset[ds][lbl].append(entry)

            result = []
            for ds, labels in by_dataset.items():
                safe_entries   = labels["safe"]
                unsafe_entries = labels["unsafe"]
                if not safe_entries:
                    # No safe samples — skip label balancing for this dataset
                    result.extend(unsafe_entries)
                else:
                    n = min(len(safe_entries), len(unsafe_entries))
                    result.extend(safe_entries[:n])
                    result.extend(unsafe_entries[:n])

        if dataset_balanced:
            # Truncate each dataset to the size of the smallest one
            groups: dict[str, list] = {}
            for entry in result:
                groups.setdefault(entry["dataset_name"], []).append(entry)
            min_count = min(len(v) for v in groups.values()) if groups else 0
            result = []
            for entries in groups.values():
                result.extend(entries[:min_count])

        return result

    def get_uncertainty(self, uncertainty_method_list=UNCERTAINTY_METHOD_LIST):
        """
        Get the uncertainty for the dataset
        Args:
            uncertainty_method_list: list of uncertainty methods
        Returns:
            list of samples: {
                "id": int,
                "prompt": str,
                "label": str,
                "category": str,
                "dataset_name": str,
                "split": str,
                "uncertainty": dict,
            }
        """
        for entry in tqdm(self.data["labeled"]):
            entry["uncertainty"] = lookup_uncertainty(entry["dataset_name"], entry["id"], uncertainty_method_list)
        for entry in tqdm(self.data["unlabeled"]):
            entry["uncertainty"] = lookup_uncertainty(entry["dataset_name"], entry["id"], uncertainty_method_list)
        for entry in tqdm(self.data["val"]):
            entry["uncertainty"] = lookup_uncertainty(entry["dataset_name"], entry["id"], uncertainty_method_list)
        return self.data

def test_safety_dataset():
    """
    Test the SafetyDataset class
    """
    dataset = SafetyDataset(labeled_size=[100, 100, 100, 100, 100, 100, 100], val_size=[500, 500, 500, 500, 500, 500, 500])
    data = dataset.get_dataset(dataset_name_list=["figtxt", "xstest", "repe"], split="labeled", label="safe")
    for entry in data:
        print(entry)
        print("-" * 100)
    print(f"data_size: {len(data)}")

def test_format_prompt():
    """
    Test the format_prompt function
    """
    prompt = "What is the capital of France?"
    model_name = "llama3"
    formatted_prompt = SafetyDataset.format_prompt(prompt, model_name)
    print(formatted_prompt)
    print("-" * 100)

if __name__ == "__main__":
    # test_safety_dataset()
    test_format_prompt()