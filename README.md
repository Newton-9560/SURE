# SURE: Semi-supervised Uncertainty-aware REfusal

## Setup

```bash
pip install torch transformers safetensors tqdm termcolor fastchat openai together
```

## Step 1: Save Hidden States

Extract hidden states from a local LLM for all dataset samples:

```bash
python inference.py --model_name llama3 -m "save hidden states"
```

Hidden states are saved to `outputs/hidden_states/llama3/`.

## Step 2: Train (FixMatch, multi-seed)

```bash
python train_fixmatch_multiseed.py \
  -m llama3 \
  --layer 17 \
  --labeled_train_size 80 \
  --per_dataset_threshold \
  --min_pseudo_tau 0.7 \
  --epochs 15 \
  --warmup_epochs 20
```

Results are saved to `outputs/fixmatch_multiseed/`.

## Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | `llama3` | Model name (`llama3`, `llama3_70b`, `mistral`) |
| `--layer` | `17` | Transformer layer for hidden states |
| `--labeled_train_size` | `80` | Number of labeled training samples |
| `--per_dataset_threshold` | `False` | Enable per-dataset adaptive thresholds |
| `--min_pseudo_tau` | `0.8` | Floor for FlexMatch thresholds |
| `--threshold` | `0.95` | Base confidence threshold for pseudo-labels |
| `--epochs` | `15` | FixMatch training epochs |
| `--warmup_epochs` | `20` | Supervised warmup epochs |

## Project Structure

```
SURE/
├── train_fixmatch_multiseed.py   # Main entry: multi-seed training
├── inference.py                  # Hidden state extraction & LLM inference
├── MAMP/
│   ├── model.py                  # MLP classifier
│   ├── trainer.py                # Train/evaluate/predict
│   ├── fixmatch.py               # FixMatch + FlexMatch training loop
│   └── dataset.py                # HiddenStateDataset
├── dataset/
│   ├── main.py                   # SafetyDataset loader
│   └── formatted/                # Dataset JSON files
├── utils/
│   ├── utils.py                  # Helpers (args_to_str, parse_size_list, etc.)
│   ├── augmentations.py          # Weak/strong augmentations for hidden states
│   └── summary.py                # Evaluation metrics & LaTeX table output
└── outputs/                      # All outputs (hidden states, predictions, checkpoints)
```
