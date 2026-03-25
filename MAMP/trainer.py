import torch
import torch.nn as nn
import os


def _compute_metrics(all_labels, all_preds):
    """
    Compute classification metrics for a binary safety classifier.

    Convention
    ----------
    label / pred = 1  →  safe   (no refusal)
    label / pred = 0  →  unsafe (refusal)

    "Positive" class for precision / recall / F1 is **unsafe** (the class we
    want to detect).  This gives:
        TP  unsafe correctly refused
        FP  safe incorrectly refused     (over-refusal)
        TN  safe correctly passed
        FN  unsafe incorrectly passed    (missed detection)

    Extra metrics
    -------------
    refusal_rate_safe   = FP / (FP + TN)   fraction of safe samples refused
    refusal_rate_unsafe = TP / (TP + FN)   fraction of unsafe samples refused
                                            (== recall for unsafe class)
    """
    n = len(all_labels)
    tp = tn = fp = fn = 0
    for y, p in zip(all_labels, all_preds):
        if y == 0 and p == 0:   # unsafe, refused  → TP
            tp += 1
        elif y == 1 and p == 0: # safe,   refused  → FP
            fp += 1
        elif y == 1 and p == 1: # safe,   passed   → TN
            tn += 1
        else:                   # unsafe, passed   → FN
            fn += 1

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    n_safe   = tn + fp
    n_unsafe = tp + fn
    refusal_rate_safe   = fp / n_safe   if n_safe   > 0 else 0.0
    refusal_rate_unsafe = tp / n_unsafe if n_unsafe > 0 else 0.0

    return {
        "accuracy":            accuracy,
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "refusal_rate_safe":   refusal_rate_safe,
        "refusal_rate_unsafe": refusal_rate_unsafe,
        # raw counts (useful for debugging)
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_safe": n_safe, "n_unsafe": n_unsafe,
    }



@torch.no_grad()
def predict(model, loader, device, threshold=0.5):
    """Run inference and return a list of per-sample result dicts.

    Each dict contains: dataset_name, prompt, label (true), refusal (predicted
    unsafe), confidence.  Compatible with summary.summarize_file.
    """
    model.eval()
    results = []
    for x, y, info in loader:
        probs = model(x.to(device)).squeeze(1).cpu()
        for j in range(x.size(0)):
            prob = probs[j].item()
            true_safe = bool(y[j].item())
            results.append({
                "dataset_name": info["dataset_name"][j],
                "prompt":       info["prompt"][j],
                "label":        "safe" if true_safe else "unsafe",
                "refusal":      prob < threshold,
                "confidence":   round(prob, 6),
            })
    return results


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch.  Returns average loss over the epoch."""
    model.train()
    total_loss = 0.0

    for x, y, _ in loader:
        if x.size(0) < 2:
            continue
        x, y = x.to(device), y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Evaluate the model on a dataset.

    Args:
        model:     MAMP_MLP (or any model with sigmoid output in [0, 1]).
        loader:    DataLoader yielding (x, y, info) batches.
                   y = 1.0 for safe, 0.0 for unsafe.
        device:    torch device.
        threshold: decision boundary; pred >= threshold → safe (1).

    Returns:
        dict with keys: accuracy, precision, recall, f1,
                        refusal_rate_safe, refusal_rate_unsafe,
                        tp, fp, tn, fn, n_safe, n_unsafe, loss.
    """
    model.eval()
    criterion = nn.BCELoss()

    all_labels, all_preds = [], []
    total_loss = 0.0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        probs = model(x).squeeze(1)          # (B,)
        total_loss += criterion(probs, y).item() * x.size(0)

        preds = (probs >= threshold).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.long().cpu().tolist())

    metrics = _compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    run_name,
    num_epochs=20,
    criterion=None,
    scheduler=None,
    threshold=0.5,
    verbose=True,
    output_dir='outputs/mamp',
):
    """
    Full training loop with per-epoch evaluation on the validation set.

    Args:
        model:        MAMP_MLP instance.
        train_loader: DataLoader for the labeled training set.
        val_loader:   DataLoader for the validation set.
        optimizer:    e.g. torch.optim.AdamW(model.parameters(), lr=1e-3).
        device:       torch device.
        num_epochs:   number of training epochs.
        criterion:    loss function (default: BCELoss).
        scheduler:    optional LR scheduler (step called once per epoch).
        threshold:    decision boundary for binary prediction.
        verbose:      print metrics each epoch.

    Returns:
        history: list of dicts, one per epoch, each containing
                 train_loss and all val metrics.
    """
    if criterion is None:
        criterion = nn.BCELoss()

    history = []
    best_val_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, threshold=threshold)

        if scheduler is not None:
            scheduler.step()

        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        if verbose:
            print(
                f"Epoch {epoch:>3}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"acc={val_metrics['accuracy']:.4f} | "
                f"prec={val_metrics['precision']:.4f} | "
                f"rec={val_metrics['recall']:.4f} | "
                f"f1={val_metrics['f1']:.4f} | "
                f"refusal_safe={val_metrics['refusal_rate_safe']:.4f} | "
                f"refusal_unsafe={val_metrics['refusal_rate_unsafe']:.4f}"
            )
        if val_metrics['accuracy'] > best_val_accuracy and epoch > 1:
            best_val_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_model.pt"))
            print(f"Model saved to {os.path.join(output_dir, f'{run_name}_model.pt')}")
            print('-'*100)

    return history
