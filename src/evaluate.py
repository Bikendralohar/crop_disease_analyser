"""
evaluate.py
-----------
Evaluates the trained model on the test set.
Outputs classification report, confusion matrix, and saves plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def evaluate_model(model, test_loader, class_names, device, save_dir="outputs/plots"):
    """
    Runs inference on the test set and prints/saves evaluation results.

    Args:
        model        : trained model (best checkpoint loaded)
        test_loader  : test DataLoader
        class_names  : list of class name strings
        device       : 'cuda' or 'cpu'
        save_dir     : folder to save plots
    """
    model.eval()
    all_preds  = []
    all_labels = []

    print("[Eval] Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Testing"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification Report ────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── Confusion Matrix ─────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    fig_size = max(16, len(class_names) // 2)
    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )
    plt.title("Confusion Matrix — Test Set", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.show()
    print(f"[Eval] Confusion matrix saved to {save_dir}/confusion_matrix.png")

    # ── Per-class Accuracy Bar Chart ─────────────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(20, 6))
    colors = ["#e74c3c" if a < 85 else "#2ecc71" for a in per_class_acc]
    plt.bar(class_names, per_class_acc, color=colors)
    plt.axhline(y=90, color="gray", linestyle="--", label="90% threshold")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/per_class_accuracy.png", dpi=150)
    plt.show()
    print(f"[Eval] Per-class accuracy plot saved to {save_dir}/per_class_accuracy.png")

    # ── Summary ──────────────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n[Eval] Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"[Eval] Worst class : {class_names[per_class_acc.argmin()]} ({per_class_acc.min():.1f}%)")
    print(f"[Eval] Best class  : {class_names[per_class_acc.argmax()]} ({per_class_acc.max():.1f}%)")
