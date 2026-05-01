"""
main.py
-------
Main entry point for training the crop disease detection model.
Runs Phase 1 (head only) then Phase 2 (fine-tuning), then evaluates.

Usage:
    python main.py
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.dataset  import get_loaders
from src.model    import build_model, unfreeze_backbone, count_parameters
from src.train    import train_one_epoch, validate
from src.evaluate import evaluate_model

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "data/PlantDisease"   # Path to extracted Kaggle dataset
BATCH_SIZE  = 32                    # Reduce to 16 if you get CUDA out-of-memory
EPOCHS_P1   = 10                    # Phase 1: train head only
EPOCHS_P2   = 15                    # Phase 2: fine-tune backbone
LR_P1       = 1e-3                  # Higher LR for Phase 1 (head only)
LR_P2       = 1e-4                  # Lower LR for Phase 2 (fine-tuning)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR    = "outputs/checkpoints"
PLOT_DIR    = "outputs/plots"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Crop Disease Detection — Training Script")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")

train_loader, val_loader, test_loader, classes = get_loaders(DATA_DIR, BATCH_SIZE)
num_classes = len(classes)

# Save class names to file (useful for inference later)
with open("outputs/class_names.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")
print(f"[Main] Class names saved to outputs/class_names.txt\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_model(num_classes=num_classes, freeze_backbone=True).to(DEVICE)
count_parameters(model)

# Loss with label smoothing — reduces overconfidence, improves generalisation
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Training History ──────────────────────────────────────────────────────────
history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  []
}

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: Train classification head only (backbone frozen)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  PHASE 1 — Training classifier head ({EPOCHS_P1} epochs)")
print(f"{'='*60}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P1
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_P1)

for epoch in range(EPOCHS_P1):
    print(f"\nEpoch [{epoch+1}/{EPOCHS_P1}]")
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    print(f"  Train — Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
    print(f"  Val   — Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: Fine-tune backbone layers (unfrozen from layer 5 onwards)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  PHASE 2 — Fine-tuning backbone ({EPOCHS_P2} epochs)")
print(f"{'='*60}")

unfreeze_backbone(model, unfreeze_from_layer=5)
count_parameters(model)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P2
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_P2)

best_val_acc = 0.0

for epoch in range(EPOCHS_P2):
    print(f"\nEpoch [{epoch+1}/{EPOCHS_P2}]")
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    print(f"  Train — Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
    print(f"  Val   — Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%")

    # Save best checkpoint
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), f"{CKPT_DIR}/best_model.pth")
        print(f"  ✓ New best model saved (Val Acc: {vl_acc:.2f}%)")

print(f"\n[Main] Best validation accuracy: {best_val_acc:.2f}%")

# ── Plot Training Curves ──────────────────────────────────────────────────────
total_epochs = EPOCHS_P1 + EPOCHS_P2
epochs_range = range(1, total_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#3498db")
ax1.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="#e74c3c")
ax1.axvline(x=EPOCHS_P1, color="gray", linestyle="--", label="Phase 1→2")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss Curve"); ax1.legend()

ax2.plot(epochs_range, history["train_acc"], label="Train Acc", color="#3498db")
ax2.plot(epochs_range, history["val_acc"],   label="Val Acc",   color="#e74c3c")
ax2.axvline(x=EPOCHS_P1, color="gray", linestyle="--", label="Phase 1→2")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Curve"); ax2.legend()

plt.suptitle("Training History — Crop Disease Detection", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/training_curves.png", dpi=150)
plt.show()
print(f"[Main] Training curves saved to {PLOT_DIR}/training_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION on Test Set
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  EVALUATION — Test Set")
print(f"{'='*60}")

# Load the best saved model before evaluating
model.load_state_dict(torch.load(f"{CKPT_DIR}/best_model.pth"))
model.to(DEVICE)

evaluate_model(model, test_loader, classes, DEVICE, save_dir=PLOT_DIR)

print(f"\n[Main] All done! Outputs saved to outputs/")
print(f"  • Best model   : {CKPT_DIR}/best_model.pth")
print(f"  • Plots        : {PLOT_DIR}/")
print(f"  • Class names  : outputs/class_names.txt")
