"""
predict.py
----------
Single-image inference with Test-Time Augmentation (TTA).
Averages predictions across multiple crops/flips for better
real-world robustness.

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --image path/to/leaf.jpg --no-tta       # single-pass only
    python predict.py --image path/to/leaf.jpg --threshold 60  # custom confidence threshold
"""

import torch
import argparse
import os
from PIL import Image
from torchvision import transforms

from src.model import build_model

# If confidence is below this %, warn the user
DEFAULT_CONFIDENCE_THRESHOLD = 50.0


# ── Transforms ────────────────────────────────────────────────────────────────

def get_base_transform():
    """Standard single-pass transform matching training val transform."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),        # Center crop, not just resize — closer to training data
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_tta_transforms():
    """
    5 transforms for Test-Time Augmentation.
    Predictions are averaged across all of them to improve robustness
    on real-world images that differ from training distribution.
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        # 1. Standard center crop
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(), norm
        ]),
        # 2. Direct resize (no crop)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), norm
        ]),
        # 3. Horizontal flip + center crop
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm
        ]),
        # 4. Slight brightness boost (helps with dark/shadowed images)
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(), norm
        ]),
        # 5. Slight saturation boost (helps with faded/low-contrast images)
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(saturation=0.3),
            transforms.ToTensor(), norm
        ]),
    ]


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_single(image, model, device, transform):
    """Run a single forward pass and return softmax probabilities."""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
    return probs


def predict(image_path, model, class_names, device, use_tta=True, threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Runs inference on a single image, optionally with TTA.

    Args:
        image_path  : path to the input image
        model       : loaded trained model
        class_names : list of class name strings
        device      : 'cuda' or 'cpu'
        use_tta     : if True, average over 5 augmented versions
        threshold   : confidence % below which a warning is shown

    Returns:
        predicted class name (str), confidence (float)
    """
    image = Image.open(image_path).convert("RGB")

    if use_tta:
        tta_transforms = get_tta_transforms()
        all_probs = [predict_single(image, model, device, t) for t in tta_transforms]
        probs = torch.stack(all_probs).mean(0)
    else:
        probs = predict_single(image, model, device, get_base_transform())

    top5_probs, top5_idx = probs.topk(5, dim=1)
    confidence  = top5_probs[0][0].item() * 100
    pred_class  = class_names[top5_idx[0][0].item()]

    # ── Display ─────────────────────────────────────────────────────────────
    mode_label = "TTA (5-pass)" if use_tta else "Single-pass"
    print(f"\n── Top 5 Predictions [{mode_label}] ──────────────────────")
    for i in range(min(5, len(class_names))):
        cls  = class_names[top5_idx[0][i].item()]
        prob = top5_probs[0][i].item() * 100
        bar  = "█" * int(prob // 5)
        print(f"  {cls:<50} {prob:5.1f}%  {bar}")
    print("─" * 70)

    # ── Confidence warning ───────────────────────────────────────────────────
    if confidence < threshold:
        print(f"\n⚠️  Low confidence ({confidence:.1f}% < {threshold:.0f}% threshold)")
        print("   Possible reasons:")
        print("   • Image has a complex/noisy background")
        print("   • Leaf is partially visible or at unusual angle")
        print("   • Disease type may not be well-represented in training data")
        print("   • Try cropping the image to isolate a single leaf")
    else:
        print(f"\n✅ Predicted: {pred_class}")
        print(f"   Confidence: {confidence:.1f}%")

    return pred_class, confidence


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, num_classes, device):
    """Loads the trained model from a checkpoint file."""
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image (JPG, PNG, WEBP)")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--data_dir",   type=str, default="data/PlantDisease")
    parser.add_argument("--no-tta",     action="store_true",
                        help="Disable Test-Time Augmentation (faster but less robust)")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Confidence %% below which a warning is shown (default: 50)")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = sorted(os.listdir(args.data_dir))
    num_classes = len(class_names)

    print(f"[Predict] Device      : {device}")
    print(f"[Predict] Classes     : {num_classes}")
    print(f"[Predict] Image       : {args.image}")
    print(f"[Predict] Checkpoint  : {args.checkpoint}")
    print(f"[Predict] TTA         : {'disabled' if args.no_tta else 'enabled (5-pass)'}")

    model = load_model(args.checkpoint, num_classes, device)
    predict(args.image, model, class_names, device,
            use_tta=not args.no_tta,
            threshold=args.threshold)