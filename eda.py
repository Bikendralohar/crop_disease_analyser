"""
eda.py
------
Exploratory Data Analysis script.
Run this BEFORE training to understand the dataset.

Usage:
    python eda.py
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

DATA_DIR = "data/PlantDisease"
SAVE_DIR = "outputs/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_class_distribution(data_dir):
    """Bar chart of how many images exist per class."""
    classes = sorted(os.listdir(data_dir))
    counts  = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]

    total = sum(counts)
    print(f"\n{'='*50}")
    print(f"  Total classes : {len(classes)}")
    print(f"  Total images  : {total}")
    print(f"  Min per class : {min(counts)}  ({classes[counts.index(min(counts))]})")
    print(f"  Max per class : {max(counts)}  ({classes[counts.index(max(counts))]})")
    print(f"  Avg per class : {total // len(classes)}")
    print(f"{'='*50}\n")

    plt.figure(figsize=(22, 7))
    colors = sns.color_palette("viridis", len(classes))
    bars = plt.bar(classes, counts, color=colors)
    plt.axhline(y=sum(counts)/len(counts), color="red",
                linestyle="--", label=f"Mean ({sum(counts)//len(counts)} imgs)")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution — Plant Disease Dataset", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/class_distribution.png", dpi=150)
    plt.show()
    print(f"[EDA] Saved class distribution to {SAVE_DIR}/class_distribution.png")


def show_sample_images(data_dir, n_classes=6, n_per_class=4):
    """Shows sample images from random classes."""
    classes   = sorted(os.listdir(data_dir))
    selected  = random.sample(classes, min(n_classes, len(classes)))

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))

    for i, cls in enumerate(selected):
        cls_path = os.path.join(data_dir, cls)
        images   = os.listdir(cls_path)[:n_per_class]
        for j, img_name in enumerate(images):
            img = mpimg.imread(os.path.join(cls_path, img_name))
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                # Wrap long class names onto two lines
                label = cls.replace("___", "\n").replace("_", " ")
                axes[i, j].set_ylabel(label, fontsize=8, rotation=0,
                                      labelpad=80, va="center")

    plt.suptitle("Sample Images per Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[EDA] Saved sample images to {SAVE_DIR}/sample_images.png")


def plot_plant_vs_disease(data_dir):
    """Pie chart: healthy vs diseased images."""
    classes   = sorted(os.listdir(data_dir))
    healthy   = sum(len(os.listdir(os.path.join(data_dir, c)))
                    for c in classes if "healthy" in c.lower())
    diseased  = sum(len(os.listdir(os.path.join(data_dir, c)))
                    for c in classes if "healthy" not in c.lower())

    plt.figure(figsize=(7, 7))
    plt.pie(
        [healthy, diseased],
        labels=["Healthy", "Diseased"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    plt.title("Healthy vs Diseased Image Ratio", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/healthy_vs_diseased.png", dpi=150)
    plt.show()
    print(f"[EDA] Saved pie chart to {SAVE_DIR}/healthy_vs_diseased.png")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"[EDA] ERROR: Data directory not found at '{DATA_DIR}'")
        print("[EDA] Please extract your Kaggle dataset to data/PlantDisease/")
        exit(1)

    print("[EDA] Starting Exploratory Data Analysis...")
    plot_class_distribution(DATA_DIR)
    show_sample_images(DATA_DIR)
    plot_plant_vs_disease(DATA_DIR)
    print("\n[EDA] All plots saved to outputs/plots/")
