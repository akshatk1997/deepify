"""
DEEPGUARD AI - Forensic Metrics & Dataset Analytics Engine
===========================================================
Calculates dataset distributions, class balances, classification metrics,
and generates forensic visualization figures.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/web server execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def compute_dataset_statistics(base_dir: str) -> Dict[str, Any]:
    """
    Scans dataset/ directory and calculates sample distributions across splits and classes.
    """
    base_path = Path(base_dir)
    splits = ["train", "validation", "test"]
    classes = ["real", "fake"]
    
    stats: Dict[str, Any] = {
        "splits": {},
        "total_images": 0,
        "total_real": 0,
        "total_fake": 0,
        "class_balance": {}
    }

    for s in splits:
        stats["splits"][s] = {"real": 0, "fake": 0, "total": 0}
        for c in classes:
            folder = base_path / s / c
            if folder.exists():
                count = len([f for f in folder.glob("*") if f.is_file() and not f.name.startswith(".")])
            else:
                count = 0
            stats["splits"][s][c] = count
            stats["splits"][s]["total"] += count
            
            if c == "real":
                stats["total_real"] += count
            else:
                stats["total_fake"] += count

        stats["total_images"] += stats["splits"][s]["total"]

    if stats["total_images"] > 0:
        stats["class_balance"]["real_pct"] = round((stats["total_real"] / stats["total_images"]) * 100, 2)
        stats["class_balance"]["fake_pct"] = round((stats["total_fake"] / stats["total_images"]) * 100, 2)
    else:
        stats["class_balance"]["real_pct"] = 0.0
        stats["class_balance"]["fake_pct"] = 0.0

    return stats


def compute_class_weights(stats: Dict[str, Any]) -> Dict[int, float]:
    """
    Calculate inverse class frequencies to balance cross-entropy loss during training.
    0: REAL, 1: FAKE
    """
    train_real = stats["splits"]["train"]["real"]
    train_fake = stats["splits"]["train"]["fake"]
    total = train_real + train_fake

    if total == 0 or train_real == 0 or train_fake == 0:
        return {0: 1.0, 1: 1.0}

    weight_real = (1.0 / train_real) * (total / 2.0)
    weight_fake = (1.0 / train_fake) * (total / 2.0)

    return {0: float(weight_real), 1: float(weight_fake)}


def plot_dataset_distribution(stats: Dict[str, Any], output_path: Optional[str] = None) -> plt.Figure:
    """
    Renders a dark-navy styled bar chart of dataset distribution across Train/Val/Test splits.
    """
    splits = ["Train", "Validation", "Test"]
    real_counts = [
        stats["splits"]["train"]["real"],
        stats["splits"]["validation"]["real"],
        stats["splits"]["test"]["real"]
    ]
    fake_counts = [
        stats["splits"]["train"]["fake"],
        stats["splits"]["validation"]["fake"],
        stats["splits"]["test"]["fake"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0a0e17")
    ax.set_facecolor("#0f172a")

    x = np.arange(len(splits))
    width = 0.35

    rects1 = ax.bar(x - width/2, real_counts, width, label="REAL (Authentic)", color="#10b981", edgecolor="#34d399")
    rects2 = ax.bar(x + width/2, fake_counts, width, label="FAKE (Manipulated)", color="#ef4444", edgecolor="#f87171")

    ax.set_ylabel("Face Image Count", color="#94a3b8", fontsize=11)
    ax.set_title("DEEPGUARD AI – Dataset Partition & Class Distribution", color="#00d2ff", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, color="#f1f5f9", fontsize=11)
    ax.tick_params(axis="y", colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9")
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)

    # Attach count labels above bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        color="#f1f5f9", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes rigorous performance evaluation metrics.
    Assumes Class 0 = REAL, Class 1 = FAKE.
    """
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Optional[str] = None) -> plt.Figure:
    """
    Renders a cybersecurity dark-themed confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0a0e17")
    ax.set_facecolor("#0f172a")

    labels = ["REAL (0)", "FAKE (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "#ffffff"}
    )

    ax.set_xlabel("Predicted Class", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Class", color="#94a3b8", fontsize=11)
    ax.set_title("DEEPGUARD AI – Confusion Matrix", color="#00d2ff", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#f1f5f9")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
