"""
DEEPIFY - Model Training & Evaluation Engine
==================================================
End-to-end training pipeline for XceptionNet / Depthwise Separable deepfake detection.
Executes dual-stage transfer learning + fine-tuning, generates evaluation
artifacts (loss/accuracy curves, confusion matrix, ROC-AUC, classification reports),
and exports the validated Keras model.
"""

import os
import sys

# Disable buggy MKL custom memory primitives to ensure 100% stable execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# Ensure project root is accessible
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.xception_model import build_xception_deepfake_model, unfreeze_for_fine_tuning
from utils.metrics import (
    compute_dataset_statistics,
    compute_class_weights,
    evaluate_predictions,
    plot_confusion_matrix
)

# Constants
TARGET_SIZE: Tuple[int, int] = (299, 299)
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_data_pipeline(
    data_dir: Path,
    batch_size: int = 8,
    img_size: Tuple[int, int] = TARGET_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]:
    """
    Constructs high-performance tf.data pipelines for Train, Validation, and Test splits.
    Applies on-the-fly normalization: [-1.0, 1.0].
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing required dataset directory: {d}")

    # Standard class ordering: 0 = 'real', 1 = 'fake'
    class_names = ["real", "fake"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Forensic data augmentation layer (applied only on training set)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.04)
    ], name="data_augmentation")

    # Xception normalization: (pixel / 127.5) - 1.0
    normalization_layer = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="xception_norm")

    # Optimize pipeline with prefetching
    train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def plot_training_history(history_stage1, history_stage2, output_path: Path):
    """
    Renders clean loss and accuracy curves across both training stages.
    """
    acc = history_stage1.history["accuracy"]
    val_acc = history_stage1.history["val_accuracy"]
    loss = history_stage1.history["loss"]
    val_loss = history_stage1.history["val_loss"]

    if history_stage2 is not None:
        acc += history_stage2.history["accuracy"]
        val_acc += history_stage2.history["val_accuracy"]
        loss += history_stage2.history["loss"]
        val_loss += history_stage2.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#ffffff")
    ax1.set_facecolor("#f8fafc")
    ax2.set_facecolor("#f8fafc")

    # Accuracy Plot
    ax1.plot(epochs, acc, color="#2563eb", lw=2.2, label="Train Accuracy")
    ax1.plot(epochs, val_acc, color="#059669", lw=2.2, linestyle="--", label="Val Accuracy")
    if history_stage2 is not None:
        stage1_end = len(history_stage1.history["accuracy"])
        ax1.axvline(x=stage1_end, color="#d97706", linestyle=":", label="Fine-Tuning Start")
    ax1.set_title("Model Accuracy Across Epochs", color="#0f172a", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs", color="#475569")
    ax1.set_ylabel("Accuracy", color="#475569")
    ax1.tick_params(colors="#334155")
    ax1.grid(color="#e2e8f0", linestyle="--", alpha=0.8)
    ax1.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a")

    # Loss Plot
    ax2.plot(epochs, loss, color="#dc2626", lw=2.2, label="Train Loss")
    ax2.plot(epochs, val_loss, color="#d97706", lw=2.2, linestyle="--", label="Val Loss")
    if history_stage2 is not None:
        stage1_end = len(history_stage1.history["loss"])
        ax2.axvline(x=stage1_end, color="#d97706", linestyle=":", label="Fine-Tuning Start")
    ax2.set_title("Model Loss Across Epochs", color="#0f172a", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs", color="#475569")
    ax2.set_ylabel("Binary Crossentropy Loss", color="#475569")
    ax2.tick_params(colors="#334155")
    ax2.grid(color="#e2e8f0", linestyle="--", alpha=0.8)
    ax2.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def train_deepguard_model(
    data_dir: Path = BASE_DIR / "dataset",
    output_model_path: Path = SAVED_MODELS_DIR / "xception_deepfake.keras",
    epochs_stage1: int = 6,
    epochs_stage2: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    fine_tune_lr: float = 2e-4
) -> Dict[str, Any]:
    """
    Executes the dual-stage training protocol.
    """
    print("=" * 75)
    print("DEEPIFY - FORENSIC NEURAL MODEL TRAINING PIPELINE")
    print("=" * 75)

    # 1. Inspect dataset distribution and class weights
    stats = compute_dataset_statistics(str(data_dir))
    class_weights = compute_class_weights(stats)
    print(f"Dataset summary: {stats['total_images']} total images.")
    print(f"Computed Class Weights: {class_weights}")

    # 2. Build tf.data datasets
    train_ds, val_ds, test_ds, class_names = create_data_pipeline(data_dir, batch_size=batch_size)

    # 3. Instantiate architecture
    print("\n[STAGE 1] Initializing Forensic Depthwise Architecture...")
    model = build_xception_deepfake_model(learning_rate=learning_rate, force_compact=True)
    model.summary()

    # Define training callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(output_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # 4. Train Stage 1
    print(f"\n[STAGE 1] Training forensic feature extractor for {epochs_stage1} epochs...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
    )

    history2 = None
    if epochs_stage2 > 0:
        # 5. Train Stage 2: Fine-Tuning
        print(f"\n[STAGE 2] Fine-tuning depthwise layers for {epochs_stage2} epochs...")
        model = unfreeze_for_fine_tuning(model, fine_tune_lr=fine_tune_lr)
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage2,
            class_weight=class_weights,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
        )

    # Ensure best weights are loaded, or explicitly save current model
    if output_model_path.exists():
        print(f"\nLoading best checkpointed model weights from: {output_model_path}")
        try:
            model = tf.keras.models.load_model(str(output_model_path))
        except Exception as e:
            print(f"[WARN] Checkpoint reload notice: {e}. Saving current trained model graph.")
            model.save(str(output_model_path))
    else:
        print(f"\nSaving trained model to: {output_model_path}")
        model.save(str(output_model_path))

    # 6. Evaluation on Test Partition
    print("\n[EVALUATION] Evaluating model performance on held-out test partition...")
    y_true_list = []
    y_pred_probs_list = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true_list.extend(batch_y.numpy().flatten())
        y_pred_probs_list.extend(preds.flatten())

    y_true = np.array(y_true_list)
    y_probs = np.array(y_pred_probs_list)

    metrics_result = evaluate_predictions(y_true, y_probs, threshold=0.5)

    print("\n" + "=" * 50)
    print("TEST EVALUATION PERFORMANCE:")
    print("=" * 50)
    print(f"Accuracy:  {metrics_result['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics_result['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics_result['recall'] * 100:.2f}%")
    print(f"F1-Score:  {metrics_result['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC:   {metrics_result['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics_result['confusion_matrix'])}")

    # 7. Save Evaluation Artifacts
    history_png = SAVED_MODELS_DIR / "training_history.png"
    plot_training_history(history1, history2, history_png)
    print(f"Training curves saved to: {history_png}")

    cm_png = SAVED_MODELS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(np.array(metrics_result["confusion_matrix"]), str(cm_png))
    print(f"Confusion matrix plot saved to: {cm_png}")

    metrics_json = SAVED_MODELS_DIR / "model_metrics.json"
    metrics_data = {
        "architecture": "Deepify Forensic Depthwise CNN (Extreme Inception)",
        "input_shape": [299, 299, 3],
        "preprocessing": "[-1.0, 1.0] scaling",
        "classes": {"0": "REAL", "1": "FAKE"},
        "test_metrics": metrics_result
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Evaluation metrics JSON saved to: {metrics_json}")

    report_txt = SAVED_MODELS_DIR / "classification_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("DEEPIFY - MODEL PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture: Deepify Forensic Depthwise CNN\n")
        f.write(f"Accuracy:     {metrics_result['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision:    {metrics_result['precision'] * 100:.2f}%\n")
        f.write(f"Recall:       {metrics_result['recall'] * 100:.2f}%\n")
        f.write(f"F1-Score:     {metrics_result['f1_score'] * 100:.2f}%\n")
        f.write(f"ROC-AUC:      {metrics_result['roc_auc']:.4f}\n")
    print(f"Classification text report saved to: {report_txt}")

    return metrics_result


train_deepify_model = train_deepguard_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepify Model Training")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to preprocessed dataset")
    parser.add_argument("--epochs1", type=int, default=6, help="Stage 1 epochs")
    parser.add_argument("--epochs2", type=int, default=3, help="Stage 2 epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train_deepify_model(
        data_dir=Path(args.data_dir),
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        batch_size=args.batch_size
    )
