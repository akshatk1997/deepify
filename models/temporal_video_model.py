"""
DEEPGUARD AI - Temporal Video Deepfake Classification Model
============================================================
Learns temporal sequence dynamics of video deepfakes:
1. Inter-frame probability velocity & jitter
2. Temporal variance and anomaly burst dynamics
3. Boundary flickering and facial consistency
4. Sequence-level confidence calibration
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DEFAULT_TEMPORAL_MODEL_PATH = SAVED_MODELS_DIR / "temporal_video_model.json"


class TemporalVideoClassifier:
    """
    Supervised sequence classifier for video-level deepfake verification.
    Combines frame-level probability distributions, temporal velocity,
    inter-frame frequency variance, and burst anomaly metrics.
    """

    FEATURE_NAMES = [
        "mean_fake_prob",
        "max_burst_prob",
        "top_k_burst_prob",
        "temporal_variance",
        "temporal_velocity",
        "manipulated_ratio",
        "flicker_index",
        "continuity_score"
    ]

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_TEMPORAL_MODEL_PATH
        self.is_trained = False
        self.weights = np.array([2.4, 2.8, 3.2, 1.8, 2.2, 3.0, 1.5, -2.0], dtype=np.float32)
        self.bias = -4.2
        self.decision_threshold = 0.50
        self.metrics = {}

        if self.model_path.exists():
            self.load(self.model_path)

    @classmethod
    def extract_sequence_features(cls, frame_predictions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts 8-dimensional temporal feature vector from sampled frame predictions.
        """
        valid_frames = [f for f in frame_predictions if f.get("has_face", False)]
        if not valid_frames:
            return np.zeros(len(cls.FEATURE_NAMES), dtype=np.float32)

        fake_probs = np.array([float(f.get("fake_probability", 0.0)) for f in valid_frames], dtype=np.float32)
        n = len(fake_probs)

        # 1. Central tendency
        mean_prob = float(np.mean(fake_probs))
        max_prob = float(np.max(fake_probs))

        # 2. Top-K 25% burst average
        k = max(1, int(math.ceil(n * 0.25)))
        top_k = float(np.mean(np.sort(fake_probs)[-k:]))

        # 3. Temporal variance & standard deviation
        variance = float(np.var(fake_probs))

        # 4. Temporal velocity (inter-frame probability fluctuation)
        if n > 1:
            diffs = np.abs(np.diff(fake_probs))
            velocity = float(np.mean(diffs))
        else:
            velocity = 0.0

        # 5. Manipulated frame ratio (frames >= 0.50)
        manip_ratio = float(np.mean(fake_probs >= 0.50))

        # 6. Flicker index (second-order temporal acceleration)
        if n > 2:
            second_diffs = np.abs(np.diff(np.diff(fake_probs)))
            flicker = float(np.mean(second_diffs))
        else:
            flicker = 0.0

        # 7. Continuity score (higher for stable organic camera motion)
        continuity = float(max(0.0, 1.0 - (velocity * 2.2 + flicker * 1.5)))

        return np.array([
            mean_prob,
            max_prob,
            top_k,
            variance,
            velocity,
            manip_ratio,
            flicker,
            continuity
        ], dtype=np.float32)

    def predict_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Runs calibrated logistic inference on 8-dim feature vector.
        """
        z = float(np.dot(features, self.weights) + self.bias)
        # Numerically stable sigmoid
        if z >= 0:
            prob_fake = float(1.0 / (1.0 + math.exp(-z)))
        else:
            exp_z = math.exp(z)
            prob_fake = float(exp_z / (1.0 + exp_z))

        prob_real = float(1.0 - prob_fake)
        is_fake = prob_fake >= self.decision_threshold
        verdict = "FAKE" if is_fake else "REAL"
        confidence = round((prob_fake if is_fake else prob_real) * 100.0, 2)

        if is_fake:
            risk_tier = "CRITICAL" if prob_fake > 0.85 else "HIGH"
        else:
            risk_tier = "LOW (AUTHENTIC)" if prob_real > 0.85 else "MODERATE"

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_probability": round(prob_fake, 4),
            "real_probability": round(prob_real, 4),
            "risk_tier": risk_tier,
            "feature_vector": {name: round(float(val), 4) for name, val in zip(self.FEATURE_NAMES, features)},
            "model_type": "Temporal Biometric Sequence Classifier (TBSC)"
        }

    def predict_video_timeline(self, frame_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts features from video frame timeline and produces calibrated prediction.
        """
        features = self.extract_sequence_features(frame_predictions)
        return self.predict_features(features)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.05,
        epochs: int = 150,
        l2_reg: float = 0.01
    ) -> Dict[str, Any]:
        """
        Trains the temporal classifier using regularized logistic regression with gradient descent.
        """
        num_samples, num_features = X.shape
        w = np.zeros(num_features, dtype=np.float32)
        b = 0.0

        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X, w) + b
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

            # Gradients
            err = preds - y
            dw = (np.dot(X.T, err) / num_samples) + (l2_reg * w)
            db = float(np.mean(err))

            # Update
            w -= learning_rate * dw
            b -= learning_rate * db

        self.weights = w
        self.bias = b
        self.is_trained = True

        # Compute empirical training metrics
        final_logits = np.dot(X, w) + b
        final_preds = 1.0 / (1.0 + np.exp(-np.clip(final_logits, -20.0, 20.0)))
        pred_labels = (final_preds >= 0.50).astype(int)

        accuracy = float(np.mean(pred_labels == y))
        tp = float(np.sum((pred_labels == 1) & (y == 1)))
        fp = float(np.sum((pred_labels == 1) & (y == 0)))
        fn = float(np.sum((pred_labels == 0) & (y == 1)))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        self.metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "num_training_sequences": num_samples
        }
        return self.metrics

    def save(self, path: Optional[Path] = None):
        """Serializes model parameters and weights to JSON."""
        target_path = Path(path) if path else self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "Temporal Biometric Sequence Classifier (TBSC)",
            "version": "1.2.0",
            "feature_names": self.FEATURE_NAMES,
            "weights": [round(float(x), 6) for x in self.weights],
            "bias": round(float(self.bias), 6),
            "decision_threshold": self.decision_threshold,
            "metrics": self.metrics
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Temporal video classifier saved to: {target_path}")

    def load(self, path: Path):
        """Loads model parameters and weights from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.weights = np.array(data["weights"], dtype=np.float32)
        self.bias = float(data["bias"])
        self.decision_threshold = float(data.get("decision_threshold", 0.50))
        self.metrics = data.get("metrics", {})
        self.is_trained = True


def train_temporal_video_model(output_path: Optional[Path] = None) -> TemporalVideoClassifier:
    """
    Generates realistic video sequence training tracks and fits the temporal model.
    """
    print("=" * 75)
    print("DEEPGUARD AI - TEMPORAL VIDEO MODEL TRAINING")
    print("=" * 75)

    np.random.seed(42)
    sequences_X = []
    sequences_y = []

    # 1. Generate 50 authentic video sequence trajectories
    # Authentic videos: low probabilities (0.01 - 0.25), smooth transitions, high continuity
    for i in range(50):
        length = np.random.randint(15, 40)
        base_prob = np.random.uniform(0.02, 0.15)
        # Gaussian walk with small step size
        probs = np.clip(base_prob + np.random.normal(0, 0.03, length), 0.01, 0.35)
        mock_timeline = [{"has_face": True, "fake_probability": float(p)} for p in probs]
        feats = TemporalVideoClassifier.extract_sequence_features(mock_timeline)
        sequences_X.append(feats)
        sequences_y.append(0)

    # 2. Generate 50 deepfake video sequence trajectories
    # Deepfake videos: recurring high burst spikes, jitter, high variance
    for i in range(50):
        length = np.random.randint(15, 40)
        # Deepfake mode A: continuous manipulation
        if i % 2 == 0:
            base_prob = np.random.uniform(0.70, 0.95)
            probs = np.clip(base_prob + np.random.normal(0, 0.08, length), 0.40, 0.99)
        # Deepfake mode B: selective burst manipulation (e.g. face swap every few frames)
        else:
            probs = np.random.uniform(0.10, 0.30, length)
            burst_start = np.random.randint(0, length // 2)
            burst_len = np.random.randint(4, length - burst_start)
            probs[burst_start:burst_start + burst_len] = np.random.uniform(0.80, 0.98, burst_len)
            # Add synthetic flicker
            probs[::2] += np.random.uniform(0.05, 0.15, len(probs[::2]))
            probs = np.clip(probs, 0.05, 0.99)

        mock_timeline = [{"has_face": True, "fake_probability": float(p)} for p in probs]
        feats = TemporalVideoClassifier.extract_sequence_features(mock_timeline)
        sequences_X.append(feats)
        sequences_y.append(1)

    X = np.array(sequences_X, dtype=np.float32)
    y = np.array(sequences_y, dtype=np.float32)

    classifier = TemporalVideoClassifier()
    metrics = classifier.train(X, y, epochs=200, learning_rate=0.08, l2_reg=0.005)
    classifier.save(output_path or DEFAULT_TEMPORAL_MODEL_PATH)

    print(f"Temporal Model Training Results on 100 Video Sequences:")
    print(f"  • Sequence Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Sequence Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  • Sequence Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  • Sequence F1-Score:  {metrics['f1_score'] * 100:.2f}%")
    print("=" * 75)
    return classifier


if __name__ == "__main__":
    train_temporal_video_model()
