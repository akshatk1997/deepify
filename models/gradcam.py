"""
DEEPGUARD AI - Explainable AI (Grad-CAM) Engine
================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to expose and visualize the spatial convolutional features in facial imagery
that drive XceptionNet's deepfake detection predictions.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from utils.image_processing import (
    preprocess_face_for_model,
    resize_image,
    TARGET_SIZE
)


class GradCAMExplainer:
    """
    Computes class activation heatmaps by backpropagating target logit gradients
    to the final depthwise separable convolutional activations of XceptionNet.
    """

    def __init__(self, model: Any, target_layer_name: Optional[str] = None):
        """
        Args:
            model: Trained Keras model instance.
            target_layer_name: Name of target convolutional feature layer.
                               Defaults to Xception's 'block14_sepconv2_act'.
        """
        import tensorflow as tf
        self.tf = tf
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_conv_layer()
        self.grad_model = self._build_grad_model()

    def _find_target_conv_layer(self) -> str:
        """Locates the final convolutional activation layer in XceptionNet."""
        # Standard target layers in Xception
        candidate_names = [
            "block14_sepconv2_act",
            "block14_sepconv2",
            "block14_sepconv1_act",
            "conv2d"
        ]
        layer_names = [layer.name for layer in self.model.layers]

        for cand in candidate_names:
            if cand in layer_names:
                return cand

        # Fallback: find the last layer with 4D output tensor (Conv/Activation)
        for layer in reversed(self.model.layers):
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name

        raise ValueError("Could not automatically locate a 4D convolutional feature layer in model.")

    def _build_grad_model(self):
        """Constructs an auxiliary model returning both target activations and final logits."""
        target_conv_layer = self.model.get_layer(self.target_layer_name)
        return self.tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_conv_layer.output, self.model.output]
        )

    def generate_heatmap(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Calculates normalized 2D Grad-CAM heatmap for a given face crop.

        Args:
            face_rgb: Cropped face array in RGB format of shape (H, W, 3).

        Returns:
            Normalized 2D float32 heatmap of shape (H, W) in range [0.0, 1.0].
        """
        h_orig, w_orig = face_rgb.shape[:2]
        input_tensor = preprocess_face_for_model(face_rgb, TARGET_SIZE)

        with self.tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_tensor)
            # Binary classification logit: probability of FAKE
            loss = predictions[:, 0]

        # Compute gradients of the loss with respect to convolutional feature maps
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            # Return uniform attention if gradients cannot be resolved
            return np.ones((h_orig, w_orig), dtype=np.float32) * 0.5

        # Global average pooling of gradients gives feature channel importance weights (alpha)
        pooled_grads = self.tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature channels by their respective gradients
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., self.tf.newaxis]
        heatmap = self.tf.squeeze(heatmap)

        # Apply ReLU to retain only features that positively influence the prediction
        heatmap = self.tf.maximum(heatmap, 0.0) / (self.tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # Resize heatmap to match the original facial crop dimensions
        resized_heatmap = cv2.resize(heatmap_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return np.clip(resized_heatmap, 0.0, 1.0)

    @staticmethod
    def overlay_heatmap(
        face_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays the normalized thermal heatmap onto the original RGB face image.

        Args:
            face_rgb: Original face crop uint8 RGB.
            heatmap: 2D float32 heatmap in [0.0, 1.0].
            alpha: Transparency factor for heatmap overlay (0.0 to 1.0).
            colormap: OpenCV colormap constant (default JET for cold-to-hot spectrum).

        Returns:
            Tuple of (colorized_heatmap_rgb, superimposed_overlay_rgb).
        """
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Apply thermal color mapping (BGR)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Superimpose onto original face image
        superimposed = (heatmap_rgb.astype(np.float32) * alpha) + (face_rgb.astype(np.float32) * (1.0 - alpha))
        superimposed_rgb = np.clip(superimposed, 0, 255).astype(np.uint8)

        return heatmap_rgb, superimposed_rgb

    def explain(self, face_rgb: np.ndarray, alpha: float = 0.45) -> Dict[str, Any]:
        """
        Comprehensive forensic Explainable AI report for a face crop.

        Returns:
            Dict containing:
                - 'target_layer': Name of explained convolutional block
                - 'raw_heatmap': 2D float32 array
                - 'heatmap_rgb': Colorized thermal map
                - 'overlay_rgb': Superimposed visualization
                - 'dominant_regions': Textual description of activated focus areas
        """
        h_orig, w_orig = face_rgb.shape[:2]
        heatmap = self.generate_heatmap(face_rgb)
        heatmap_rgb, overlay_rgb = self.overlay_heatmap(face_rgb, heatmap, alpha=alpha)

        # Analyze dominant activation quadrants
        mid_y, mid_x = h_orig // 2, w_orig // 2
        quad_top = np.mean(heatmap[:mid_y, :])
        quad_bottom = np.mean(heatmap[mid_y:, :])
        quad_center = np.mean(heatmap[int(h_orig * 0.25):int(h_orig * 0.75), int(w_orig * 0.25):int(w_orig * 0.75)])
        quad_perimeter = (np.mean(heatmap) * (h_orig * w_orig) - quad_center * (mid_y * mid_x)) / (h_orig * w_orig - mid_y * mid_x + 1e-6)

        dominant_features = []
        if quad_perimeter > quad_center * 1.15:
            dominant_features.append("Facial Boundary & Jawline Seams (blending discontinuity focus)")
        if quad_top > quad_bottom * 1.20:
            dominant_features.append("Ocular & Forehead Region (gaze / eyebrow synthesis focus)")
        elif quad_bottom > quad_top * 1.20:
            dominant_features.append("Mouth & Chin Region (lip-sync / audio-visual mismatch focus)")
        else:
            dominant_features.append("Central Facial Symmetry & Skin Texture Frequency")

        return {
            "target_layer": self.target_layer_name,
            "raw_heatmap": heatmap,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "dominant_regions": dominant_features
        }
