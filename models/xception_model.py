"""
DEEPGUARD AI - XceptionNet Architecture Definition
===================================================
Constructs the deepfake detection neural network using XceptionNet
(Extreme Inception with Depthwise Separable Convolutions) as feature extractor,
coupled with an enterprise forensic classification head and dual-stage fine-tuning.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import Xception


def build_xception_deepfake_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    dropout_rate: float = 0.5,
    dense_units: int = 512,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Constructs the XceptionNet deepfake detection network.

    Architecture:
    1. Input layer: (299, 299, 3) normalized in [-1, 1]
    2. Base: Xception (ImageNet pre-trained), frozen initially for transfer learning
    3. Global Average Pooling (aggregates spatial convolutional features)
    4. Dense feature projection (512 units + ReLU + L2 regularization)
    5. Batch Normalization & Forensic Dropout (prevents overfitting to specific camera artifacts)
    6. Intermediate Dense (128 units + ReLU)
    7. Output Sigmoid logit: Probability that input face is FAKE (0 = REAL, 1 = FAKE)
    """
    inputs = layers.Input(shape=input_shape, name="input_face")

    # Load pre-trained Xception base without standard ImageNet 1000-class top
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # Freeze base model weights for initial transfer learning stage
    base_model.trainable = False

    # Extract target convolutional activations for Grad-CAM explainability
    # 'block14_sepconv2_act' is the final convolutional layer of Xception
    conv_features = base_model.output

    # Forensic Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(conv_features)
    
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc_dense_1"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(dropout_rate, name="forensic_dropout_1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc_dense_2"
    )(x)
    x = layers.Dropout(dropout_rate * 0.6, name="forensic_dropout_2")(x)

    # Output logit: Probability of being FAKE
    outputs = layers.Dense(1, activation="sigmoid", name="deepfake_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DeepGuard_XceptionNet")

    # Compile with Adam optimizer, binary crossentropy, and forensic metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from_block: int = 12,
    fine_tune_lr: float = 1e-5
) -> tf.keras.Model:
    """
    Unfreezes the upper convolutional blocks of Xception for fine-tuning
    domain-specific deepfake artifacts (blending seams and frequency distortions).
    
    Args:
        model: Trained Stage-1 model
        unfreeze_from_block: Xception block number from which layers are trained (12, 13, 14)
        fine_tune_lr: Reduced learning rate to prevent catastrophic forgetting
    """
    # Locate base model or iterate layers
    trainable_flag = False
    target_layer_prefix = f"block{unfreeze_from_block}"

    for layer in model.layers:
        if target_layer_prefix in layer.name:
            trainable_flag = True
        
        # Keep BatchNormalization frozen during fine-tuning for statistical stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif trainable_flag:
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model
