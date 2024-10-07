import keras
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce_loss = keras.ops.categorical_crossentropy(y_true, y_pred, from_logits=False)
        pt = keras.ops.sum(y_true * y_pred, axis=-1)
        focal_loss = keras.ops.power(1. - pt, self.gamma) * ce_loss
        
        # Modify alpha_factor calculation
        alpha_factor = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        alpha_factor = keras.ops.sum(alpha_factor, axis=-1)
        
        modulated_loss = alpha_factor * focal_loss
        return keras.ops.mean(modulated_loss)

class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.argmax(y_true, axis=-1)
        y_pred = keras.ops.argmax(y_pred, axis=-1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + jnp.finfo(jnp.float32).eps))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class MetricsLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            print(f"Epoch {epoch+1}: {metric} = {value:.4f}")

def train_model(model, train_ds, val_ds, config, steps_per_epoch, validation_steps):
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    loss = FocalLoss(gamma=config['training']['focal_loss_gamma'])

    # Set up JAX devices and mesh
    devices = jax.devices()
    mesh = Mesh(create_device_mesh((2,)), ('devices',))

    # Compile the model with distributed strategy
    with mesh:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', F1Score()],
            jit_compile=True,
        )

    class DebugCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1} ended. Logs: {logs}")

    # Train the model with distributed strategy
    with mesh:
        history = model.fit(
            x=train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=config['training']['epochs'],
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3),
                MetricsLogger(),
                DebugCallback()
            ]
        )

    return history