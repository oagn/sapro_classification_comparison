import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss

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
        
        # Use keras.ops.equal instead of keras.ops.eq
        condition = keras.ops.equal(p + r, 0)
        
        # Use a conditional instead of keras.ops.switch
        return keras.ops.where(
            condition,
            keras.ops.cast(0.0, p.dtype),
            2 * ((p * r) / (p + r + keras.config.epsilon()))
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def train_model(model, train_ds, val_ds, config, steps_per_epoch, validation_steps, learning_rate, epochs):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = FocalLoss(
        alpha=config['training'].get('focal_loss_alpha', 0.25),
        gamma=config['training']['focal_loss_gamma'],
        from_logits=False
    )

    # Set up JAX devices and mesh
    devices = jax.devices("gpu")
    n_devices = len(devices)
    
    if n_devices == 0:
        print("No GPU devices found. Using CPU.")
        devices = jax.devices("cpu")
        n_devices = len(devices)
    
    print(f"Using {n_devices} device(s)")
    
    if n_devices > 1:
        mesh = Mesh(create_device_mesh((n_devices,)), ('devices',))
    else:
        mesh = None

    # Compile the model
    if mesh:
        with mesh:
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', F1Score()], jit_compile=True)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', F1Score()], jit_compile=True)

    class DebugCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1} ended. Logs: {logs}")

    # Train the model with distributed strategy
    with mesh:
        history = model.fit(
            x=train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,  # Use the epochs parameter here
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,  # Increase patience
                    restore_best_weights=True,
                    mode='min'
                ),
                keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3),
                DebugCallback()
            ]
        )

    return history