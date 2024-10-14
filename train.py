import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss

def train_model(model, train_ds, val_ds, config, learning_rate, epochs):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps = 4000,
        decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
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
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)

    # Train the model with distributed strategy
    with mesh:
        history = model.fit(
            x=train_ds,
            epochs=epochs,  # Use the epochs parameter here
            validation_data=val_ds,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,  # Increase patience
                    restore_best_weights=True,
                    mode='min'
                ),
                keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
            ]
        )

    return history
