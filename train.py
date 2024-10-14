import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss
from data_loader import create_fixed_train, create_tensorset


def train_model(model, train_ds, val_ds, config, learning_rate, epochs, image_size=224, model_name=None, is_fine_tuning=False):

    class NewDatasetCallback(keras.callbacks.Callback):
        def __init__(self, config):
            super().__init__()
            self.config = config
        
        def on_epoch_begin(self, epoch, logs=None):
            if self.config['training']['new_dataset_per_epoch'] and is_fine_tuning:
                print(f"Creating new training dataset for epoch {epoch+1}")
                samples_per_class = self.config['sampling'].get('samples_per_class', None)
                new_train_df = create_fixed_train(self.config['data']['train_dir'], samples_per_class)
                new_train_ds = create_tensorset(
                    new_train_df, 
                    image_size,
                    self.config['data']['batch_size'],
                    self.config['data'].get('augmentation_magnitude', 0.3),
                    ds_name="train",
                    model_name=model_name
                )
                self.model.train_dataset = new_train_ds


    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, mode='min'),
        NewDatasetCallback(config),
    ]


    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
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
            epochs=epochs,  
            validation_data=val_ds,
            callbacks=callbacks
        )

    return history
