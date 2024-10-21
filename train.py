import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss
from data_loader import create_fixed_train, create_tensorset
import pandas as pd
import numpy as np
import os


def train_model(model, train_ds, val_ds, config, learning_rate, epochs, image_size=224, model_name=None, is_fine_tuning=False, combined_df=None):

    class NewDatasetCallback(keras.callbacks.Callback):
        def __init__(self, config, combined_df=None):
            super().__init__()
            self.config = config
            self.combined_df = combined_df
        
        def on_epoch_begin(self, epoch, logs=None):
            if self.config['training']['new_dataset_per_epoch'] and is_fine_tuning:
                if self.combined_df is not None:
                    # Determine the number of samples to use
                    if self.config['pseudo_labeling'].get('use_all_samples', True):
                        new_train_df = self.combined_df
                    else:
                        num_samples = self.config['pseudo_labeling'].get('num_samples_per_epoch', len(self.combined_df))
                        
                        # Option 1: Simple random sampling
                        new_train_df = self.combined_df.sample(n=num_samples, replace=False, random_state=epoch)
                        
                        # Option 2: Stratified sampling (if you want to maintain class balance)
                        # new_train_df = self.stratified_sample(num_samples, epoch)
                    
                    new_train_df = new_train_df.sample(frac=1, random_state=epoch).reset_index(drop=True)
                else:
                    # Fall back to original behavior if combined_df is not provided
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

        def stratified_sample(self, num_samples, random_state):
            # Implement stratified sampling here if needed
            pass


    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['data']['output_dir'], f'{model_name}_best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            patience=config['training'].get('early_stopping_patience', 10),  # Default to 10 if not specified
            restore_best_weights=True
        ),
        NewDatasetCallback(config, combined_df)
    ]


    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    loss = FocalLoss(
        alpha=config['training'].get('focal_loss_alpha', 0.25),
        gamma=config['training']['focal_loss_gamma'],
        from_logits=False,
        name="focal_loss",
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
