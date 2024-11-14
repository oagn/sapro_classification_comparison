import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss
from data_loader import create_fixed_train, create_tensorset
from models import create_model, unfreeze_model
import pandas as pd
import numpy as np
import os

class PredictionDistributionCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on validation data
        val_pred = self.model.predict(self.validation_data, verbose=0)
        pos_pred = np.mean(val_pred > 0.5)
        print(f"\nPrediction distribution: {pos_pred:.3%} positive")
        
        # Add more detailed distribution info
        print(f"Prediction stats: min={np.min(val_pred):.3f}, max={np.max(val_pred):.3f}, "
              f"mean={np.mean(val_pred):.3f}, std={np.std(val_pred):.3f}")

def train_model(model, train_ds, val_ds, config, learning_rate, epochs, image_size, model_name, is_fine_tuning=False):
    """
    Train the model with JAX backend and class weights
    """
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

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['data']['output_dir'], f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        PredictionDistributionCallback()
    ]

    # Add model checkpoint for unfrozen phase
    if is_fine_tuning:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config['data']['output_dir'], f'best_model_{model_name}.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Configure Focal Loss
    num_classes = len(config['data']['class_names'])
    loss = FocalLoss(
        gamma=config['training']['focal_loss_gamma'],
        from_logits=False,  # Since we're using sigmoid/softmax activation
        alpha=config['training'].get('focal_loss_alpha', 0.25),  # Default from paper
        name="focal_loss"
    )

    # Compile and train with mesh if available
    if mesh:
        with mesh:
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'],
                jit_compile=True
            )
            history = model.fit(
                x=train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks
            )
    else:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'],
            jit_compile=True
        )
        history = model.fit(
            x=train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

    return history, model

def train_fold(model, train_ds, val_ds, config, model_name, fold_idx):
    """
    Train a single fold with both frozen and unfrozen phases using JAX
    """
    print(f"\nTraining fold {fold_idx + 1}")
    
    # Phase 1: Frozen training
    print("Phase 1: Training with frozen base model...")
    frozen_history, model = train_model(
        model, 
        train_ds, 
        val_ds, 
        config, 
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['initial_epochs'],
        image_size=config['models'][model_name]['img_size'],
        model_name=f"{model_name}_fold_{fold_idx}_frozen",
        is_fine_tuning=False
    )

    # Check if unfrozen training is needed based on validation performance
    final_val_loss = frozen_history.history['val_loss'][-1]
    best_val_loss = min(frozen_history.history['val_loss'])
    
    if final_val_loss > best_val_loss * 1.1:  # If final loss is significantly worse than best
        print("Early stopping triggered during frozen phase. Skipping unfrozen phase.")
        return {'frozen': frozen_history.history}, model
    
    # Phase 2: Unfrozen training
    print(f"Phase 2: Fine-tuning with unfrozen layers...")
    model = unfreeze_model(model, config['models'][model_name]['unfreeze_layers'])
    
    unfrozen_history, model = train_model(
        model, 
        train_ds, 
        val_ds, 
        config, 
        learning_rate=config['training']['fine_tuning_lr'],
        epochs=config['training']['fine_tuning_epochs'],
        image_size=config['models'][model_name]['img_size'],
        model_name=f"{model_name}_fold_{fold_idx}_unfrozen",
        is_fine_tuning=True
    )

    # Combine histories
    combined_history = {
        'frozen': frozen_history.history,
        'unfrozen': unfrozen_history.history
    }

    return combined_history, model
