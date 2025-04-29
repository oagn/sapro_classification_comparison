import keras
import jax.numpy as jnp
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from keras_cv.losses import FocalLoss
from models import unfreeze_model
import os


def train_model(model, train_ds, val_ds, config, learning_rate, epochs, model_name, is_fine_tuning=False):
    """
    Compiles and trains the model for a specified number of epochs.

    Sets up JAX multi-device execution if available.
    Uses Adam optimizer and FocalLoss.
    Includes EarlyStopping and ReduceLROnPlateau callbacks.
    Adds ModelCheckpoint callback during fine-tuning.

    Args:
        model (keras.Model): The Keras model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        config (dict): Configuration dictionary.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train.
        model_name (str): Base name for the model (used for checkpointing).
        is_fine_tuning (bool): If True, enables ModelCheckpoint callback.

    Returns:
        tuple: (training_history, trained_model)
    """
    # --- Device Setup ---
    devices = jax.devices("gpu")
    n_devices = len(devices)
    
    if n_devices == 0:
        print("No GPU devices found. Using CPU.")
        devices = jax.devices("cpu")
        n_devices = len(devices)
    
    print(f"Using {n_devices} device(s)")
    
    mesh = None
    if n_devices > 1:
        # Use a device mesh for distributed training
        mesh = Mesh(create_device_mesh((n_devices,)), ('devices',))
        print("Using JAX device mesh for distributed training.")
    
    # --- Callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
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

    # --- Optimizer and Loss ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = FocalLoss(
        gamma=config['training']['focal_loss_gamma'],
        from_logits=False,
    )

    # --- Compile and Train ---
    # Compile and train with mesh context if available
    if mesh:
        print("Compiling and training with JAX mesh")
        with mesh:
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)
            history = model.fit(
                x=train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks
            )
    else:
        print("Compiling and training on single device")
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], jit_compile=True)
        history = model.fit(
            x=train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

    return history, model

def train_fold(model, train_ds, val_ds, config, model_name, fold_idx):
    """
    Train a single cross-validation fold with frozen and unfrozen phases.

    Args:
        model (keras.Model): Initial model instance for the fold.
        train_ds (tf.data.Dataset): Training dataset for the fold.
        val_ds (tf.data.Dataset): Validation dataset for the fold.
        config (dict): Configuration dictionary.
        model_name (str): Base name for the model architecture.
        fold_idx (int): The index of the current fold (0-based).

    Returns:
        tuple: (combined_history, final_model_for_fold)
            combined_history (dict): Dictionary containing 'frozen' and 'unfrozen' history objects.
            final_model_for_fold (keras.Model): The model after both training phases.
    """
    print(f"\n--- Training Fold {fold_idx + 1} / {config['training']['n_folds']} ---")
    
    # --- Phase 1: Frozen Training ---
    print("\nPhase 1: Training with frozen base model...")
    frozen_history, model = train_model(
        model, 
        train_ds, 
        val_ds, 
        config, 
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['initial_epochs'],
        model_name=f"{model_name}_fold_{fold_idx+1}_frozen",
        is_fine_tuning=False
    )

    # Check if unfrozen training is needed based on validation performance
    final_val_loss = frozen_history.history['val_loss'][-1]
    best_val_loss = min(frozen_history.history['val_loss'])
    
    
    # --- Phase 2: Unfrozen Fine-tuning ---
    print(f"\nPhase 2: Fine-tuning with {config['models'][model_name]['unfreeze_layers']} unfrozen layers...")
    model = unfreeze_model(model, config['models'][model_name]['unfreeze_layers'])
    
    unfrozen_history, model = train_model(
        model, 
        train_ds, 
        val_ds, 
        config, 
        learning_rate=config['training']['fine_tuning_lr'],
        epochs=config['training']['fine_tuning_epochs'],
        model_name=f"{model_name}_fold_{fold_idx+1}_unfrozen",
        is_fine_tuning=True
    )

    # Combine histories
    combined_history = {
        'frozen': frozen_history.history,
        'unfrozen': unfrozen_history.history
    }

    return combined_history, model