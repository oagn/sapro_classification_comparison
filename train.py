import keras
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

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
        self.f1_score = self.add_weight(name='f1', initializer='zeros')
        self.step_counter = self.add_weight(name='step_counter', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_argmax = keras.ops.argmax(y_true, axis=-1)
        y_pred_argmax = keras.ops.argmax(y_pred, axis=-1)
        
        self.precision.update_state(y_true_argmax, y_pred_argmax, sample_weight)
        self.recall.update_state(y_true_argmax, y_pred_argmax, sample_weight)
        
        p = self.precision.result()
        r = self.recall.result()
        
        # Avoid division by zero
        f1 = 2 * ((p * r) / (p + r + jnp.finfo(jnp.float32).eps))
        self.f1_score.assign(f1)
        
        self.step_counter.assign_add(1)
        
        print(f"Step {self.step_counter.numpy()}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    def result(self):
        return self.f1_score

    def reset_state(self):
        print(f"Resetting F1Score metric. Final F1: {self.f1_score.numpy():.4f}")
        self.precision.reset_state()
        self.recall.reset_state()
        self.f1_score.assign(0.)
        self.step_counter.assign(0)

class MetricsLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            print(f"Epoch {epoch+1}: {metric} = {value:.4f}")

def train_model(model, train_ds, val_ds, config, steps_per_epoch, validation_steps):
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    loss = FocalLoss(gamma=config['training']['focal_loss_gamma'])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', F1Score()])

    class DebugCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1} ended. Logs: {logs}")
            
            # Get predictions for a batch of training data
            for x, y in train_ds.take(1):
                y_pred = model.predict(x)
                print(f"Debug - Training batch predictions:")
                print(f"y_true shape: {y.shape}, y_pred shape: {y_pred.shape}")
                print(f"y_true sample: {y[:5]}")
                print(f"y_pred sample: {y_pred[:5]}")
                print(f"y_true argmax: {keras.ops.argmax(y, axis=-1)[:10]}")
                print(f"y_pred argmax: {keras.ops.argmax(y_pred, axis=-1)[:10]}")
                
                # Calculate F1 score manually for this batch
                from sklearn.metrics import f1_score
                y_true_argmax = keras.ops.argmax(y, axis=-1).numpy()
                y_pred_argmax = keras.ops.argmax(y_pred, axis=-1).numpy()
                batch_f1 = f1_score(y_true_argmax, y_pred_argmax, average='macro')
                print(f"Manually calculated F1 score for this batch: {batch_f1:.4f}")

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