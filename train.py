import keras
import tensorflow as tf

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss

def train_model(model, train_ds, val_ds, config):
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    loss = FocalLoss(gamma=config['training']['focal_loss_gamma'])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['training']['epochs'],
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
        ]
    )

    return history