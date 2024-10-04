import tensorflow as tf
import numpy as np

def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['train_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['val_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['test_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )

    # Apply sampling weights to training data
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])
    train_ds = train_ds.map(lambda x, y: (x, y, sample_weights))

    return train_ds, val_ds, test_ds