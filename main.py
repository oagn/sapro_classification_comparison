import yaml
import keras
import jax
from data_loader import load_data
from models import create_model
from train import train_model
from evaluate import evaluate_model

# Set JAX as the backend
keras.backend.set_backend('jax')

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for model_name in config['models']:
        print(f"Training {model_name}...")
        train_ds, val_ds, test_ds = load_data(config, model_name)
        model = create_model(model_name, num_classes=2, config=config)  # Assuming binary classification
        history = train_model(model, train_ds, val_ds, config)
        
        print(f"Evaluating {model_name}...")
        evaluate_model(model, test_ds)

        # Save the model
        model.save(f"{model_name}_model.keras")

if __name__ == "__main__":
    main()