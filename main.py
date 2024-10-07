import os
os.environ["KERAS_BACKEND"] = "jax"

import yaml
import keras
import jax
from data_loader import load_data
from models import create_model
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'], label='Training F1')
    plt.plot(history.history['val_f1_score'], label='Validation F1')
    plt.title(f'{model_name} - Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = {}

    for model_name in config['models']:
        print(f"Training {model_name}...")
        train_ds, val_ds, test_ds, num_train_samples, num_val_samples = load_data(config, model_name)
        steps_per_epoch = num_train_samples // config['data']['batch_size']
        validation_steps = num_val_samples // config['data']['batch_size']

        model = create_model(model_name, num_classes=2, config=config)
        history = train_model(model, train_ds, val_ds, config, steps_per_epoch, validation_steps)
        
        # Plot and save training history
        plot_training_history(history, model_name)
        
        print(f"Evaluating {model_name}...")
        eval_results = evaluate_model(model, test_ds)
        
        results[model_name] = eval_results

        # Save the model
        model.save(f"{model_name}_model.keras")

    # Print summary of results
    print("\nSummary of Results:")
    for model_name, eval_results in results.items():
        print(f"\n{model_name}:")
        print(f"  Macro F1 score: {eval_results['macro_f1']:.4f}")
        print(f"  Weighted F1 score: {eval_results['weighted_f1']:.4f}")
        for i, f1 in enumerate(eval_results['f1_scores']):
            print(f"  F1 score for class {i}: {f1:.4f}")

if __name__ == "__main__":
    main()