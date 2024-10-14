import os
os.environ["KERAS_BACKEND"] = "jax"

import yaml
from data_loader import load_data
from models import create_model, unfreeze_model
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import datetime

def plot_training_history(history, model_name, output_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title(f'{model_name} - Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'))
    plt.close()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for model_name in config['models']:
        print(f"Training {model_name}...")
        train_ds, val_ds, test_ds, num_train_samples, num_val_samples = load_data(config, model_name)
        print(f"Number of training samples: {num_train_samples}")
        print(f"Number of validation samples: {num_val_samples}")

        # Check a batch from each dataset
        for batch in train_ds.take(1):
            if len(batch) == 2:
                x, y = batch
                print(f"Training batch shape: features {x.shape}, labels {y.shape}")
            elif len(batch) == 3:
                x, y, w = batch
                print(f"Training batch shape: features {x.shape}, labels {y.shape}, weights {w.shape}")
            else:
                print(f"Unexpected number of elements in training batch: {len(batch)}")
                for i, element in enumerate(batch):
                    print(f"Element {i} shape: {element.shape}")

        for batch in val_ds.take(1):
            if len(batch) == 2:
                x, y = batch
                print(f"Validation batch shape: features {x.shape}, labels {y.shape}")
            elif len(batch) == 3:
                x, y, w = batch
                print(f"Validation batch shape: features {x.shape}, labels {y.shape}, weights {w.shape}")
            else:
                print(f"Unexpected number of elements in validation batch: {len(batch)}")
                for i, element in enumerate(batch):
                    print(f"Element {i} shape: {element.shape}")

        steps_per_epoch = num_train_samples // config['data']['batch_size']
        if steps_per_epoch == 0:
            steps_per_epoch = 1  # Ensure at least one step per epoch
        validation_steps = num_val_samples // config['data']['batch_size']

        model = create_model(model_name, num_classes=2, config=config)
           
        # Initial training with frozen base model
        print("Initial training with frozen base model...")
        history_frozen = train_model(
            model, 
            train_ds, 
            val_ds, 
            config, 
            learning_rate=config['training']['learning_rate'],
            epochs=config['training']['initial_epochs'],
            is_fine_tuning=False
        )
        
        # Plot and save training history for frozen model
        plot_training_history(history_frozen, f"{model_name}_frozen", output_dir)
        
        # Unfreeze the top layers of the base model
        print(f"Unfreezing top {config['models'][model_name]['unfreeze_layers']} layers of the model...")
        model = unfreeze_model(model, config['models'][model_name]['unfreeze_layers'])
        # Continue training with partially unfrozen model
        print("Continuing training with partially unfrozen model...")
        history_unfrozen = train_model(
            model, 
            train_ds, 
            val_ds, 
            config, 
            learning_rate=config['training']['fine_tuning_lr'],
            epochs=config['training']['fine_tuning_epochs'],
            is_fine_tuning=True
        )
        
        # Plot and save training history for unfrozen model
        plot_training_history(history_unfrozen, f"{model_name}_unfrozen", output_dir)
        
        print(f"Evaluating {model_name}...")
        eval_results = evaluate_model(model, config['data']['test_dir'], ['healthy','sapro'], 
                                      batch_size=config['data']['batch_size'], 
                                      img_size=config['models'][model_name]['img_size'], 
                                      output_path=config['data']['output_dir'])
        
        results[model_name] = eval_results

        # Save the final model
        model.save(os.path.join(output_dir, f"{model_name}_model_final.keras"))

    # Print and save summary of results
    summary = "\nSummary of Results:\n"
    for model_name, eval_results in results.items():
        summary += f"\n{model_name}:\n"
        summary += f"  Macro F1 score: {eval_results['macro_f1']:.4f}\n"
        summary += f"  Weighted F1 score: {eval_results['weighted_f1']:.4f}\n"
        for i, f1 in enumerate(eval_results['f1_scores']):
            summary += f"  F1 score for class {i}: {f1:.4f}\n"
    
    print(summary)
    
    # Save summary to file with model name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f'results_summary_{model_name}_{timestamp}.txt'
    with open(os.path.join(output_dir, summary_filename), 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main()
