
import yaml
import torch
from src.data_loader import prepare_data
from src.train import train
from src.evaluation import run_quantitative_evaluation

# Load and process the configuration file
with open('configs/M1_config.yaml', 'r') as f:
    config_nested = yaml.safe_load(f)

# Flatten the configuration for easy access in the scripts
CONFIG = {}
for key, value in config_nested.items():
    for sub_key, sub_value in value.items():
        CONFIG[f'{key}_{sub_key}'] = sub_value

# Add additional parameters not in the YAML file
CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG['temperature'] = 0.07  # Common default for contrastive losses
CONFIG['loss_alpha'] = 0.5  # Equal weight to token-sentence and sentence-document losses
CONFIG['model_save_path'] = 'muses_model.pth'
CONFIG['doc_size'] = 8  # A reasonable default for document size

# Make CONFIG globally accessible for the other scripts
# This is a simple way to share config without changing function signatures.
import src.data_loader
import src.train
import src.evaluation
import src.losses
import src.models

src.data_loader.CONFIG = CONFIG
src.train.CONFIG = CONFIG
src.evaluation.CONFIG = CONFIG
src.losses.CONFIG = CONFIG
src.models.CONFIG = CONFIG


def main():
    # 1. Prepare the data
    train_data, test_data = prepare_data()

    # 2. Train the model
    train(train_data, test_data)

    # 3. Run the evaluation
    run_quantitative_evaluation()

if __name__ == '__main__':
    main()
