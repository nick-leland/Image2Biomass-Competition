"""
Optuna objective function for hyperparameter optimization.
"""

import sys
import pandas as pd
import torch
from pathlib import Path
from typing import Dict

import optuna

# Ensure project root is in path
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

from src.config import TRAIN_CSV, TRAIN_IMG_DIR, TARGET_NAMES
from src.utils.seed import set_seed, worker_init_fn
from src.data.dataset import BiomassDataset, create_dataloaders
from src.data.splitter import create_stratified_split
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.models.loss_functions import create_loss_function
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import BiomassTrainer
from src.optuna_optimization.hyperparameters import define_hyperparameter_space


class OptunaObjective:
    """
    Objective function for Optuna optimization.

    This class is callable and will be used by Optuna to evaluate trials.
    """

    def __init__(self, device='cuda', checkpoint_dir=None):
        """
        Initialize the objective function.

        Args:
            device: Device to run training on
            checkpoint_dir: Directory to save trial checkpoints (optional)
        """
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Load data once (shared across all trials)
        print("Loading data for optimization...")
        self.df = pd.read_csv(TRAIN_CSV)
        self.df['image_id'] = self.df['sample_id'].str.split('__').str[0]
        self.df_wide = self.df.pivot_table(
            index=['image_id', 'image_path', 'Sampling_Date', 'State', 'Species',
                   'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            values='target'
        ).reset_index()
        print(f"Loaded {len(self.df_wide)} images for optimization\n")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Evaluate a single trial.

        Args:
            trial: Optuna trial object

        Returns:
            best_val_loss: Best validation loss achieved
        """
        # Sample hyperparameters
        config = define_hyperparameter_space(trial)

        # Print trial info
        print(f"\n{'=' * 70}")
        print(f"Trial {trial.number}")
        print(f"{'=' * 70}")
        print("Hyperparameters:")
        for key, value in config.items():
            if key != 'task_weights':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}:")
                for target, weight in value.items():
                    print(f"    {target}: {weight}")
        print(f"{'=' * 70}\n")

        # Set seed for reproducibility
        set_seed(config['seed'])

        # Create stratified split
        train_df, val_df = create_stratified_split(
            self.df_wide,
            stratify_by=config['stratify_by'],
            val_split=config['val_split'],
            random_seed=config['seed']
        )

        # Get transforms
        train_transform = get_train_transforms(
            level=config['augmentation_level'],
            image_size=config['image_size']
        )
        val_transform = get_val_transforms(image_size=config['image_size'])

        # Convert to long format for datasets
        train_long = []
        for _, row in train_df.iterrows():
            for target_name in TARGET_NAMES:
                train_long.append({
                    'sample_id': f"{row['image_id']}__{target_name}",
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'target_name': target_name,
                    'target': row[target_name],
                    'State': row.get('State', ''),
                    'Species': row.get('Species', ''),
                    'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                    'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                    'Sampling_Date': row.get('Sampling_Date', ''),
                })
        train_long_df = pd.DataFrame(train_long)

        val_long = []
        for _, row in val_df.iterrows():
            for target_name in TARGET_NAMES:
                val_long.append({
                    'sample_id': f"{row['image_id']}__{target_name}",
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'target_name': target_name,
                    'target': row[target_name],
                    'State': row.get('State', ''),
                    'Species': row.get('Species', ''),
                    'Pre_GSHH_NDVI': row.get('Pre_GSHH_NDVI', 0),
                    'Height_Ave_cm': row.get('Height_Ave_cm', 0),
                    'Sampling_Date': row.get('Sampling_Date', ''),
                })
        val_long_df = pd.DataFrame(val_long)

        # Save temporary CSVs
        temp_dir = Path('experiments/temp')
        temp_dir.mkdir(exist_ok=True, parents=True)
        train_csv_path = temp_dir / f'train_trial_{trial.number}.csv'
        val_csv_path = temp_dir / f'val_trial_{trial.number}.csv'
        train_long_df.to_csv(train_csv_path, index=False)
        val_long_df.to_csv(val_csv_path, index=False)

        # Create datasets
        train_dataset = BiomassDataset(
            csv_path=train_csv_path,
            img_dir=TRAIN_IMG_DIR,
            transform=train_transform,
            is_test=False
        )

        val_dataset = BiomassDataset(
            csv_path=val_csv_path,
            img_dir=TRAIN_IMG_DIR,
            transform=val_transform,
            is_test=False,
            target_stats=train_dataset.get_target_stats()
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            worker_init_fn=worker_init_fn
        )

        # Create model
        model = create_model(config)
        model = model.to(self.device)

        # Create loss function
        criterion = create_loss_function(config)

        # Create optimizer
        optimizer = create_optimizer(model, config)

        # Create scheduler (need steps_per_epoch for OneCycleLR)
        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

        # Create checkpoint directory for this trial if specified
        if self.checkpoint_dir is not None:
            trial_checkpoint_dir = self.checkpoint_dir / f'trial_{trial.number}'
        else:
            trial_checkpoint_dir = Path('experiments/optuna_checkpoints') / f'trial_{trial.number}'

        # Create trainer
        trainer = BiomassTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            checkpoint_dir=trial_checkpoint_dir,
            early_stopping_patience=config['early_stopping_patience'],
            denormalize_fn=train_dataset.denormalize_targets
        )

        # Set trial for pruning
        trainer.trial = trial

        # Train
        try:
            best_val_loss = trainer.train(num_epochs=config['num_epochs'])
        except optuna.TrialPruned:
            # Trial was pruned by Optuna
            raise
        except Exception as e:
            # Other errors - report as failed trial
            print(f"\nTrial {trial.number} failed with error: {e}")
            raise optuna.TrialPruned()
        finally:
            # Clean up temporary CSVs
            if train_csv_path.exists():
                train_csv_path.unlink()
            if val_csv_path.exists():
                val_csv_path.unlink()

            # Clean up GPU memory to prevent OOM in subsequent trials
            del model
            del trainer
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return best_val_loss


def create_objective(device='cuda', checkpoint_dir=None) -> OptunaObjective:
    """
    Create an Optuna objective function.

    Args:
        device: Device to run training on
        checkpoint_dir: Directory to save trial checkpoints

    Returns:
        objective: Callable objective function
    """
    return OptunaObjective(device=device, checkpoint_dir=checkpoint_dir)
