"""
Trainer class for multi-task biomass regression.

Handles training loop, validation, checkpointing, and early stopping.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.evaluation.metrics import compute_metrics, print_metrics


class BiomassTrainer:
    """
    Trainer for biomass prediction models.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: 'cuda' or 'cpu'
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Epochs to wait before early stopping
        denormalize_fn: Function to denormalize predictions/targets for metrics
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        checkpoint_dir='experiments/checkpoints',
        early_stopping_patience=10,
        denormalize_fn=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.early_stopping_patience = early_stopping_patience
        self.denormalize_fn = denormalize_fn

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_metrics': [],
            'val_metrics': []
        }

        # For Optuna pruning
        self.trial = None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = {name: [] for name in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']}
        all_targets = {name: [] for name in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions and targets for metrics
            for name in all_predictions.keys():
                all_predictions[name].append(predictions[name].detach())
                all_targets[name].append(targets[name].detach())

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)

        # Concatenate all batches
        all_predictions = {name: torch.cat(preds) for name, preds in all_predictions.items()}
        all_targets = {name: torch.cat(tgts) for name, tgts in all_targets.items()}

        # Compute metrics
        train_metrics = compute_metrics(all_predictions, all_targets, self.denormalize_fn)

        return avg_loss, train_metrics

    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = {name: [] for name in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']}
        all_targets = {name: [] for name in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)

                total_loss += loss.item()

                # Store predictions and targets
                for name in all_predictions.keys():
                    all_predictions[name].append(predictions[name])
                    all_targets[name].append(targets[name])

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)

        # Concatenate all batches
        all_predictions = {name: torch.cat(preds) for name, preds in all_predictions.items()}
        all_targets = {name: torch.cat(tgts) for name, tgts in all_targets.items()}

        # Compute metrics
        val_metrics = compute_metrics(all_predictions, all_targets, self.denormalize_fn)

        return avg_loss, val_metrics

    def train(self, num_epochs):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            best_val_loss: Best validation loss achieved
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train MAE: {train_metrics['overall_MAE']:.4f} | Val MAE: {val_metrics['overall_MAE']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break

            # Optuna pruning
            if self.trial is not None:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    print(f"\nTrial pruned by Optuna at epoch {epoch + 1}")
                    raise Exception("Trial pruned")  # Optuna will catch this

        # Print final metrics
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model and compute final metrics
        self.load_checkpoint('best_model.pth')
        _, final_val_metrics = self.validate()
        print_metrics(final_val_metrics, prefix="Final Validation")

        return self.best_val_loss

    def save_checkpoint(self, filename):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")

    def get_history(self):
        """Get training history."""
        return self.history
