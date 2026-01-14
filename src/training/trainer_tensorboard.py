"""
Enhanced Trainer with TensorBoard logging and full resume capability.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.metrics import compute_metrics, print_metrics


class BiomassTrainerTensorBoard:
    """
    Enhanced Trainer for biomass prediction with TensorBoard and resume capability.

    New features:
    - TensorBoard logging for real-time monitoring
    - Full resume capability (optimizer state, scheduler state, epoch, history)
    - Automatic checkpoint management
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
        denormalize_fn=None,
        tensorboard_dir=None,
        fold_idx=None
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
        self.fold_idx = fold_idx

        # Training state (will be loaded from checkpoint if resuming)
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

        # TensorBoard setup
        if tensorboard_dir is None:
            tensorboard_dir = self.checkpoint_dir / 'tensorboard'
        self.tensorboard_dir = Path(tensorboard_dir)
        self.tensorboard_dir.mkdir(exist_ok=True, parents=True)

        log_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir / f'run{log_suffix}'))

        print(f"TensorBoard logs: {self.tensorboard_dir}")
        print(f"  Run: tensorboard --logdir {self.tensorboard_dir.parent}")

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

    def log_to_tensorboard(self, train_loss, val_loss, train_metrics, val_metrics, lr):
        """Log metrics to TensorBoard."""
        epoch = self.current_epoch

        # Loss
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)

        # Learning rate
        self.writer.add_scalar('Learning_Rate', lr, epoch)

        # Overall metrics
        self.writer.add_scalar('MAE/train', train_metrics['overall_MAE'], epoch)
        self.writer.add_scalar('MAE/val', val_metrics['overall_MAE'], epoch)
        self.writer.add_scalar('RMSE/train', train_metrics['overall_RMSE'], epoch)
        self.writer.add_scalar('RMSE/val', val_metrics['overall_RMSE'], epoch)
        self.writer.add_scalar('R2/train', train_metrics['overall_R2'], epoch)
        self.writer.add_scalar('R2/val', val_metrics['overall_R2'], epoch)

        # Per-target MAE
        for target_name in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']:
            self.writer.add_scalar(f'MAE_Train/{target_name}', train_metrics[f'{target_name}_MAE'], epoch)
            self.writer.add_scalar(f'MAE_Val/{target_name}', val_metrics[f'{target_name}_MAE'], epoch)
            self.writer.add_scalar(f'R2_Train/{target_name}', train_metrics[f'{target_name}_R2'], epoch)
            self.writer.add_scalar(f'R2_Val/{target_name}', val_metrics[f'{target_name}_R2'], epoch)

    def train(self, num_epochs, resume_from=None):
        """
        Main training loop with resume capability.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)

        Returns:
            best_val_loss: Best validation loss achieved
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from, load_training_state=True)
            print(f"Resumed from epoch {self.current_epoch + 1}")
            print(f"Best val loss so far: {self.best_val_loss:.4f}")

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, num_epochs):
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

            # Log to TensorBoard
            self.log_to_tensorboard(train_loss, val_loss, train_metrics, val_metrics, current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train MAE: {train_metrics['overall_MAE']:.4f} | Val MAE: {val_metrics['overall_MAE']:.4f}")
            print(f"  Train R2: {train_metrics['overall_R2']:.4f} | Val R2: {val_metrics['overall_R2']:.4f}")
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

            # Save resumable checkpoint every epoch (lightweight)
            self.save_checkpoint('last_checkpoint.pth')

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break

        # Print final metrics
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model and compute final metrics
        self.load_checkpoint('best_model.pth')
        _, final_val_metrics = self.validate()
        print_metrics(final_val_metrics, prefix="Final Validation")

        # Close TensorBoard writer
        self.writer.close()

        return self.best_val_loss

    def save_checkpoint(self, filename):
        """
        Save full checkpoint with training state.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'history': self.history,
        }

        # Save scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename, load_training_state=True):
        """
        Load checkpoint with optional training state resume.

        Args:
            filename: Checkpoint filename
            load_training_state: If True, restore optimizer, scheduler, and training state
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_training_state:
            # Restore full training state
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            self.history = checkpoint.get('history', self.history)

            # Restore optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler state if exists
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"  Resumed from epoch: {self.current_epoch + 1}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
        else:
            # Only load model for inference
            print(f"Loaded model weights from: {checkpoint_path}")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")
