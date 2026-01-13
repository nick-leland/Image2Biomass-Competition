"""
Run Optuna hyperparameter optimization.

Usage:
    python scripts/optimize_hyperparams.py
    python scripts/optimize_hyperparams.py --n_trials 100 --timeout 86400
    python scripts/optimize_hyperparams.py --study_name my_study --storage sqlite:///my_study.db
"""

import sys
sys.path.insert(0, '/home/chaot/kaggle/Image2Biomass-Competition')

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.config import OPTUNA_DIR
from src.optuna_optimization.objective import create_objective


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of trials to run (default: 100)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: None)')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Study name (default: auto-generated with timestamp)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Storage URL (default: SQLite in OPTUNA_DIR)')
    parser.add_argument('--load_if_exists', action='store_true',
                       help='Load existing study if it exists')
    parser.add_argument('--n_startup_trials', type=int, default=5,
                       help='Number of startup trials before pruning (default: 5)')
    parser.add_argument('--n_warmup_steps', type=int, default=10,
                       help='Number of warmup steps for pruner (default: 10)')
    parser.add_argument('--dashboard_port', type=int, default=8080,
                       help='Port for Optuna dashboard (default: 8080)')
    args = parser.parse_args()

    print("=" * 70)
    print("Image2Biomass - Optuna Hyperparameter Optimization")
    print("=" * 70)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Generate study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f'biomass_optimization_{timestamp}'
    else:
        study_name = args.study_name

    # Generate storage URL if not provided
    if args.storage is None:
        OPTUNA_DIR.mkdir(exist_ok=True, parents=True)
        storage = f'sqlite:///{OPTUNA_DIR}/{study_name}.db'
    else:
        storage = args.storage

    print(f"\nStudy name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Number of trials: {args.n_trials}")
    if args.timeout is not None:
        print(f"Timeout: {args.timeout} seconds ({args.timeout/3600:.1f} hours)")

    # Print dashboard info
    print(f"\n{'=' * 70}")
    print("Optuna Dashboard (Live Monitoring)")
    print(f"{'=' * 70}")
    print("\nTo view live progress, open a new terminal and run:")
    print(f"  source .venv/bin/activate")
    print(f"  optuna-dashboard {storage} --port {args.dashboard_port}")
    print(f"\nThen open in browser: http://localhost:{args.dashboard_port}")
    print(f"{'=' * 70}")

    # Create sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps
    )

    # Create study
    print(f"\nCreating Optuna study...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction='minimize',
        load_if_exists=args.load_if_exists
    )

    # Create objective
    checkpoint_dir = OPTUNA_DIR / study_name / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    objective = create_objective(device=device, checkpoint_dir=checkpoint_dir)

    # Run optimization
    print(f"\n{'=' * 70}")
    print("Starting optimization...")
    print(f"{'=' * 70}\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("Optimization complete!")
    print(f"{'=' * 70}")

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # Best trial
    print(f"\n{'=' * 70}")
    print("Best trial:")
    print(f"{'=' * 70}")

    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  Best validation loss: {trial.value:.4f}")

    print("\n  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best config
    best_config_path = OPTUNA_DIR / study_name / 'best_config.json'
    best_config_path.parent.mkdir(exist_ok=True, parents=True)

    # Reconstruct full config from best trial params
    best_config = {
        'backbone': trial.params['backbone'],
        'pretrained': True,
        'dropout': trial.params['dropout'],
        'head_hidden_dim': trial.params['head_hidden_dim'],
        'constraint_mode': trial.params['constraint_mode'],
        'constraint_weight': trial.params.get('constraint_weight', 0.0),
        'loss_function': trial.params['loss_function'],
        'huber_delta': trial.params.get('huber_delta', 1.0),
        'task_weights': {
            'Dry_Clover_g': trial.params['weight_clover'],
            'Dry_Dead_g': trial.params['weight_dead'],
            'Dry_Green_g': trial.params['weight_green'],
            'Dry_Total_g': trial.params['weight_total'],
            'GDM_g': trial.params['weight_gdm'],
        },
        'optimizer': trial.params['optimizer'],
        'learning_rate': trial.params['learning_rate'],
        'weight_decay': trial.params['weight_decay'],
        'momentum': trial.params.get('momentum', 0.9),
        'scheduler': trial.params['scheduler'],
        'scheduler_patience': trial.params.get('scheduler_patience', 5),
        'augmentation_level': trial.params['augmentation_level'],
        'image_size': trial.params['image_size'],
        'batch_size': trial.params['batch_size'],
        'num_epochs': 50,  # Use full epochs for final training
        'early_stopping_patience': 10,
        'num_workers': 4,
        'seed': 42,
        'stratify_by': 'state',
        'val_split': 0.2,
    }

    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest config saved to: {best_config_path}")

    # Generate optimization history plot
    try:
        import optuna.visualization as vis
        import plotly

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig_path = OPTUNA_DIR / study_name / 'optimization_history.html'
        plotly.offline.plot(fig, filename=str(fig_path), auto_open=False)
        print(f"Optimization history plot saved to: {fig_path}")

        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig_path = OPTUNA_DIR / study_name / 'param_importances.html'
        plotly.offline.plot(fig, filename=str(fig_path), auto_open=False)
        print(f"Parameter importances plot saved to: {fig_path}")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig_path = OPTUNA_DIR / study_name / 'parallel_coordinate.html'
        plotly.offline.plot(fig, filename=str(fig_path), auto_open=False)
        print(f"Parallel coordinate plot saved to: {fig_path}")

    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")
        print("Install plotly to enable visualization: uv pip install plotly")

    print(f"\n{'=' * 70}")
    print("Next steps:")
    print(f"  1. Review best config: {best_config_path}")
    print(f"  2. Train final model with best config")
    print(f"  3. Generate submission")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
