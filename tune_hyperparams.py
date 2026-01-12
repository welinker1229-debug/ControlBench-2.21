"""
Hyperparameter tuning orchestration script.
This script runs multiple train.py instances with different hyperparameter combinations
and manages memory efficiently between runs.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import torch
import time
import itertools
from datetime import datetime
from config import get_hyperparameter_search_space
from utils import clear_gpu_memory, get_memory_usage

SEED = 42
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 100

def _create_result_dict(trial_idx, trial_params, metrics=None, train_time=0.0, error=None):
    """Create a standardized result dictionary for a trial."""
    result = {
        'trial': trial_idx + 1,
        'val_f1': metrics.get('best_val_f1', 0.0) if metrics else 0.0,
        'val_acc': metrics.get('best_val_acc', 0.0) if metrics else 0.0,
        'final_test_f1': metrics.get('final_test_f1', 0.0) if metrics else 0.0,
        'final_test_acc': metrics.get('final_test_acc', 0.0) if metrics else 0.0,
        'train_time': train_time,
        **trial_params
    }
    if error:
        result['error'] = error
    return result


def _estimate_tuning_time(search_space, epochs_per_trial=DEFAULT_EPOCHS, time_per_epoch=30):
    """Estimate total time for hyperparameter tuning."""
    total_combinations = 1
    for values in search_space.values():
        total_combinations *= len(values)
    
    time_per_trial = epochs_per_trial * time_per_epoch
    total_time_seconds = total_combinations * time_per_trial
    
    return {
        "total_combinations": total_combinations,
        "time_per_trial_minutes": time_per_trial / 60,
        "total_time_hours": total_time_seconds / 3600,
        "total_time_days": total_time_seconds / (3600 * 24)
    }


def _generate_combinations(search_space, max_trials):
    """Generate hyperparameter combinations, sampling if needed."""
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    if len(all_combinations) > max_trials:
        print(f"⚠️  Too many combinations ({len(all_combinations)}), randomly sampling {max_trials}")
        np.random.seed(SEED)
        selected_indices = np.random.choice(len(all_combinations), max_trials, replace=False)
        return [all_combinations[i] for i in selected_indices], param_names
    
    return all_combinations, param_names


def _verify_dataset_files(data_dir, dataset_name):
    """Verify that required dataset files exist."""
    required_files = ["train.json", "validation.json", "test.json"]
    missing = []
    for filename in required_files:
        filepath = os.path.join(data_dir, dataset_name, filename)
        if not os.path.exists(filepath):
            missing.append(filepath)
    
    if missing:
        raise FileNotFoundError(
            f"Split files not found for {dataset_name}. "
            f"Missing: {', '.join(missing)}. Run split_dataset.py first."
        )

def _run_single_trial(trial_idx, trial_params, model_type, dataset_name, total_trials):
    """Run a single hyperparameter trial and return results."""
    print(f"\n📊 Trial {trial_idx + 1}/{total_trials}")
    print(f"Parameters: {trial_params}")
    
    clear_gpu_memory()
    memory = get_memory_usage()
    print(f"Initial GPU Memory: {memory['allocated']:.2f}GB allocated")
    
    try:
        from train import train_model
        
        trial_config = {
            **trial_params,
            'num_epochs': DEFAULT_EPOCHS,
            'patience': DEFAULT_PATIENCE,
            'seed': SEED
        }
        
        start_time = time.time()
        metrics = train_model(model_type, dataset_name, trial_config)
        train_time = time.time() - start_time
        
        clear_gpu_memory()
        
        if "best_val_f1" in metrics and "best_val_acc" in metrics:
            print(f"✅ Val F1: {metrics['best_val_f1']:.4f}, Val Acc: {metrics['best_val_acc']:.4f}")
            print(f"   Test F1: {metrics.get('final_test_f1', 0.0):.4f}, Time: {train_time:.1f}s")
            return _create_result_dict(trial_idx, trial_params, metrics, train_time), metrics
        else:
            print(f"❌ Invalid metrics returned: {list(metrics.keys())}")
            return _create_result_dict(trial_idx, trial_params, None, train_time, 'Invalid metrics'), None
            
    except Exception as e:
        print(f"❌ Trial failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return _create_result_dict(trial_idx, trial_params, None, 0.0, str(e)), None


def _save_results(results_dir, trial_results, best_config, best_val_f1, 
                  model_type, dataset_name, search_space, num_trials, timestamp):
    """Save all results to files."""
    if not trial_results:
        return
    
    results_df = pd.DataFrame(trial_results)
    results_df.to_csv(os.path.join(results_dir, "hyperparameter_search_results.csv"), index=False)
    
    plot_hyperparameter_results(results_df, os.path.join(results_dir, "hyperparameter_plots.png"))
    
    if best_config:
        best_result = {
            "model_type": model_type,
            "dataset": dataset_name,
            "best_hyperparams": best_config,
            "best_val_f1": best_val_f1,
            "search_space": search_space,
            "num_trials": num_trials,
            "timestamp": timestamp
        }
        
        # Save in search directory
        with open(os.path.join(results_dir, "best_config.json"), "w") as f:
            json.dump(best_result, f, indent=2)
        
        # Save to tuning_results for compatibility
        tuning_results_dir = f"tuning_results/{model_type}_{dataset_name}"
        os.makedirs(tuning_results_dir, exist_ok=True)
        with open(os.path.join(tuning_results_dir, "best_config.json"), "w") as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\n🏆 Best hyperparameter configuration:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"  Best Validation F1: {best_val_f1:.4f}")
    else:
        print(f"\n❌ No valid configurations found!")


def run_comprehensive_hyperparameter_search(model_type, dataset_name, max_trials=30, 
                                           data_dir="split_datasets"):
    """
    Run comprehensive hyperparameter search using train.py with validation.
    
    Args:
        model_type: Type of GNN model
        dataset_name: Name of the dataset
        max_trials: Maximum number of trials to run
        data_dir: Directory containing split datasets
        
    Returns:
        best_config: Best hyperparameter configuration
        results_df: DataFrame with all trial results
    """
    print(f"🔍 Starting hyperparameter search")
    print(f"Model: {model_type}, Dataset: {dataset_name}, Max trials: {max_trials}")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_search_{model_type}_{dataset_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get search space
    try:
        search_space = get_hyperparameter_search_space(model_type, dataset_name)
        print(f"Search space: {search_space}")
    except Exception as e:
        print(f"❌ Error getting search space: {e}")
        return None, None
    
    # Generate combinations
    combinations_to_try, param_names = _generate_combinations(search_space, max_trials)
    
    # Time estimates
    time_estimates = _estimate_tuning_time(search_space)
    estimated_hours = len(combinations_to_try) * time_estimates['time_per_trial_minutes'] / 60
    print(f"\n⏱️  Time Estimates:")
    print(f"  Total combinations: {time_estimates['total_combinations']:,}")
    print(f"  Running {len(combinations_to_try)} trials")
    print(f"  Estimated time: {estimated_hours:.1f} hours")
    
    # Verify dataset files
    _verify_dataset_files(data_dir, dataset_name)
    
    # Run trials
    trial_results = []
    best_val_f1 = 0
    best_config = None
    
    for trial_idx, param_combination in enumerate(combinations_to_try):
        trial_params = dict(zip(param_names, param_combination))
        result, metrics = _run_single_trial(
            trial_idx, trial_params, model_type, dataset_name, len(combinations_to_try)
        )
        trial_results.append(result)
        
        # Update best configuration
        if metrics and metrics.get("best_val_f1", 0) > best_val_f1:
            best_val_f1 = float(metrics["best_val_f1"])
            best_config = trial_params.copy()
            print(f"🎉 New best! Val F1: {best_val_f1:.4f}")
        
        clear_gpu_memory()
        time.sleep(2)  # Brief pause for memory cleanup
    
    # Save results
    _save_results(results_dir, trial_results, best_config, best_val_f1,
                  model_type, dataset_name, search_space, len(combinations_to_try), timestamp)
    
    print(f"\n📊 Results saved to: {results_dir}")
    results_df = pd.DataFrame(trial_results) if trial_results else pd.DataFrame()
    return best_config, results_df

def plot_hyperparameter_results(results_df, save_path):
    """Plot hyperparameter tuning results."""
    valid_results = results_df[results_df['val_f1'] > 0] if not results_df.empty else pd.DataFrame()
    
    if valid_results.empty:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=16)
    
    # Define plots: (param_name, axis, use_log_scale)
    plots = [
        ('lr', axes[0, 0], True),
        ('hidden_size', axes[0, 1], False),
        ('n_layers', axes[1, 0], False),
        ('dropout', axes[1, 1], False)
    ]
    
    for param_name, ax, use_log in plots:
        if param_name in valid_results.columns:
            ax.scatter(valid_results[param_name], valid_results['val_f1'], alpha=0.7, s=50)
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Validation F1')
            ax.set_title(f'F1 vs {param_name.replace("_", " ").title()}')
            if use_log:
                ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Hyperparameter plots saved to {save_path}")

def _create_model_summary(best_config, results_df):
    """Create summary dictionary for a model's hyperparameter search."""
    if best_config is None or results_df.empty:
        return {
            'best_config': None,
            'best_val_f1': 0.0,
            'num_trials': 0,
            'successful_trials': 0,
            'error': 'No valid configuration found' if best_config is None else None
        }
    
    return {
        'best_config': best_config,
        'best_val_f1': results_df['val_f1'].max(),
        'num_trials': len(results_df),
        'successful_trials': len(results_df[results_df['val_f1'] > 0])
    }


def run_all_models_hyperparameter_search(dataset_name, models=None, max_trials=20):
    """
    Run hyperparameter search for all models on a specific dataset.
    Memory is cleared between each model to prevent OOM.
    
    Args:
        dataset_name: Name of the dataset
        models: List of model names (default: all models)
        max_trials: Maximum trials per model
    """
    if models is None:
        models = ["RGCN", "HAN", "HGT", "HinSAGE"]
    
    print(f"🚀 Running hyperparameter search for all models on {dataset_name}")
    print(f"Models: {models}, Max trials per model: {max_trials}")
    
    results_summary = {}
    
    for model_idx, model_type in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"🔧 Model {model_idx}/{len(models)}: {model_type}")
        print(f"{'='*60}")
        
        clear_gpu_memory()
        
        try:
            best_config, results_df = run_comprehensive_hyperparameter_search(
                model_type, dataset_name, max_trials
            )
            results_summary[model_type] = _create_model_summary(best_config, results_df)
            
            status = "✅ completed" if best_config else "❌ failed"
            print(f"{status} {model_type} hyperparameter search")
            
        except Exception as e:
            print(f"❌ Error during {model_type} hyperparameter search: {str(e)}")
            results_summary[model_type] = _create_model_summary(None, pd.DataFrame())
            results_summary[model_type]['error'] = str(e)
        
        clear_gpu_memory()
        time.sleep(5)  # Longer pause between models
        
        print(f"✓ Completed {model_idx}/{len(models)} models")
        if torch.cuda.is_available():
            memory = get_memory_usage()
            print(f"GPU Memory: {memory['allocated']:.2f}GB allocated")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"hyperparameter_search_all_models_{dataset_name}_{timestamp}"
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(os.path.join(summary_dir, "all_models_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final summary
    print(f"\n🎯 FINAL SUMMARY for {dataset_name}")
    print("="*80)
    for model_type, result in results_summary.items():
        if result['best_config'] is not None:
            print(f"{model_type:10} | Best F1: {result['best_val_f1']:.4f} | "
                  f"Trials: {result['successful_trials']}/{result['num_trials']}")
        else:
            error = result.get('error', 'Unknown')
            print(f"{model_type:10} | FAILED | Error: {error}")
    
    print(f"\n📁 Summary saved to: {summary_dir}")
    return results_summary

def _set_deterministic_behavior():
    """Set seeds and environment variables for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for GNN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single model:  python tune_hyperparams.py --model HinSAGE --dataset lgbtq --max_trials 20
  All models:    python tune_hyperparams.py --dataset trump --all_models --max_trials 10
  Custom models: python tune_hyperparams.py --dataset abortion --models RGCN,HAN --max_trials 15
        """
    )
    
    parser.add_argument("--model", type=str, default="RGCN",
                       help="Model type (default: RGCN)")
    parser.add_argument("--dataset", type=str, default="abortion",
                       help="Dataset name (default: abortion)")
    parser.add_argument("--max_trials", type=int, default=20,
                       help="Maximum number of trials (default: 20)")
    parser.add_argument("--all_models", action="store_true",
                       help="Run hyperparameter search for all models")
    parser.add_argument("--models", type=str,
                       help="Comma-separated list of models (e.g., RGCN,HAN)")
    
    return parser.parse_args()


if __name__ == "__main__":
    _set_deterministic_behavior()
    args = _parse_arguments()
    
    if args.all_models or args.models:
        models = None
        if args.models:
            models = [m.strip() for m in args.models.split(',')]
        
        run_all_models_hyperparameter_search(args.dataset, models, args.max_trials)
    else:
        print(f"🔍 Running hyperparameter search for {args.model} on {args.dataset}")
        print(f"Max trials: {args.max_trials}")
        
        best_config, results_df = run_comprehensive_hyperparameter_search(
            args.model, args.dataset, args.max_trials
        )
        
        if best_config is not None:
            print(f"\n🎉 Hyperparameter search completed successfully!")
            print(f"🎯 Run final training with optimized hyperparameters:")
            print(f"   python train.py --model {args.model} --dataset {args.dataset}")
        else:
            print(f"\n❌ Hyperparameter search failed to find valid configuration")