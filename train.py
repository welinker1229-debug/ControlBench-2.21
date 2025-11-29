import os
os.environ['DGL_DISABLE_GRAPHBOLT'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dgl
import json
import argparse
import time
import random
import itertools
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Import project modules
from models import (
    RGCNNodeClassifier, 
    HANNodeClassifier, 
    HGTNodeClassifier,
    HinSAGENodeClassifier
)
from config import get_config, print_config
from datasets import load_3way_split_graph_data
import utils

# Constants
SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_MAX_TRIALS = 20
DEFAULT_DATA_DIR = "split_datasets"
DEFAULT_OUTPUT_DIR = "experiments"

# Hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACE = {
    'lr': [0.01, 0.005, 0.001],
    'weight_decay': [1e-4, 5e-4, 1e-3],
    'dropout': [0.3, 0.5, 0.7],
    'hidden_size': [128, 256, 512],
    'n_layers': [2, 3, 4]
}


def _prepare_node_features(graph):
    """Extract node features from a graph."""
    return {
        "user": graph.nodes["user"].data["feat"],
        "post": graph.nodes["post"].data["feat"]
    }


def _prepare_edge_features(graph):
    """Extract edge features from a graph."""
    edge_features = {}
    if "feat" in graph.edges["comment"].data:
        edge_features["comment"] = graph.edges["comment"].data["feat"]
    if "feat" in graph.edges["user_comment_user"].data:
        edge_features["user_comment_user"] = graph.edges["user_comment_user"].data["feat"]
    return edge_features


def _prepare_features(graph):
    """Prepare both node and edge features from a graph."""
    return _prepare_node_features(graph), _prepare_edge_features(graph)


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


def _load_tuned_hyperparameters(model_type, dataset_name):
    """Load tuned hyperparameters if available."""
    tuning_result_path = f"tuning_results/{model_type}_{dataset_name}/best_config.json"
    try:
        with open(tuning_result_path, "r") as f:
            best_config = json.load(f)
            if "best_hyperparams" in best_config:
                return best_config["best_hyperparams"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _run_single_hyperparameter_trial(trial_idx, param_combination, param_names, model_type, 
                                      trial_config, num_classes, train_graph, val_graph, 
                                      train_labels, val_labels, total_trials):
    """Run a single hyperparameter trial."""
    print(f"\n📊 Trial {trial_idx + 1}/{total_trials}")
    trial_params = dict(zip(param_names, param_combination))
    print(f"Parameters: {trial_params}")
    
    try:
        trial_custom_config = {
            **trial_params,
            'num_epochs': DEFAULT_EPOCHS,
            'patience': DEFAULT_PATIENCE,
            'seed': SEED
        }
        
        val_f1, val_acc, train_time = train_single_trial(
            model_type, trial_config, num_classes,
            train_graph, val_graph, train_labels, val_labels,
            trial_custom_config
        )
        
        result = {
            'trial': trial_idx + 1,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'train_time': train_time,
            **trial_params
        }
        
        print(f"✅ Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Time: {train_time:.1f}s")
        return result, val_f1, trial_params
        
    except Exception as e:
        print(f"❌ Trial failed: {str(e)}")
        result = {
            'trial': trial_idx + 1,
            'val_f1': 0.0,
            'val_acc': 0.0,
            'train_time': 0.0,
            'error': str(e),
            **dict(zip(param_names, param_combination))
        }
        return result, 0.0, None


def _save_hyperparameter_results(results_dir, trial_results, best_config, best_val_f1,
                                model_type, dataset_name, search_space, num_trials, param_names):
    """Save hyperparameter search results."""
    import pandas as pd
    results_df = pd.DataFrame(trial_results)
    results_df.to_csv(os.path.join(results_dir, "hyperparameter_search_results.csv"), index=False)
    
    if best_config:
        with open(os.path.join(results_dir, "best_config.json"), "w") as f:
            json.dump({
                "model_type": model_type,
                "dataset": dataset_name,
                "best_config": best_config,
                "best_val_f1": best_val_f1,
                "search_space": search_space,
                "num_trials": num_trials
            }, f, indent=2)
        
        print(f"\n🏆 Best configuration found:")
        for key in param_names:
            print(f"  {key}: {best_config[key]}")
        print(f"  Validation F1: {best_val_f1:.4f}")
        
        # Save to tuning_results format for compatibility
        tuning_results_dir = f"tuning_results/{model_type}_{dataset_name}"
        os.makedirs(tuning_results_dir, exist_ok=True)
        with open(os.path.join(tuning_results_dir, "best_config.json"), "w") as f:
            json.dump({
                "model_type": model_type,
                "dataset": dataset_name,
                "best_hyperparams": {k: best_config[k] for k in param_names},
                "best_val_f1": best_val_f1
            }, f, indent=2)


def hyperparameter_search(model_type, dataset_name, search_space=None, max_trials=DEFAULT_MAX_TRIALS, data_dir=DEFAULT_DATA_DIR):
    """
    Perform hyperparameter search using validation set.
    
    Args:
        model_type: Type of GNN model
        dataset_name: Name of the dataset
        search_space: Dictionary of hyperparameters to search
        max_trials: Maximum number of trials to run
        data_dir: Directory containing split datasets
        
    Returns:
        best_config: Best hyperparameter configuration
        results_df: DataFrame with all trial results
    """
    if search_space is None:
        search_space = HYPERPARAMETER_SEARCH_SPACE
    
    print(f"🔍 Starting hyperparameter search for {model_type} on {dataset_name}")
    print(f"Search space: {search_space}, Max trials: {max_trials}")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_search_{model_type}_{dataset_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate combinations
    combinations_to_try, param_names = _generate_combinations(search_space, max_trials)
    print(f"🎯 Running {len(combinations_to_try)} hyperparameter trials")
    
    # Verify and load data
    _verify_dataset_files(data_dir, dataset_name)
    train_file = os.path.join(data_dir, dataset_name, "train.json")
    val_file = os.path.join(data_dir, dataset_name, "validation.json")
    test_file = os.path.join(data_dir, dataset_name, "test.json")
    
    (train_graph, val_graph, test_graph, 
     train_labels, val_labels, test_labels,
     flair_to_idx, idx_to_flair, 
     train_user_map, val_user_map, test_user_map) = load_3way_split_graph_data(train_file, val_file, test_file)
    
    num_classes = len(flair_to_idx)
    base_config = get_config(model_type, dataset_name)
    
    # Run trials
    trial_results = []
    best_val_f1 = 0
    best_config = None
    
    for trial_idx, param_combination in enumerate(combinations_to_try):
        trial_config = base_config.copy()
        trial_config.update(dict(zip(param_names, param_combination)))
        
        result, val_f1, trial_params = _run_single_hyperparameter_trial(
            trial_idx, param_combination, param_names, model_type,
            trial_config, num_classes, train_graph, val_graph,
            train_labels, val_labels, len(combinations_to_try)
        )
        trial_results.append(result)
        
        # Update best configuration
        if val_f1 > best_val_f1 and trial_params:
            best_val_f1 = val_f1
            best_config = trial_config.copy()
            print(f"🎉 New best! Val F1: {val_f1:.4f}")
    
    # Save results
    _save_hyperparameter_results(results_dir, trial_results, best_config, best_val_f1,
                                 model_type, dataset_name, search_space, len(combinations_to_try), param_names)
    
    print(f"\n📊 Results saved to: {results_dir}")
    import pandas as pd
    return best_config, pd.DataFrame(trial_results)

def _setup_loss_fn(train_labels, weighted=True, device=None):
    """Setup loss function with optional class weighting."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weighted:
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights * len(class_weights) / class_weights.sum()
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    return nn.CrossEntropyLoss()


def train_single_trial(model_type, config, num_classes, train_graph, val_graph, train_labels, val_labels, custom_config):
    """
    Train a single trial for hyperparameter search.
    
    Returns:
        val_f1: Validation F1 score
        val_acc: Validation accuracy  
        train_time: Training time in seconds
    """
    seed = custom_config.get('seed', SEED)
    set_deterministic_behavior(seed)
    config.update(custom_config)
    
    # Initialize model
    model = utils.initialize_model(model_type, config, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Move to device
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    
    # Prepare features
    train_node_features, train_edge_features = _prepare_features(train_graph)
    val_node_features, val_edge_features = _prepare_features(val_graph)
    
    # Setup training
    loss_fn = _setup_loss_fn(train_labels, weighted=True, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)
    
    # Training loop
    best_val_f1 = 0
    best_val_acc = 0
    patience_counter = 0
    num_epochs = config.get('num_epochs', DEFAULT_EPOCHS)
    patience = config.get('patience', DEFAULT_PATIENCE)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        logits = model(train_graph, train_node_features, train_edge_features)
        loss = loss_fn(logits, train_labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_graph, val_node_features, val_edge_features)
            val_preds = torch.argmax(val_logits, dim=1)
            
            val_labels_np = val_labels.cpu().numpy()
            val_preds_np = val_preds.cpu().numpy()
            
            val_acc = accuracy_score(val_labels_np, val_preds_np)
            val_f1 = f1_score(val_labels_np, val_preds_np, average='macro')
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    train_time = time.time() - start_time
    return best_val_f1, best_val_acc, train_time

def _load_and_merge_config(model_type, dataset_name, custom_config):
    """Load base config, merge with tuned hyperparameters and custom config."""
    config = get_config(model_type, dataset_name)
    
    # Load tuned hyperparameters if available
    tuned_params = _load_tuned_hyperparameters(model_type, dataset_name)
    if tuned_params:
        config.update(tuned_params)
        print(f"✅ Using tuned hyperparameters: {tuned_params}")
    else:
        print(f"ℹ No tuning results found, using default hyperparameters")
    
    # Override with custom config
    for key, value in custom_config.items():
        if key in config:
            config[key] = value
            print(f"Using custom {key}: {value}")
    
    return config


def _load_training_data(data_dir, dataset_name):
    """Load and verify training data."""
    _verify_dataset_files(data_dir, dataset_name)
    train_file = os.path.join(data_dir, dataset_name, "train.json")
    val_file = os.path.join(data_dir, dataset_name, "validation.json")
    test_file = os.path.join(data_dir, dataset_name, "test.json")
    
    return load_3way_split_graph_data(train_file, val_file, test_file)


def _initialize_training_metrics():
    """Initialize metrics dictionary for tracking training progress."""
    return {
        'best_val_acc': 0.0,
        'best_val_f1': 0.0,
        'best_val_micro_f1': 0.0,
        'final_test_acc': 0.0,
        'final_test_f1': 0.0,
        'final_test_micro_f1': 0.0,
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_acc': [],
        'val_f1': []
    }


def train_model(model_type, dataset_name, custom_config=None):
    """
    Main training function with validation set support.
    """
    if custom_config is None:
        custom_config = {}
    
    # Setup parameters
    seed = custom_config.get('seed', SEED)
    output_dir = custom_config.get('output_dir', DEFAULT_OUTPUT_DIR)
    data_dir = custom_config.get('data_dir', DEFAULT_DATA_DIR)
    weighted_loss = custom_config.get('weighted_loss', True)
    
    # Set deterministic behavior
    set_deterministic_behavior(seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{model_type}_{dataset_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load and merge configuration
    config = _load_and_merge_config(model_type, dataset_name, custom_config)
    
    # Save experiment configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        merged_config = {**config, "seed": seed, "weighted_loss": weighted_loss}
        json.dump(merged_config, f, indent=4)
    
    print_config(config)
    
    return_metrics = _initialize_training_metrics()
    
    try:
        # Load data
        (train_graph, val_graph, test_graph, 
         train_labels, val_labels, test_labels,
         flair_to_idx, idx_to_flair, 
         train_user_map, val_user_map, test_user_map) = _load_training_data(data_dir, dataset_name)
        
        num_classes = len(flair_to_idx)
        print(f"Number of classes: {num_classes}")
        
        # Print class distribution
        class_counts = torch.bincount(train_labels)
        for idx, count in enumerate(class_counts):
            flair = idx_to_flair.get(idx, f"Class {idx}")
            print(f"Class '{flair}': {count.item()} instances")
        
        # Update config with actual feature dimensions
        if "feat" in train_graph.nodes["post"].data:
            config['post_feats'] = train_graph.nodes["post"].data["feat"].shape[1]
        
        # Initialize model
        model = utils.initialize_model(model_type, config, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Move to device
        train_graph = train_graph.to(device)
        val_graph = val_graph.to(device)
        test_graph = test_graph.to(device)
        train_labels = train_labels.to(device)
        val_labels = val_labels.to(device)
        test_labels = test_labels.to(device)
        
        # Prepare features
        train_node_features, train_edge_features = _prepare_features(train_graph)
        val_node_features, val_edge_features = _prepare_features(val_graph)
        test_node_features, test_edge_features = _prepare_features(test_graph)
        
        # Setup training
        loss_fn = _setup_loss_fn(train_labels, weighted=weighted_loss, device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # Training loop with validation
        best_val_f1 = 0
        best_model = None
        patience_counter = 0
        train_losses = []
        train_accs = []
        train_f1s = []
        val_accs = []
        val_f1s = []
        
        num_epochs = custom_config.get('num_epochs', config.get('num_epochs', 100))
        patience = custom_config.get('patience', config.get('patience', 15))
        
        print(f"Starting training for {num_epochs} epochs with validation...")
        
        pbar = tqdm(range(num_epochs), desc="Training")
        
        for epoch in pbar:
            # Train epoch
            loss, train_acc, train_f1 = train_epoch(
                model, train_graph, train_node_features, train_edge_features, 
                optimizer, loss_fn, train_labels)
            
            # Validate
            val_acc, val_f1, val_micro_f1 = evaluate(model, val_graph, val_node_features, val_edge_features, val_labels)
            
            scheduler.step(val_f1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss:.4f}",
                'Train F1': f"{train_f1:.4f}",
                'Val F1': f"{val_f1:.4f}"
            })
            
            # Save metrics
            train_losses.append(float(loss))
            train_accs.append(float(train_acc))
            train_f1s.append(float(train_f1))
            val_accs.append(float(val_acc))
            val_f1s.append(float(val_f1))
            
            # Check for best model
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_val_acc = float(val_acc)
                best_model = model.state_dict().copy()
                torch.save(best_model, os.path.join(exp_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model and evaluate on test set
        if best_model is not None:
            model.load_state_dict(best_model)
            print(f"\nLoaded best model with validation F1: {best_val_f1:.4f}")
        
        # Final test evaluation
        test_acc, test_f1, test_micro_f1, test_preds, detailed_metrics = evaluate(
            model, test_graph, test_node_features, test_edge_features, test_labels, idx_to_flair, detailed=True)
        
        # Update return metrics
        return_metrics.update({
            'train_loss': train_losses,
            'train_acc': train_accs,
            'train_f1': train_f1s,
            'val_acc': val_accs,
            'val_f1': val_f1s,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'final_test_acc': float(test_acc),
            'final_test_f1': float(test_f1),
            'final_test_micro_f1': float(test_micro_f1)
        })
        
        # Save metrics
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(return_metrics, f, indent=4)
        
        print(f"\n🎯 Final Results:")
        print(f"Best Validation F1: {best_val_f1:.4f}")
        print(f"Final Test F1: {test_f1:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return_metrics["error"] = str(e)
    
    return return_metrics

def set_deterministic_behavior(seed=42):
    """Set comprehensive deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass
    
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except:
        pass

def train_epoch(model, g, node_features, edge_features, optimizer, loss_fn, labels):
    """Train for one epoch."""
    model.train()
    
    # Forward pass
    logits = model(g, node_features, edge_features)
    loss = loss_fn(logits, labels)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        train_acc = accuracy_score(labels_np, preds_np)
        train_f1 = f1_score(labels_np, preds_np, average='macro')
    
    return loss.item(), train_acc, train_f1

def evaluate(model, g, node_features, edge_features, labels, idx_to_flair=None, detailed=False):
    """Evaluate the model."""
    model.eval()
    
    with torch.no_grad():
        logits = model(g, node_features, edge_features)
        preds = torch.argmax(logits, dim=1)
        
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        acc = accuracy_score(labels_np, preds_np)
        macro_f1 = f1_score(labels_np, preds_np, average='macro')
        micro_f1 = f1_score(labels_np, preds_np, average='micro')
        
        if detailed and idx_to_flair:
            cm = confusion_matrix(labels_np, preds_np)
            report = classification_report(
                labels_np, 
                preds_np,
                target_names=[idx_to_flair[i] for i in range(len(idx_to_flair))],
                digits=4,
                output_dict=True
            )
            
            class_metrics = {}
            for idx, flair in idx_to_flair.items():
                class_mask = labels == idx
                if class_mask.sum() > 0:
                    class_correct = (preds[class_mask] == idx).sum().item()
                    class_total = class_mask.sum().item()
                    class_acc = class_correct / class_total
                    class_metrics[flair] = {
                        'accuracy': class_acc,
                        'correct': class_correct,
                        'total': class_total
                    }
            
            detailed_metrics = {
                'confusion_matrix': cm,
                'classification_report': report,
                'class_metrics': class_metrics
            }
            
            return acc, macro_f1, micro_f1, preds, detailed_metrics
    
    return acc, macro_f1, micro_f1

def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN models with validation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:     python train.py --model HAN --dataset religion
  Hyperparameter:  python train.py --model RGCN --dataset abortion --hyperparameter_search
        """
    )
    
    parser.add_argument("--model", type=str, default="HAN",
                       help="Model type (default: HAN)")
    parser.add_argument("--dataset", type=str, default="religion",
                       help="Dataset name (default: religion)")
    parser.add_argument("--hyperparameter_search", action="store_true",
                       help="Run hyperparameter search before training")
    
    return parser.parse_args()


if __name__ == "__main__":
    set_deterministic_behavior(SEED)
    args = _parse_arguments()
    
    if args.hyperparameter_search:
        print(f"🔍 Running hyperparameter search for {args.model} on {args.dataset}")
        best_config, results_df = hyperparameter_search(args.model, args.dataset)
        if best_config:
            print(f"\n🎯 Now running full training with best configuration...")
            train_model(args.model, args.dataset)
    else:
        print(f"Running {args.model} on {args.dataset} dataset with validation")
        train_model(args.model, args.dataset)