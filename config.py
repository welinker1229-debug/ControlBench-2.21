"""
Enhanced configuration settings for conversation-aware GNN models with hyperparameter search spaces.
This file centralizes all hyperparameters and model configurations including
conversation-aware parameters and hyperparameter search spaces for proper tuning.
"""

# Define hyperparameter search spaces for each model type
HYPERPARAMETER_SEARCH_SPACES = {
    'RGCN': {
        'lr': [0.01, 0.005, 0.001],
        'weight_decay': [1e-4, 5e-4, 1e-3],
        'dropout': [0.3, 0.5, 0.7],
        'hidden_size': [128, 256, 512],
        'n_layers': [2, 3, 4],
        'conversation_weight': [0.2, 0.3, 0.4]  # For conversation-aware datasets
    },
    'HAN': {
        'lr': [0.01, 0.005, 0.001],
        'weight_decay': [1e-4, 5e-4, 1e-3],
        'dropout': [0.3, 0.5, 0.7],
        'hidden_size': [128, 256, 512],
        'n_layers': [2, 3, 4],
        'num_heads': [4, 8, 12],
        'conversation_weight': [0.15, 0.25, 0.35]
    },
    'HGT': {
        'lr': [0.01, 0.005, 0.001],
        'weight_decay': [1e-4, 5e-4, 1e-3],
        'dropout': [0.2, 0.4, 0.6],
        'hidden_size': [128, 256, 512],
        'n_layers': [2, 3, 4],
        'num_heads': [2, 4, 8],
        'conversation_weight': [0.25, 0.35, 0.45]
    },
    'HinSAGE': {
        'lr': [0.01, 0.005, 0.001],
        'weight_decay': [1e-4, 5e-4, 1e-3],
        'dropout': [0.3, 0.5, 0.7],
        'hidden_size': [128, 256, 512],
        'n_layers': [2, 3, 4],
        'aggregator_type': ['mean', 'pool', 'lstm'],
        'conversation_weight': [0.2, 0.3, 0.4]
    },
    'H2GFormer': {
        'lr': [0.001, 0.0005],
        'weight_decay': [1e-4, 1e-3],
        'dropout': [0.2, 0.5],
        'hidden_size': [128, 256],
        'n_layers': [2, 3],
        'num_heads': [4, 8],
        'conversation_weight': [0.0] # H2G-Former handles context via attention
    }
}

def get_hyperparameter_search_space(model_name, dataset_name=None):
    """
    Get hyperparameter search space for a specific model and dataset.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset (to filter conversation parameters)
        
    Returns:
        Dictionary with hyperparameter search space
    """
    if model_name not in HYPERPARAMETER_SEARCH_SPACES:
        raise ValueError(f"Unknown model for hyperparameter search: {model_name}")
    
    search_space = HYPERPARAMETER_SEARCH_SPACES[model_name].copy()
    
    # Remove conversation parameters for non-conversation datasets
    if dataset_name and not is_conversation_aware_dataset(dataset_name):
        search_space.pop('conversation_weight', None)
        print(f"ℹ Removed conversation parameters for dataset '{dataset_name}'")
    
    return search_space

def is_conversation_aware_dataset(dataset_name):
    """Check if dataset supports conversation awareness."""
    conversation_datasets = ['abortion', 'religion', 'capitalism', 'lgbtq', 'trump']
    return dataset_name.lower() in conversation_datasets

def get_config(model_name='RGCN', dataset_name='abortion'):
    """
    Get complete configuration by combining default, model-specific, and dataset-specific configs.
    Updated for 3-way split (train/validation/test) and hyperparameter tuning support.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with complete configuration
    """
    # Start with default configuration
    config = {
        # Data settings
        'embedding_dim': 768,    # BERT base embedding size
        'in_feats': 768,         # Input feature size (BERT dimension)
        'edge_feats': 768,       # Edge feature size (BERT dimension)
        'post_feats': 1536,      # Post features when title + content combined (2x768)
        
        # Model architecture - these are defaults, overridden by model-specific configs
        'hidden_size': 256,    
        'dropout': 0.5,        
        
        # Training settings (updated for hyperparameter tuning)
        'lr': 0.01,            # Learning rate (will be tuned)
        'weight_decay': 1e-4,  # Weight decay for regularization (will be tuned)
        'num_epochs': 100,     # Maximum number of epochs for final training
        'patience': 15,        # Patience for early stopping in final training
        'batch_size': 32,      # Batch size (for mini-batch training if implemented)
        
        # Hyperparameter tuning settings
        'tuning_epochs': 50,   # Shorter epochs for hyperparameter search
        'tuning_patience': 10, # Shorter patience for hyperparameter search
        'max_tuning_trials': 20, # Maximum hyperparameter combinations to try
        
        # Data split ratios (3-way split)
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        
        # Conversation-aware parameters
        'use_parent_context': True,      # Enable parent content usage
        'conversation_weight': 0.3,      # Balance between reply and parent content (will be tuned)
        'cross_attention': True,         # Enable cross-attention mechanisms
        'conversation_dropout': 0.1,     # Dropout for conversation components
    }
    
    # Add model-specific configuration WITH conversation parameters
    model_configs = {
        'RGCN': {
            'description': 'Conversation-aware Relational Graph Convolutional Network',
            'num_rels': 3,     # Number of relation types
            'n_layers': 2,     # Default layer count (will be tuned)
            'conversation_weight': 0.3,  # Parent vs reply content balance
            'use_parent_context': True,
        },
        
        'HAN': {
            'description': 'Conversation-aware Heterogeneous Graph Attention Network',
            'num_heads': 8,    # Number of attention heads (will be tuned)
            'n_layers': 2,     # Default layer count (will be tuned)
            'conversation_weight': 0.25, # Lower weight for HAN due to multiple attention mechanisms
            'use_parent_context': True,
        },
        
        'HGT': {
            'description': 'Conversation-aware Heterogeneous Graph Transformer',
            'num_heads': 4,    # Number of attention heads (will be tuned)
            'num_ntypes': 2,   # Number of node types ('user' and 'post')
            'num_etypes': 3,   # Number of edge types ('publish', 'comment', 'user_comment_user')
            'n_layers': 2,     # Default layer count (will be tuned)
            'conversation_weight': 0.35, # Higher weight for transformer-based models
            'use_parent_context': True,
        },
        
        'HinSAGE': {
            'description': 'Conversation-aware Heterogeneous GraphSAGE',
            'aggregator_type': 'mean',  # Options: 'mean', 'max', 'sum' (will be tuned)
            'n_layers': 2,     # Default layer count (will be tuned)
            'conversation_weight': 0.3,
            'use_parent_context': True,
        },

        'H2GFormer': {
            'description': 'H2G-Former: Sparse Graph Transformer (Local Attention)',
            'num_heads': 4,
            'n_layers': 2,
            'conversation_weight': 0.0,
            'use_parent_context': False, # Handles context via Sparse Attention
        }
    }
    
    # Get and apply model-specific settings
    if model_name in model_configs:
        config.update(model_configs[model_name])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Add dataset-specific configuration with conversation support
    dataset_configs = {
        'trump': {
            'description': 'Trump dataset with user-post interactions and conversation threads',
            'conversation_aware': True,
            'conversation_weight': 0.4,  # Higher weight for debate datasets
            'use_parent_context': True,
            'min_class_percentage': 0.01
        },
        
        'abortion': {
            'description': 'Abortion debate dataset with conversation threads and parent-reply context',
            'conversation_aware': True,   # Enhanced dataset with parent context
            'conversation_weight': 0.4,  # Higher weight for debate datasets
            'use_parent_context': True,
            'min_class_percentage': 0.05
        },
        
        'religion': {
            'description': 'Religion dataset with multiple religious belief classes and conversations',
            'conversation_aware': True,   # Enhanced dataset with parent context
            'conversation_weight': 0.4,  # Higher weight for debate datasets
            'use_parent_context': True,
            'min_class_percentage': 0.05
        },
        
        'capitalism': {
            'description': 'Political/Economic ideology dataset with conversation context',
            'conversation_aware': True,   # Enhanced dataset with parent context
            'conversation_weight': 0.4,  # Higher weight for debate datasets
            'use_parent_context': True,
            'min_class_percentage': 0.05
        },
        
        'lgbtq': {
            'description': 'LGBTQ dataset with identity discussions and conversation threads',
            'conversation_aware': True,   # Enhanced dataset with parent context
            'conversation_weight': 0.4,  # Higher weight for debate datasets
            'use_parent_context': True,
            'min_class_percentage': 0.03
        },
    }
    
    # Apply dataset-specific settings
    if dataset_name in dataset_configs:
        dataset_config = dataset_configs[dataset_name]
        config.update(dataset_config)
        
        # Override conversation settings based on dataset capability
        if not dataset_config.get('conversation_aware', False):
            config['use_parent_context'] = False
            config['conversation_weight'] = 0.0
            print(f"ℹ Dataset '{dataset_name}' doesn't support conversation awareness - disabling parent context")
        else:
            print(f"✓ Dataset '{dataset_name}' supports conversation awareness - enabling parent context")
    else:
        # Default settings for unknown datasets
        config.update({
            'description': 'Custom dataset',
            'conversation_aware': False,
            'use_parent_context': False,
            'min_class_percentage': 0.01
        })
        print(f"ℹ Unknown dataset '{dataset_name}' - disabling conversation features")
    
    return config

def print_config(config):
    """
    Print configuration in a readable format with conversation parameters highlighted.
    Updated to show hyperparameter tuning settings.
    
    Args:
        config: Configuration dictionary
    """
    print("Configuration:")
    print("=" * 60)
    
    # Print by categories with conversation-aware and tuning sections
    categories = {
        "Model Architecture": ["in_feats", "edge_feats", "post_feats", "hidden_size", "n_layers", "dropout", "num_heads", "num_ntypes", "num_etypes"],
        "Training Settings": ["lr", "weight_decay", "num_epochs", "patience", "batch_size"],
        "Hyperparameter Tuning": ["tuning_epochs", "tuning_patience", "max_tuning_trials"],
        "Data Splits": ["train_ratio", "val_ratio", "test_ratio", "min_class_percentage"],
        "Conversation-Aware": ["use_parent_context", "conversation_weight", "cross_attention", "conversation_dropout", "conversation_aware"],
        "Model-Specific": ["aggregator_type", "attn_dropout", "num_rels", "num_bases", "self_loop"]
    }
    
    for category, params in categories.items():
        category_params = []
        for param in params:
            if param in config:
                category_params.append((param, config[param]))
        
        if category_params:  # Only show categories that have parameters
            print(f"\n{category}:")
            print("-" * 60)
            for param, value in category_params:
                if param in ["use_parent_context", "conversation_aware"] and value:
                    print(f"  {param}: {value} ✓")  # Highlight enabled conversation features
                elif param == "conversation_weight" and value > 0:
                    print(f"  {param}: {value} (parent context: {int(value*100)}%, reply: {int((1-value)*100)}%)")
                else:
                    print(f"  {param}: {value}")
    
    # Print any remaining parameters not in the categories
    all_params = [p for params in categories.values() for p in params]
    remaining = {k: v for k, v in config.items() if k not in all_params and k != "description"}
    
    if remaining:
        print("\nOther Settings:")
        print("-" * 60)
        for param, value in remaining.items():
            print(f"  {param}: {value}")
    
    print("\nDescription:")
    print("-" * 60)
    if "description" in config:
        print(f"  {config['description']}")
    
    # Status summary
    if config.get('use_parent_context', False):
        print(f"\n🗣️  CONVERSATION-AWARE MODE ENABLED")
        print(f"   Parent context weight: {config.get('conversation_weight', 0.0)}")
        print(f"   Cross-attention: {config.get('cross_attention', False)}")
    else:
        print(f"\n📝 Standard mode (no conversation context)")
    
    print(f"\n🔧 3-WAY SPLIT: Train={config.get('train_ratio', 0.6)*100:.0f}%, "
          f"Val={config.get('val_ratio', 0.2)*100:.0f}%, Test={config.get('test_ratio', 0.2)*100:.0f}%")
    print(f"🎯 HYPERPARAMETER TUNING: {config.get('tuning_epochs', 50)} epochs, "
          f"max {config.get('max_tuning_trials', 20)} trials")
    
    print("=" * 60)

if __name__ == "__main__":
    # Simple usage without argparse
    import sys
    
    # Default values
    model = "RGCN"
    dataset = "abortion"  # Use conversation-aware dataset as default
    
    # Manual argument parsing
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == "--model" and i+1 < len(sys.argv):
                model = sys.argv[i+1]
            elif sys.argv[i] == "--dataset" and i+1 < len(sys.argv):
                dataset = sys.argv[i+1]
            elif sys.argv[i] == "--show_search_space":
                # Show hyperparameter search space
                try:
                    search_space = get_hyperparameter_search_space(model, dataset)
                    print(f"\nHyperparameter Search Space for {model} on {dataset}:")
                    print("=" * 60)
                    for param, values in search_space.items():
                        print(f"  {param}: {values}")
                    print("=" * 60)
                except Exception as e:
                    print(f"Error getting search space: {e}")
                sys.exit(0)
    
    # Get and print configuration
    config = get_config(model, dataset)
    print_config(config)
    
    # Also show search space
    try:
        search_space = get_hyperparameter_search_space(model, dataset)
        print(f"\nAvailable Hyperparameter Search Space:")
        print("-" * 60)
        for param, values in search_space.items():
            print(f"  {param}: {values}")
    except Exception as e:
        print(f"Search space unavailable: {e}")