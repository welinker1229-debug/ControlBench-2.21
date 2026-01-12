import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gc
from models import (
    RGCNNodeClassifier, 
    HANNodeClassifier, 
    HGTNodeClassifier,
    HinSAGENodeClassifier,
    H2GFormerNodeClassifier,
    # LLM4HeGNodeClassifier,
)

def set_seeds(seed=42):
    """
    Set seeds for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def clear_gpu_memory():
    """
    Clear GPU memory and run garbage collection.
    Essential for hyperparameter tuning to prevent OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def get_memory_usage():
    """
    Get current GPU memory usage for monitoring.
    
    Returns:
        Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def save_model(model, path):
    """
    Save model to disk.
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load model from disk.
    
    Args:
        model: PyTorch model
        path: Path to load the model from
        
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def plot_training_curve(train_metrics, val_metrics=None, metric_name="Metric", save_path=None):
    """
    Plot training curve with validation metrics for 3-way split.
    
    Args:
        train_metrics: List of training metrics
        val_metrics: List of validation metrics (optional)
        metric_name: Name of the metric
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}', marker='o', markersize=4)
    if val_metrics is not None:
        plt.plot(val_metrics, label=f'Validation {metric_name}', marker='s', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at the best value
    if val_metrics is not None:
        best_val = max(val_metrics)
        best_epoch = val_metrics.index(best_val)
        plt.axhline(y=best_val, color='r', linestyle='--', alpha=0.5)
        plt.plot(best_epoch, best_val, 'ro', markersize=8)
        plt.annotate(f'Best: {best_val:.4f}', 
                     xy=(best_epoch, best_val),
                     xytext=(best_epoch+1, best_val),
                     arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
    else:
        plt.show()

def initialize_model(model_type, config, num_classes):
    """
    Initialize the specified conversation-aware GNN model.
    Updated to handle conversation parameters and separate embedding dimensions.
    Includes memory-efficient initialization for hyperparameter tuning.
    
    Args:
        model_type: Type of GNN model
        config: Model configuration
        num_classes: Number of output classes from the dataset
        
    Returns:
        Initialized model with conversation awareness
    """
    # Get embedding dimensions
    in_feats = config.get('in_feats', 768)  # User node features
    edge_feats = config.get('edge_feats', 768)  # Edge features
    post_feats = config.get('post_feats', 1536)  # Post node features might be larger
    
    # Get conversation parameters
    use_parent_context = config.get('use_parent_context', True)
    conversation_weight = config.get('conversation_weight', 0.4)
    
    # Clear memory before initializing new model
    clear_gpu_memory()
    
    if model_type == "RGCN":
        model = RGCNNodeClassifier(
            in_feats=in_feats,
            edge_feats=edge_feats,
            hidden_size=config['hidden_size'],
            out_classes=num_classes,
            dropout=config['dropout'],
            n_layers=config['n_layers'],
            num_rels=config.get('num_rels', 3),
            post_feats=post_feats,
            use_parent_context=use_parent_context,
            conversation_weight=conversation_weight
        )
    elif model_type == "HAN":
        model = HANNodeClassifier(
            in_feats=in_feats,
            edge_feats=edge_feats,
            hidden_size=config['hidden_size'],
            out_classes=num_classes,
            dropout=config['dropout'],
            n_layers=config['n_layers'],
            num_heads=config.get('num_heads', 8),
            post_feats=post_feats,
            use_parent_context=use_parent_context,
            conversation_weight=conversation_weight
        )
    elif model_type == "HGT":
        model = HGTNodeClassifier(
            in_feats=in_feats,
            edge_feats=edge_feats,
            hidden_size=config['hidden_size'],
            out_classes=num_classes,
            dropout=config['dropout'],
            n_layers=config['n_layers'],
            num_heads=config.get('num_heads', 4),
            num_ntypes=config.get('num_ntypes', 2),
            num_etypes=config.get('num_etypes', 3),
            post_feats=post_feats,
            use_parent_context=use_parent_context,
            conversation_weight=conversation_weight
        )
    elif model_type == "HinSAGE":
        model = HinSAGENodeClassifier(
            in_feats=in_feats,
            edge_feats=edge_feats,
            hidden_size=config['hidden_size'],
            out_classes=num_classes,
            dropout=config['dropout'],
            n_layers=config['n_layers'],
            aggregator_type=config.get('aggregator_type', 'mean'),
            post_feats=post_feats,
            use_parent_context=use_parent_context,
            conversation_weight=conversation_weight
        )
    elif model_type == "H2GFormer":
        model = H2GFormerNodeClassifier(
            in_feats=in_feats,
            edge_feats=edge_feats,
            hidden_size=config['hidden_size'],
            out_classes=num_classes,
            dropout=config['dropout'],
            n_layers=config['n_layers'],
            num_heads=config.get('num_heads', 8),
            post_feats=post_feats,
            layers_pre_gt=config.get('layers_pre_gt', 1),
            layers_post_gt=config.get('layers_post_gt', 1),
            edge_weight=config.get('edge_weight', False),
            num_classes=config.get('num_classes', None),
            use_label_emb=config.get('use_label_emb', False)
        )
    # elif model_type == "LLM4HeG":
    #     model = LLM4HeGNodeClassifier(
    #         in_feats=in_feats,
    #         edge_feats=edge_feats,
    #         hidden_size=config['hidden_size'],
    #         out_classes=num_classes,
    #         dropout=config['dropout'],
    #         n_layers=config['n_layers'],
    #         eps=config.get('eps', 0.3),
    #         post_feats=post_feats,
    #         num_classes=config.get('num_classes', None),
    #         use_label_emb=config.get('use_label_emb', False),
    #         alpha_yn=config.get('alpha_yn', 0.3),
    #         lambda_reg=config.get('lambda_reg', 0.1)
    #     )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Log model initialization
    param_count = count_parameters(model)
    memory_usage = get_memory_usage()
    
    if use_parent_context:
        print(f"✓ Initialized {model_type} with conversation awareness (weight: {conversation_weight})")
    else:
        print(f"ℹ Initialized {model_type} in standard mode (no conversation context)")
    
    print(f"  Parameters: {param_count:,}")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {memory_usage['allocated']:.2f}GB allocated, {memory_usage['reserved']:.2f}GB reserved")
    
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_features(train_graph, val_graph, test_graph):
    """
    Analyze the features of the graph data including conversation context.
    Updated for 3-way split (train/validation/test).
    
    Args:
        train_graph: DGL graph for training
        val_graph: DGL graph for validation
        test_graph: DGL graph for testing
        
    Returns:
        Dictionary containing feature statistics including conversation metrics
    """
    stats = {}
    
    # Analyze node features
    for ntype in train_graph.ntypes:
        if "feat" in train_graph.nodes[ntype].data:
            feat = train_graph.nodes[ntype].data["feat"]
            stats[f"{ntype}_feat_shape"] = tuple(feat.shape)
            stats[f"{ntype}_feat_min"] = float(feat.min())
            stats[f"{ntype}_feat_max"] = float(feat.max())
            stats[f"{ntype}_feat_mean"] = float(feat.mean())
            stats[f"{ntype}_feat_std"] = float(feat.std())
    
    # Analyze edge features including conversation context
    for etype in train_graph.canonical_etypes:
        src, rel, dst = etype
        if "feat" in train_graph.edges[rel].data:
            feat = train_graph.edges[rel].data["feat"]
            stats[f"{rel}_feat_shape"] = tuple(feat.shape)
            stats[f"{rel}_feat_min"] = float(feat.min())
            stats[f"{rel}_feat_max"] = float(feat.max())
            stats[f"{rel}_feat_mean"] = float(feat.mean())
            stats[f"{rel}_feat_std"] = float(feat.std())
        
        # Analyze parent content features for conversation edges
        if rel == "user_comment_user" and "parent_feat" in train_graph.edges[rel].data:
            parent_feat = train_graph.edges[rel].data["parent_feat"]
            stats[f"{rel}_parent_feat_shape"] = tuple(parent_feat.shape)
            stats[f"{rel}_parent_feat_min"] = float(parent_feat.min())
            stats[f"{rel}_parent_feat_max"] = float(parent_feat.max())
            stats[f"{rel}_parent_feat_mean"] = float(parent_feat.mean())
            stats[f"{rel}_parent_feat_std"] = float(parent_feat.std())
            stats["conversation_aware"] = True
    
    # Count nodes and edges for all splits
    for split_name, graph in [("train", train_graph), ("val", val_graph), ("test", test_graph)]:
        for ntype in graph.ntypes:
            stats[f"{split_name}_{ntype}_count"] = graph.num_nodes(ntype)
        
        for etype in graph.canonical_etypes:
            src, rel, dst = etype
            stats[f"{split_name}_{rel}_count"] = graph.num_edges(etype)
    
    # Conversation-specific statistics
    if "conversation_aware" in stats:
        train_conversation_edges = train_graph.num_edges(("user", "user_comment_user", "user"))
        val_conversation_edges = val_graph.num_edges(("user", "user_comment_user", "user"))
        test_conversation_edges = test_graph.num_edges(("user", "user_comment_user", "user"))
        
        stats["train_conversation_edges_count"] = train_conversation_edges
        stats["val_conversation_edges_count"] = val_conversation_edges
        stats["test_conversation_edges_count"] = test_conversation_edges
        stats["total_conversation_edges"] = train_conversation_edges + val_conversation_edges + test_conversation_edges
        
        # Calculate conversation coverage
        total_users = train_graph.num_nodes("user") + val_graph.num_nodes("user") + test_graph.num_nodes("user")
        stats["conversation_coverage"] = stats["total_conversation_edges"] / max(1, total_users)
    
    return stats

def analyze_conversation_patterns(graph, labels, idx_to_flair):
    """
    Analyze conversation patterns in the graph for debate understanding.
    
    Args:
        graph: DGL graph with conversation edges
        labels: User labels/stances
        idx_to_flair: Mapping from label indices to stance names
        
    Returns:
        Dictionary with conversation pattern statistics
    """
    conversation_stats = {}
    
    if graph.num_edges(("user", "user_comment_user", "user")) == 0:
        return {"conversation_aware": False}
    
    # Get user-comment-user edges
    src_users, dst_users = graph.edges(etype="user_comment_user")
    
    # Analyze stance interactions
    stance_interactions = {}
    for src, dst in zip(src_users.tolist(), dst_users.tolist()):
        if src < len(labels) and dst < len(labels):
            src_stance = idx_to_flair.get(labels[src].item(), "Unknown")
            dst_stance = idx_to_flair.get(labels[dst].item(), "Unknown")
            
            interaction_type = f"{src_stance} -> {dst_stance}"
            stance_interactions[interaction_type] = stance_interactions.get(interaction_type, 0) + 1
    
    conversation_stats["stance_interactions"] = stance_interactions
    conversation_stats["total_conversations"] = len(src_users)
    conversation_stats["conversation_aware"] = True
    
    # Calculate agreement vs disagreement rates
    agreement_count = 0
    disagreement_count = 0
    
    for interaction, count in stance_interactions.items():
        src_stance, dst_stance = interaction.split(" -> ")
        if src_stance == dst_stance:
            agreement_count += count
        else:
            disagreement_count += count
    
    total_interactions = agreement_count + disagreement_count
    if total_interactions > 0:
        conversation_stats["agreement_rate"] = agreement_count / total_interactions
        conversation_stats["disagreement_rate"] = disagreement_count / total_interactions
    
    return conversation_stats