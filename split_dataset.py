# split_dataset.py
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split



def validate_split_ratios(train_ratio, val_ratio, test_ratio):
    """Validate that split ratios sum to 1.0."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )


def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graph_data(dataset_name, data_dir):
    """Load graph data from JSON file."""
    json_file = os.path.join(data_dir, f"{dataset_name}_graph_data_with_embeddings.json")
    print(f"Loading dataset from {json_file}")
    
    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data["nodes"]
    edges = data["edges"]
    
    user_nodes = [node for node in nodes if node["type"] == "user"]
    post_nodes = [node for node in nodes if node["type"] == "post"]
    
    print(f"Total nodes: {len(nodes)} ({len(user_nodes)} users, {len(post_nodes)} posts)")
    print(f"Total edges: {len(edges)}")
    
    return user_nodes, post_nodes, edges


def filter_rare_classes(user_nodes, min_class_percentage):
    """
    Filter out classes that are below the minimum percentage threshold.
    
    Returns:
        valid_labels: List of labels that meet the threshold
        ignored_labels: List of (label, count, percentage) tuples for ignored classes
        valid_user_nodes: User nodes with valid labels
        valid_user_ids: Set of valid user IDs
    """
    label_counts = Counter([node["label"] for node in user_nodes])
    total_users = len(user_nodes)
    
    valid_labels = []
    ignored_labels = []
    
    for label, count in label_counts.items():
        percentage = count / total_users
        if percentage >= min_class_percentage:
            valid_labels.append(label)
        else:
            ignored_labels.append((label, count, percentage * 100))
    
    # Report ignored classes
    if ignored_labels:
        print(f"\nIgnoring {len(ignored_labels)} rare classes (below {min_class_percentage*100}% threshold):")
        for label, count, percentage in ignored_labels:
            print(f"  - '{label}': {count} instances ({percentage:.2f}% of dataset)")
    
    # Filter users with valid labels
    valid_user_nodes = [node for node in user_nodes if node["label"] in valid_labels]
    valid_user_ids = set(node["id"] for node in valid_user_nodes)
    
    print(f"\nRetained {len(valid_user_nodes)}/{len(user_nodes)} users with valid labels")
    
    return valid_labels, ignored_labels, valid_user_nodes, valid_user_ids


def perform_stratified_split(valid_user_nodes, val_ratio, test_ratio, seed):
    """
    Perform stratified split of users into train, validation, and test sets.
    
    Returns:
        train_users, val_users, test_users: Lists of user nodes for each split
        train_user_ids, val_user_ids, test_user_ids: Sets of user IDs for each split
    """
    
    # First split: separate train from (val+test)
    train_users, val_test_users = train_test_split(
        valid_user_nodes,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=[node["label"] for node in valid_user_nodes]
    )
    
    # Second split: separate val from test
    val_users, test_users = train_test_split(
        val_test_users,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=[node["label"] for node in val_test_users]
    )
    
    # Create user ID sets for efficient lookup
    train_user_ids = set(node["id"] for node in train_users)
    val_user_ids = set(node["id"] for node in val_users)
    test_user_ids = set(node["id"] for node in test_users)
    
    return (train_users, val_users, test_users,
            train_user_ids, val_user_ids, test_user_ids)


def print_split_statistics(valid_user_nodes, train_users, val_users, test_users, valid_labels):
    """Print statistics about the user splits."""
    print(f"Training users: {len(train_users)} ({len(train_users)/len(valid_user_nodes)*100:.1f}%)")
    print(f"Validation users: {len(val_users)} ({len(val_users)/len(valid_user_nodes)*100:.1f}%)")
    print(f"Test users: {len(test_users)} ({len(test_users)/len(valid_user_nodes)*100:.1f}%)")
    
    # Check label distribution across splits
    train_label_counts = Counter([node["label"] for node in train_users])
    val_label_counts = Counter([node["label"] for node in val_users])
    test_label_counts = Counter([node["label"] for node in test_users])
    
    print("\nLabel distribution across splits:")
    print(f"{'Label':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Train%':<8} {'Val%':<8} {'Test%':<8}")
    print("-" * 90)
    
    for label in sorted(valid_labels):
        train_count = train_label_counts.get(label, 0)
        val_count = val_label_counts.get(label, 0)
        test_count = test_label_counts.get(label, 0)
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            train_pct = train_count / total_count * 100
            val_pct = val_count / total_count * 100
            test_pct = test_count / total_count * 100
        else:
            train_pct = val_pct = test_pct = 0
        
        print(f"{label:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8} "
              f"{train_pct:.1f}%{' ':<3} {val_pct:.1f}%{' ':<3} {test_pct:.1f}%")
    
    return train_label_counts, val_label_counts, test_label_counts


def assign_edges_to_splits(edges, train_user_ids, val_user_ids, test_user_ids, valid_user_ids):
    """
    Assign edges to appropriate splits based on connected users.
    
    Returns:
        train_edges, val_edges, test_edges: Lists of edges for each split
        train_context_users, val_context_users, test_context_users: Sets of context user IDs
    """
    train_edges = []
    val_edges = []
    test_edges = []
    
    # Track context users (source users) needed for each split
    train_context_users = set()
    val_context_users = set()
    test_context_users = set()
    
    for edge in edges:
        src, dst, edge_type = edge["source"], edge["target"], edge["type"]
        
        # User-comment-user edges: target user determines the split
        if edge_type == "user_comment_user":
            if dst in train_user_ids and src in valid_user_ids:
                train_edges.append(edge)
                train_context_users.add(src)
            elif dst in val_user_ids and src in valid_user_ids:
                val_edges.append(edge)
                val_context_users.add(src)
            elif dst in test_user_ids and src in valid_user_ids:
                test_edges.append(edge)
                test_context_users.add(src)
        
        # User-post edges: user determines the split
        elif edge_type in ["user_publish_post", "user_comment_post"]:
            if src in train_user_ids:
                train_edges.append(edge)
            elif src in val_user_ids:
                val_edges.append(edge)
            elif src in test_user_ids:
                test_edges.append(edge)
    
    print(f"\nEdge distribution:")
    print(f"Train edges: {len(train_edges)}")
    print(f"Validation edges: {len(val_edges)}")
    print(f"Test edges: {len(test_edges)}")
    
    return (train_edges, val_edges, test_edges,
            train_context_users, val_context_users, test_context_users)


def add_context_users(user_nodes, train_context_users, val_context_users, test_context_users,
                      train_user_ids, val_user_ids, test_user_ids, valid_user_ids):
    """
    Add context users to appropriate splits if they're not already assigned.
    
    Returns:
        additional_train_users, additional_val_users, additional_test_users: Lists of context user nodes
    """
    additional_train_users = []
    additional_val_users = []
    additional_test_users = []
    
    for node in user_nodes:
        node_id = node["id"]
        if node_id not in valid_user_ids:
            continue
        
        # Check if user is already in a split
        already_assigned = (node_id in train_user_ids or 
                          node_id in val_user_ids or 
                          node_id in test_user_ids)
        
        if already_assigned:
            continue
        
        # Add to appropriate split if needed for context
        if node_id in train_context_users:
            additional_train_users.append(node)
        elif node_id in val_context_users:
            additional_val_users.append(node)
        elif node_id in test_context_users:
            additional_test_users.append(node)
    
    print(f"Additional context users - Train: {len(additional_train_users)}, "
          f"Val: {len(additional_val_users)}, Test: {len(additional_test_users)}")
    
    return additional_train_users, additional_val_users, additional_test_users


def assign_posts_to_splits(post_nodes, train_edges, val_edges, test_edges):
    """
    Assign post nodes to splits based on edges connecting users to posts.
    
    Returns:
        train_post_nodes, val_post_nodes, test_post_nodes: Lists of post nodes for each split
    """
    # Extract post IDs from edges
    train_post_ids = set()
    val_post_ids = set()
    test_post_ids = set()
    
    for edge in train_edges:
        if edge["type"] in ["user_publish_post", "user_comment_post"]:
            train_post_ids.add(edge["target"])
    
    for edge in val_edges:
        if edge["type"] in ["user_publish_post", "user_comment_post"]:
            val_post_ids.add(edge["target"])
    
    for edge in test_edges:
        if edge["type"] in ["user_publish_post", "user_comment_post"]:
            test_post_ids.add(edge["target"])
    
    # Assign post nodes to splits
    train_post_nodes = [node for node in post_nodes if node["id"] in train_post_ids]
    val_post_nodes = [node for node in post_nodes if node["id"] in val_post_ids]
    test_post_nodes = [node for node in post_nodes if node["id"] in test_post_ids]
    
    return train_post_nodes, val_post_nodes, test_post_nodes


def create_final_node_lists(train_users, val_users, test_users,
                           additional_train_users, additional_val_users, additional_test_users,
                           train_post_nodes, val_post_nodes, test_post_nodes):
    """Combine users and posts into final node lists for each split."""
    final_train_nodes = train_users + additional_train_users + train_post_nodes
    final_val_nodes = val_users + additional_val_users + val_post_nodes
    final_test_nodes = test_users + additional_test_users + test_post_nodes
    
    print(f"\nFinal node counts:")
    print(f"Train: {len(final_train_nodes)} ({len(train_users + additional_train_users)} users, {len(train_post_nodes)} posts)")
    print(f"Val: {len(final_val_nodes)} ({len(val_users + additional_val_users)} users, {len(val_post_nodes)} posts)")
    print(f"Test: {len(final_test_nodes)} ({len(test_users + additional_test_users)} users, {len(test_post_nodes)} posts)")
    
    return final_train_nodes, final_val_nodes, final_test_nodes


def create_split_data(dataset_name, split_name, nodes, edges, train_ratio, val_ratio, 
                     test_ratio, seed, min_class_percentage):
    """Create a data dictionary for a split."""
    return {
        "dataset": dataset_name,
        "split": split_name,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_class_percentage": min_class_percentage,
        "nodes": nodes,
        "edges": edges
    }


def create_metadata(dataset_name, train_ratio, val_ratio, test_ratio, seed, min_class_percentage,
                   user_nodes, valid_user_nodes, train_users, val_users, test_users,
                   additional_train_users, additional_val_users, additional_test_users,
                   valid_labels, ignored_labels, train_label_counts, val_label_counts, test_label_counts):
    """Create metadata dictionary with split statistics."""
    return {
        "dataset": dataset_name,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_class_percentage": min_class_percentage,
        "total_users": len(user_nodes),
        "valid_users": len(valid_user_nodes),
        "train_users": len(train_users),
        "val_users": len(val_users),
        "test_users": len(test_users),
        "additional_train_context_users": len(additional_train_users),
        "additional_val_context_users": len(additional_val_users),
        "additional_test_context_users": len(additional_test_users),
        "valid_labels": valid_labels,
        "ignored_labels": [(label, count) for label, count, _ in ignored_labels],
        "label_distribution": {
            "train": dict(train_label_counts),
            "validation": dict(val_label_counts),
            "test": dict(test_label_counts)
        }
    }


def save_split_datasets(dataset_output_dir, train_data, val_data, test_data, metadata):
    """Save all split datasets and metadata to JSON files."""
    train_file = os.path.join(dataset_output_dir, "train.json")
    val_file = os.path.join(dataset_output_dir, "validation.json")
    test_file = os.path.join(dataset_output_dir, "test.json")
    metadata_file = os.path.join(dataset_output_dir, "metadata.json")
    
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset split and saved successfully:")
    print(f"  - Train set: {train_file}")
    print(f"  - Validation set: {val_file}")
    print(f"  - Test set: {test_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"\n🎯 Ready for hyperparameter tuning with proper validation split!")
    
    return train_file, val_file, test_file, metadata_file


def split_and_save_dataset(dataset_name, data_dir="embedded_data", output_dir="split_datasets_3way", 
                           train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, min_class_percentage=0.05):
    """
    Split a dataset into training, validation, and test sets and save them as separate files.
    Uses 60:20:20 split for proper hyperparameter tuning.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing the data files
        output_dir: Directory to save the split datasets
        train_ratio: Ratio of training set (default: 0.6)
        val_ratio: Ratio of validation set (default: 0.2) 
        test_ratio: Ratio of test set (default: 0.2)
        seed: Random seed for reproducibility
        min_class_percentage: Minimum percentage for a class to be included
        
    Returns:
        Dictionary with file paths and user counts for each split
    """
    # Validate and setup
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    set_random_seed(seed)
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Load and prepare data
    user_nodes, post_nodes, edges = load_graph_data(dataset_name, data_dir)
    
    # Filter rare classes
    valid_labels, ignored_labels, valid_user_nodes, valid_user_ids = filter_rare_classes(
        user_nodes, min_class_percentage
    )
    
    # Perform stratified split
    (train_users, val_users, test_users,
     train_user_ids, val_user_ids, test_user_ids) = perform_stratified_split(
        valid_user_nodes, val_ratio, test_ratio, seed
    )
    
    # Print statistics
    train_label_counts, val_label_counts, test_label_counts = print_split_statistics(
        valid_user_nodes, train_users, val_users, test_users, valid_labels
    )
    
    # Assign edges to splits
    (train_edges, val_edges, test_edges,
     train_context_users, val_context_users, test_context_users) = assign_edges_to_splits(
        edges, train_user_ids, val_user_ids, test_user_ids, valid_user_ids
    )
    
    # Add context users
    additional_train_users, additional_val_users, additional_test_users = add_context_users(
        user_nodes, train_context_users, val_context_users, test_context_users,
        train_user_ids, val_user_ids, test_user_ids, valid_user_ids
    )
    
    # Assign posts to splits
    train_post_nodes, val_post_nodes, test_post_nodes = assign_posts_to_splits(
        post_nodes, train_edges, val_edges, test_edges
    )
    
    # Create final node lists
    final_train_nodes, final_val_nodes, final_test_nodes = create_final_node_lists(
        train_users, val_users, test_users,
        additional_train_users, additional_val_users, additional_test_users,
        train_post_nodes, val_post_nodes, test_post_nodes
    )
    
    # Create dataset dictionaries
    train_data = create_split_data(
        dataset_name, "train", final_train_nodes, train_edges,
        train_ratio, val_ratio, test_ratio, seed, min_class_percentage
    )
    val_data = create_split_data(
        dataset_name, "validation", final_val_nodes, val_edges,
        train_ratio, val_ratio, test_ratio, seed, min_class_percentage
    )
    test_data = create_split_data(
        dataset_name, "test", final_test_nodes, test_edges,
        train_ratio, val_ratio, test_ratio, seed, min_class_percentage
    )
    
    # Create metadata
    metadata = create_metadata(
        dataset_name, train_ratio, val_ratio, test_ratio, seed, min_class_percentage,
        user_nodes, valid_user_nodes, train_users, val_users, test_users,
        additional_train_users, additional_val_users, additional_test_users,
        valid_labels, ignored_labels, train_label_counts, val_label_counts, test_label_counts
    )
    
    # Save all files
    train_file, val_file, test_file, metadata_file = save_split_datasets(
        dataset_output_dir, train_data, val_data, test_data, metadata
    )
    
    return {
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "metadata_file": metadata_file,
        "train_users": len(train_users),
        "val_users": len(val_users),
        "test_users": len(test_users)
    }


def parse_arguments():
    """Parse command-line arguments."""
    defaults = {
        "dataset": "trump",
        "data_dir": "embedded_data",
        "output_dir": "split_datasets_3",
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 42,
        "min_class_percentage": 0.05
    }
    
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == "--dataset" and i+1 < len(sys.argv):
                defaults["dataset"] = sys.argv[i+1]
            elif sys.argv[i] == "--data_dir" and i+1 < len(sys.argv):
                defaults["data_dir"] = sys.argv[i+1]
            elif sys.argv[i] == "--output_dir" and i+1 < len(sys.argv):
                defaults["output_dir"] = sys.argv[i+1]
            elif sys.argv[i] == "--train_ratio" and i+1 < len(sys.argv):
                defaults["train_ratio"] = float(sys.argv[i+1])
            elif sys.argv[i] == "--val_ratio" and i+1 < len(sys.argv):
                defaults["val_ratio"] = float(sys.argv[i+1])
            elif sys.argv[i] == "--test_ratio" and i+1 < len(sys.argv):
                defaults["test_ratio"] = float(sys.argv[i+1])
            elif sys.argv[i] == "--seed" and i+1 < len(sys.argv):
                defaults["seed"] = int(sys.argv[i+1])
            elif sys.argv[i] == "--min_class_percentage" and i+1 < len(sys.argv):
                defaults["min_class_percentage"] = float(sys.argv[i+1])
    
    return defaults


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Splitting dataset: {args['dataset']}")
    print(f"Split ratios: Train={args['train_ratio']}, Val={args['val_ratio']}, Test={args['test_ratio']}")
    print(f"Minimum class percentage: {args['min_class_percentage']}")
    print(f"🔧 Using 3-way split for proper hyperparameter tuning")
    
    try:
        split_and_save_dataset(
            args["dataset"],
            args["data_dir"],
            args["output_dir"],
            args["train_ratio"],
            args["val_ratio"],
            args["test_ratio"],
            args["seed"],
            args["min_class_percentage"]
        )
    except ImportError:
        print("❌ scikit-learn is required for stratified splitting")
        print("Install with: pip install scikit-learn")
