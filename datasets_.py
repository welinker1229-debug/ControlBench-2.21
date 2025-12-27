import torch
import dgl
import json
import numpy as np
import random

# Set all random seeds for deterministic behavior
def set_deterministic_behavior(seed=42):
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Force single-threading
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set DGL's random seed
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except:
        print("Note: DGL version does not support direct random seed setting")
    
    # Set environment variables
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def load_3way_split_graph_data(train_file, val_file, test_file):
    """
    Load pre-split train, validation, and test graph data from JSON files.
    
    Args:
        train_file: Path to training data JSON file
        val_file: Path to validation data JSON file
        test_file: Path to test data JSON file
        
    Returns:
        train_graph: DGL graph for training
        val_graph: DGL graph for validation
        test_graph: DGL graph for testing
        train_labels: Node labels for training
        val_labels: Node labels for validation
        test_labels: Node labels for testing
        flair_to_idx: Mapping from node labels to indices
        idx_to_flair: Mapping from indices to node labels
        train_user_map: User mapping for training
        val_user_map: User mapping for validation
        test_user_map: User mapping for testing
    """
    # Set deterministic behavior first
    set_deterministic_behavior(42)

    # Load training data
    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    # Load validation data
    with open(val_file, "r") as f:
        val_data = json.load(f)
    
    # Load test data
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Process training data first to establish flair mapping
    train_graph, train_user_nodes, train_user_map, train_labels, flair_to_idx = process_graph_data(train_data)
    
    # Process validation data with same flair mapping
    val_graph, val_user_nodes, val_user_map, val_labels, _ = process_graph_data(
        val_data, existing_flair_map=flair_to_idx)
    
    # Process test data with same flair mapping
    test_graph, test_user_nodes, test_user_map, test_labels, _ = process_graph_data(
        test_data, existing_flair_map=flair_to_idx)
    
    idx_to_flair = {i: flair for flair, i in flair_to_idx.items()}
    
    return (train_graph, val_graph, test_graph, 
            train_labels, val_labels, test_labels,
            flair_to_idx, idx_to_flair, 
            train_user_map, val_user_map, test_user_map)

def load_split_graph_data(train_file, test_file=None):
    """
    Load pre-split train and test graph data from JSON files with conversation support.
    Maintained for backward compatibility.
    
    Args:
        train_file: Path to training data JSON file
        test_file: Path to test data JSON file (optional)
        
    Returns:
        train_graph: DGL graph for training
        test_graph: DGL graph for testing (if test_file provided)
        train_labels: Node labels for training
        test_labels: Node labels for testing (if test_file provided)
        flair_to_idx: Mapping from node labels to indices
        idx_to_flair: Mapping from indices to node labels
    """
    # Set deterministic behavior first
    set_deterministic_behavior(42)

    # Load training data
    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    # Process training data
    train_graph, train_user_nodes, train_user_map, train_labels, flair_to_idx = process_graph_data(train_data)
    
    if test_file:
        # Load test data
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        # Process test data (use same flair_to_idx mapping)
        test_graph, test_user_nodes, test_user_map, test_labels, _ = process_graph_data(
            test_data, existing_flair_map=flair_to_idx)
        
        idx_to_flair = {i: flair for flair, i in flair_to_idx.items()}
        
        return (train_graph, test_graph, train_labels, test_labels, 
                flair_to_idx, idx_to_flair, train_user_map, test_user_map)
    else:
        idx_to_flair = {i: flair for flair, i in flair_to_idx.items()}
        return train_graph, train_labels, flair_to_idx, idx_to_flair, train_user_map

def process_graph_data(data, existing_flair_map=None):
    """
    Process JSON graph data into DGL format with enhanced conversation handling.
    Fixed to ensure node-feature alignment.
    
    Args:
        data: JSON data with nodes and edges
        existing_flair_map: Existing mapping from flairs to indices (optional)
        
    Returns:
        dgl_graph: DGL heterogeneous graph
        user_nodes: Dictionary of user nodes
        user_map: Mapping from user IDs to indices
        labels: Node labels
        flair_to_idx: Mapping from node labels to indices
    """
    nodes = data["nodes"]
    edges = data["edges"]
    
    # Separate user and post nodes - MAINTAIN ORDER
    user_nodes = {}
    post_nodes = {}
    
    # Process nodes in original order to maintain consistency
    for node in nodes:
        if node["type"] == "user":
            user_nodes[node["id"]] = node
        elif node["type"] == "post":
            post_nodes[node["id"]] = node
    
    # Create ordered lists to ensure consistent indexing
    user_ids_ordered = list(user_nodes.keys())
    post_ids_ordered = list(post_nodes.keys())
    
    num_users = len(user_nodes)
    num_posts = len(post_nodes)
    
    print(f"Detected {num_users} users and {num_posts} posts in {data.get('split', 'unknown')} set")
    
    # Create deterministic mappings based on ordered lists
    user_map = {user_id: i for i, user_id in enumerate(user_ids_ordered)}
    post_map = {post_id: i for i, post_id in enumerate(post_ids_ordered)}
    
    # Validate mappings
    assert len(user_map) == num_users, f"User mapping mismatch: {len(user_map)} != {num_users}"
    assert len(post_map) == num_posts, f"Post mapping mismatch: {len(post_map)} != {num_posts}"
    
    # Process edges with enhanced conversation handling
    publish_edges, comment_edges, user_comment_user_edges = [], [], []
    comment_features, user_comment_user_features, parent_features = [], [], []
    
    # Track valid edges only (edges where both nodes exist)
    valid_edge_count = 0
    invalid_edge_count = 0
    
    for edge in edges:
        src, dst, edge_type = edge["source"], edge["target"], edge["type"]
        
        if edge_type == "user_publish_post":
            if src in user_map and dst in post_map:
                publish_edges.append((user_map[src], post_map[dst]))
                valid_edge_count += 1
            else:
                invalid_edge_count += 1
                
        elif edge_type == "user_comment_post":
            if src in user_map and dst in post_map:
                comment_edges.append((user_map[src], post_map[dst]))
                if "embedding" in edge:
                    comment_features.append(torch.tensor(edge["embedding"]))
                valid_edge_count += 1
            else:
                invalid_edge_count += 1
                
        elif edge_type == "user_comment_user":
            if src in user_map and dst in user_map:
                # Note: DGL expects (src, dst), so we use (dst, src) as the correct direction is target -> source for user-comment-user
                user_comment_user_edges.append((user_map[dst], user_map[src]))
                
                # Handle both reply content and parent content embeddings
                if "embedding" in edge:
                    parent_features.append(torch.tensor(edge["embedding"]))
                if "reply_embedding" in edge:
                    user_comment_user_features.append(torch.tensor(edge["reply_embedding"]))
                    
                valid_edge_count += 1
            else:
                invalid_edge_count += 1
    
    if invalid_edge_count > 0:
        print(f"Warning: Skipped {invalid_edge_count} invalid edges (nodes not found)")
    
    print(f"Valid edges: {valid_edge_count}")
    print(f"Edge type distribution: publish={len(publish_edges)}, comment={len(comment_edges)}, user_comment_user={len(user_comment_user_edges)}")
    
    # Convert to DGL graph format
    graph_data = {}
    if publish_edges:
        graph_data[("user", "publish", "post")] = publish_edges
    if comment_edges:
        graph_data[("user", "comment", "post")] = comment_edges
    if user_comment_user_edges:
        graph_data[("user", "user_comment_user", "user")] = user_comment_user_edges
    
    # Create DGL graph
    dgl_graph = dgl.heterograph(graph_data)
    
    print(f"Created DGL graph with {dgl_graph.num_nodes('user')} user nodes and {dgl_graph.num_nodes('post')} post nodes")
    
    # Process post node features - ENSURE CORRECT COUNT
    post_features = _process_post_features(post_ids_ordered, user_nodes, post_nodes)
    
    # Process user node features - ENSURE CORRECT COUNT  
    user_features = torch.zeros((len(user_ids_ordered), 768))  # Default BERT dimension
    
    # CRITICAL: Verify feature tensor sizes match graph node counts
    actual_user_nodes = dgl_graph.num_nodes('user')
    actual_post_nodes = dgl_graph.num_nodes('post')
    
    print(f"Graph nodes: {actual_user_nodes} users, {actual_post_nodes} posts")
    print(f"Feature tensors: {user_features.shape[0]} user features, {post_features.shape[0]} post features")
    
    # Ensure feature tensors match graph node counts
    if user_features.shape[0] != actual_user_nodes:
        print(f"Adjusting user features from {user_features.shape[0]} to {actual_user_nodes}")
        if user_features.shape[0] > actual_user_nodes:
            user_features = user_features[:actual_user_nodes]
        else:
            padding = torch.zeros((actual_user_nodes - user_features.shape[0], user_features.shape[1]))
            user_features = torch.cat([user_features, padding], dim=0)
    
    if post_features.shape[0] != actual_post_nodes:
        print(f"Adjusting post features from {post_features.shape[0]} to {actual_post_nodes}")
        if post_features.shape[0] > actual_post_nodes:
            post_features = post_features[:actual_post_nodes]
        else:
            padding = torch.zeros((actual_post_nodes - post_features.shape[0], post_features.shape[1]))
            post_features = torch.cat([post_features, padding], dim=0)
    
    # Assign node features to graph
    dgl_graph.ndata["feat"] = {
        "user": user_features,
        "post": post_features
    }
    
    # Process edge features
    _process_edge_features(dgl_graph, comment_features, user_comment_user_features, parent_features)
    
    # Create node labels using the ordered user IDs
    labels, flair_to_idx = _create_node_labels(user_ids_ordered, user_nodes, user_map, dgl_graph, existing_flair_map)
    
    # Log statistics
    _log_statistics(labels, len(user_comment_user_edges))
    
    return dgl_graph, user_nodes, user_map, labels, flair_to_idx

def _process_post_features(post_ids_ordered, user_nodes, post_nodes):
    """Process post node features with title and content embeddings - ensure correct count."""
    default_embedding_dim = 768
    post_features = []
    post_embedding_dim = default_embedding_dim
    
    # Process posts in the same order as post_ids_ordered
    for post_id in post_ids_ordered:
        post = post_nodes[post_id]
        
        if "embedding" in post and "title_embedding" in post:
            # Both title and content embeddings available
            content_feat = torch.tensor(post["embedding"], dtype=torch.float)
            title_feat = torch.tensor(post["title_embedding"], dtype=torch.float)
            
            if len(post_features) == 0:  # Log once
                print(f"Found both title and content embeddings. Content dim: {content_feat.shape}, Title dim: {title_feat.shape}")
            
            # Concatenate features
            combined_feat = torch.cat([title_feat, content_feat], dim=0)
            post_features.append(combined_feat)
            
            if combined_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = combined_feat.shape[0]
                
        elif "embedding" in post:
            # Only content embedding
            post_feat = torch.tensor(post["embedding"], dtype=torch.float)
            post_features.append(post_feat)
            if post_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = post_feat.shape[0]
                
        elif "title_embedding" in post:
            # Only title embedding
            post_feat = torch.tensor(post["title_embedding"], dtype=torch.float)
            post_features.append(post_feat)
            if post_feat.shape[0] > post_embedding_dim:
                post_embedding_dim = post_feat.shape[0]
        else:
            # No embeddings available
            post_features.append(torch.zeros(default_embedding_dim))
    
    # Ensure all features have same dimension
    if post_features:
        for i, feat in enumerate(post_features):
            if feat.shape[0] < post_embedding_dim:
                padded = torch.zeros(post_embedding_dim)
                padded[:feat.shape[0]] = feat
                post_features[i] = padded
        
        post_features = torch.stack(post_features)
        print(f"Post feature dimension: {post_features.shape[1]}")
        print(f"Created post features tensor: {post_features.shape}")
    else:
        post_features = torch.zeros((len(post_ids_ordered), default_embedding_dim))
        print(f"No post embeddings found, created zero tensor: {post_features.shape}")
    
    return post_features

def _process_edge_features(dgl_graph, comment_features, user_comment_user_features, parent_features):
    """Process and assign edge features to the graph."""
    
    # Process comment edge features
    if comment_features:
        comment_embedding_dim = max(feat.shape[0] for feat in comment_features)
        print(f"Comment edge feature dimension: {comment_embedding_dim}")
        
        # Ensure same dimension
        for i, feat in enumerate(comment_features):
            if feat.shape[0] < comment_embedding_dim:
                padded = torch.zeros(comment_embedding_dim)
                padded[:feat.shape[0]] = feat
                comment_features[i] = padded
        
        comment_features = torch.stack(comment_features)
        
        # Verify comment features match comment edges
        expected_comment_edges = dgl_graph.num_edges(('user', 'comment', 'post'))
        if comment_features.shape[0] != expected_comment_edges:
            print(f"Warning: Comment feature count mismatch. Expected {expected_comment_edges}, got {comment_features.shape[0]}")
            # Adjust if needed
            if comment_features.shape[0] > expected_comment_edges:
                comment_features = comment_features[:expected_comment_edges]
        
        dgl_graph.edges["comment"].data["feat"] = comment_features
    
    # Process user-comment-user edge features (reply content)
    if user_comment_user_features:
        ucu_embedding_dim = max(feat.shape[0] for feat in user_comment_user_features)
        print(f"User-comment-user edge feature dimension: {ucu_embedding_dim}")
        
        # Ensure same dimension
        for i, feat in enumerate(user_comment_user_features):
            if feat.shape[0] < ucu_embedding_dim:
                padded = torch.zeros(ucu_embedding_dim)
                padded[:feat.shape[0]] = feat
                user_comment_user_features[i] = padded
        
        user_comment_user_features = torch.stack(user_comment_user_features)
        
        # Verify user-comment-user features match edges
        expected_ucu_edges = dgl_graph.num_edges(('user', 'user_comment_user', 'user'))
        if user_comment_user_features.shape[0] != expected_ucu_edges:
            print(f"Warning: User-comment-user feature count mismatch. Expected {expected_ucu_edges}, got {user_comment_user_features.shape[0]}")
            # Adjust if needed
            if user_comment_user_features.shape[0] > expected_ucu_edges:
                user_comment_user_features = user_comment_user_features[:expected_ucu_edges]
        
        dgl_graph.edges["user_comment_user"].data["feat"] = user_comment_user_features
    
    # Process parent content features (conversation context)
    if parent_features:
        parent_embedding_dim = max(feat.shape[0] for feat in parent_features)
        print(f"Parent content feature dimension: {parent_embedding_dim}")
        
        # Ensure same dimension
        for i, feat in enumerate(parent_features):
            if feat.shape[0] < parent_embedding_dim:
                padded = torch.zeros(parent_embedding_dim)
                padded[:feat.shape[0]] = feat
                parent_features[i] = padded
        
        parent_features = torch.stack(parent_features)
        
        # Verify parent features match user-comment-user edges
        expected_ucu_edges = dgl_graph.num_edges(('user', 'user_comment_user', 'user'))
        if parent_features.shape[0] != expected_ucu_edges:
            print(f"Warning: Parent feature count mismatch. Expected {expected_ucu_edges}, got {parent_features.shape[0]}")
            # Adjust if needed
            if parent_features.shape[0] > expected_ucu_edges:
                parent_features = parent_features[:expected_ucu_edges]
        
        dgl_graph.edges["user_comment_user"].data["parent_feat"] = parent_features
        print(f"Added parent content features for {len(parent_features)} user-comment-user edges")

def _create_node_labels(user_ids_ordered, user_nodes, user_map, dgl_graph, existing_flair_map):
    """Create node labels and flair mapping - ensure correct count."""
    
    if existing_flair_map is None:
        # Create new flair mapping
        flairs = list(set(node["label"] for node in user_nodes.values()))
        flair_to_idx = {flair: i for i, flair in enumerate(flairs)}
    else:
        # Use existing flair mapping
        flair_to_idx = existing_flair_map
    
    # Create labels tensor with exact size matching graph
    actual_user_count = dgl_graph.num_nodes('user')
    labels = torch.full((actual_user_count,), -1, dtype=torch.long)
    
    print(f"Creating labels for {actual_user_count} users")
    
    # Fill labels using ordered user IDs
    for i, user_id in enumerate(user_ids_ordered):
        if i >= actual_user_count:
            print(f"Warning: Stopping label assignment at index {i} (graph has {actual_user_count} nodes)")
            break
            
        if user_id in user_nodes:
            node = user_nodes[user_id]
            if node["label"] in flair_to_idx:
                labels[i] = flair_to_idx[node["label"]]
            else:
                print(f"Warning: Unknown label '{node['label']}' for user {user_id}")
    
    return labels, flair_to_idx

def _log_statistics(labels, conversation_count):
    """Log dataset statistics."""
    
    # Count class distribution
    valid_labels = labels[labels != -1]
    if len(valid_labels) > 0:
        class_counts = torch.bincount(valid_labels)
        print(f"Class distribution: {class_counts.tolist()}")
        print(f"Valid labels: {len(valid_labels)} / {len(labels)}")
    else:
        print("Warning: No valid labels found!")
    
    # Log conversation statistics
    if conversation_count > 0:
        print(f"Detected {conversation_count} user-comment-user interactions")
        print("ℹ Conversation context available for modeling")