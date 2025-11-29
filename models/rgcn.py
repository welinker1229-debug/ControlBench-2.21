import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class ConversationAwareRGCNLayer(nn.Module):
    """
    RGCN layer that incorporates conversation context in message passing
    """
    def __init__(self, in_feats, out_feats, num_rels, conversation_dim):
        super(ConversationAwareRGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_rels = num_rels
        
        # Standard relation-specific transformations
        self.rel_transforms = nn.ModuleDict({
            'publish': nn.Linear(in_feats, out_feats),
            'comment': nn.Linear(in_feats, out_feats), 
            'user_comment_user': nn.Linear(in_feats, out_feats)
        })
        
        # Conversation-aware message function for user-comment-user edges
        self.conversation_message_func = nn.Sequential(
            nn.Linear(in_feats + conversation_dim, out_feats),
            nn.LayerNorm(out_feats),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge feature integration
        self.edge_feature_proj = nn.ModuleDict({
            'comment': nn.Linear(conversation_dim, out_feats),
            'user_comment_user': nn.Linear(conversation_dim, out_feats)
        })
        
    def forward(self, g, h_dict, edge_features=None, user_context_features=None):
        """
        Forward pass with conversation-aware message passing
        
        Args:
            g: DGL graph
            h_dict: Node features by type
            edge_features: Edge features by type
            user_context_features: User context features (one per user, not per edge)
        """
        h_new = {}
        
        for src_type, rel_type, dst_type in g.canonical_etypes:
            if g.num_edges((src_type, rel_type, dst_type)) == 0:
                continue
            
            # Get subgraph for this relation
            sg = g[src_type, rel_type, dst_type]
            src_feat = h_dict[src_type]
            
            if rel_type == 'user_comment_user' and user_context_features is not None:
                # Special handling for conversation edges
                src_nodes, dst_nodes = sg.edges()
                
                # Create enhanced source features using user context
                enhanced_src_features = []
                for src_idx in src_nodes:
                    src_node_feat = src_feat[src_idx]
                    
                    # Get user context for this source node
                    if src_idx < user_context_features.size(0):
                        user_ctx = user_context_features[src_idx]
                        
                        # Combine node features with user's conversation context
                        enhanced_feat = torch.cat([src_node_feat, user_ctx], dim=0)
                        enhanced_message = self.conversation_message_func(enhanced_feat)
                    else:
                        # Fallback to standard transformation
                        enhanced_message = self.rel_transforms[rel_type](src_node_feat)
                    
                    enhanced_src_features.append(enhanced_message)
                
                if enhanced_src_features:
                    enhanced_src_features = torch.stack(enhanced_src_features)
                    
                    # FIXED: Set edge data instead of source node data
                    sg.edata['enhanced_msg'] = enhanced_src_features
                    
                    # Message passing with conversation-aware edge messages
                    sg.update_all(
                        message_func=fn.copy_e('enhanced_msg', 'm'),
                        reduce_func=fn.mean('m', 'h_neigh')
                    )
            else:
                # Standard RGCN message passing for other relations
                transformed_feat = self.rel_transforms[rel_type](src_feat)
                sg.srcdata['h_trans'] = transformed_feat
                
                # Enhanced message passing with edge features
                if (edge_features is not None and rel_type in edge_features and 
                    rel_type in self.edge_feature_proj):
                    
                    edge_feat = self.edge_feature_proj[rel_type](edge_features[rel_type])
                    sg.edata['e'] = edge_feat
                    
                    sg.update_all(
                        message_func=lambda edges: {'m': edges.src['h_trans'] * 0.7 + 
                                                      edges.data['e'] * 0.3},
                        reduce_func=fn.mean('m', 'h_neigh')
                    )
                else:
                    sg.update_all(
                        message_func=fn.copy_u('h_trans', 'm'),
                        reduce_func=fn.mean('m', 'h_neigh')
                    )
            
            # Collect results
            if 'h_neigh' in sg.dstdata:
                if dst_type not in h_new:
                    h_new[dst_type] = []
                h_new[dst_type].append(sg.dstdata['h_neigh'])
        
        return h_new

class RGCNNodeClassifier(nn.Module):
    """
    Conversation-aware RGCN: Incorporates Post→Parent Comment→Reply context in message passing
    """
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256, out_classes=9, 
                 dropout=0.5, n_layers=2, num_rels=3, post_feats=1536,
                 use_parent_context=True, conversation_weight=0.3):
        super(RGCNNodeClassifier, self).__init__()
        
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.post_feats = post_feats
        self.hidden_size = hidden_size
        self.out_classes = out_classes
        self.n_layers = n_layers
        self.num_rels = num_rels
        self.dropout = dropout
        self.use_parent_context = use_parent_context
        self.conversation_weight = conversation_weight
        
        # Input projection layers
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(self.post_feats, hidden_size)
        
        # Edge feature projections
        self.edge_projections = nn.ModuleDict({
            'comment': nn.Linear(edge_feats, hidden_size),
            'user_comment_user': nn.Linear(edge_feats, hidden_size)
        })
        
        # Conversation context encoder (Post → Parent Comment → Reply)
        if self.use_parent_context:
            self.conversation_encoder = nn.Sequential(
                nn.Linear(edge_feats + post_feats, hidden_size),  # parent_comment + post_content
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # User context encoder for comprehensive context
        self.user_context_encoder = nn.Sequential(
            nn.Linear(post_feats, hidden_size),  # For user's own posts and commented posts
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Conversation-aware RGCN layers
        self.rgcn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.rgcn_layers.append(
                ConversationAwareRGCNLayer(
                    in_feats=hidden_size,
                    out_feats=hidden_size,
                    num_rels=num_rels,
                    conversation_dim=hidden_size
                )
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            'user': nn.LayerNorm(hidden_size),
            'post': nn.LayerNorm(hidden_size)
        })
        
        # Reverse message passing (post → user information flow)
        self.reverse_transforms = nn.ModuleDict({
            'publish_rev': nn.Linear(hidden_size, hidden_size),
            'comment_rev': nn.Linear(hidden_size, hidden_size)
        })
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_classes)
        )
        
    def _build_comprehensive_user_context(self, g, node_features, conversation_context=None):
        """
        Build comprehensive user context from all available interactions:
        1. Conversation context (user_comment_user with parent)
        2. Post comment context (user → post comments)  
        3. Published post context (user → own posts)
        """
        num_users = g.num_nodes('user')
        user_context_features = torch.zeros(num_users, self.hidden_size, device=node_features['user'].device)
        
        try:
            # 1. Direct conversation context (highest priority)
            if conversation_context is not None and g.num_edges('user_comment_user') > 0:
                src_nodes, _ = g.edges(etype='user_comment_user')
                
                # Safely map conversation contexts to users
                context_count = min(len(src_nodes), conversation_context.size(0))
                for i in range(context_count):
                    src_idx = src_nodes[i].item()
                    if 0 <= src_idx < num_users:
                        user_context_features[src_idx] += conversation_context[i]
                
                # print(f"Applied conversation context to {context_count} users")
            
            # 2. Post comment context (medium priority)
            if g.num_edges('comment') > 0:
                comment_users, comment_posts = g.edges(etype='comment')
                
                # Map post content to users who commented
                user_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                
                for user_idx, post_idx in zip(comment_users.tolist(), comment_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < node_features['post'].size(0):
                        # Add post content context (weighted by 0.3)
                        post_context = self.user_context_encoder(node_features['post'][post_idx])
                        user_context_features[user_idx] += 0.3 * post_context
                        user_post_counts[user_idx] += 1
                
                # Average for users with multiple comments
                valid_users = user_post_counts > 0
                if valid_users.any():
                    user_context_features[valid_users] /= user_post_counts[valid_users].unsqueeze(1)
                
                # print(f"Applied post comment context to {valid_users.sum().item()} users")
            
            # 3. Published post context (high priority - user's own stance)
            if g.num_edges('publish') > 0:
                publish_users, publish_posts = g.edges(etype='publish')
                
                # Map user's own posts back to them
                user_own_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                
                for user_idx, post_idx in zip(publish_users.tolist(), publish_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < node_features['post'].size(0):
                        # Add own post content context (weighted by 0.5)
                        own_post_context = self.user_context_encoder(node_features['post'][post_idx])
                        user_context_features[user_idx] += 0.5 * own_post_context
                        user_own_post_counts[user_idx] += 1
                
                # Average for users with multiple posts
                valid_publishers = user_own_post_counts > 0
                if valid_publishers.any():
                    user_context_features[valid_publishers] /= user_own_post_counts[valid_publishers].unsqueeze(1)
                
                # print(f"Applied published post context to {valid_publishers.sum().item()} users")
            
        except Exception as e:
            print(f"Warning: Error in context building: {e}")
            # Return zero context if error occurs
            user_context_features = torch.zeros(num_users, self.hidden_size, device=node_features['user'].device)
        
        return user_context_features
        
    def forward(self, g, node_features, edge_features=None):
        """
        Forward pass with conversation context integration
        """
        # Project input features
        h = {
            'user': F.relu(self.user_proj(node_features['user'])),
            'post': F.relu(self.post_proj(node_features['post']))
        }
        
        # Store initial features for zero-degree handling
        initial_features = {ntype: feat.clone() for ntype, feat in h.items()}
        
        # Calculate node degrees
        degrees = {}
        for ntype in g.ntypes:
            total_deg = torch.zeros(g.number_of_nodes(ntype), device=node_features['user'].device)
            for etype in g.etypes:
                canonical = g.to_canonical_etype(etype)
                if canonical[0] == ntype or canonical[2] == ntype:
                    u, v = g.edges(etype=etype)
                    if canonical[0] == ntype:
                        total_deg.index_add_(0, u, torch.ones_like(u, dtype=torch.float))
                    if canonical[2] == ntype:
                        total_deg.index_add_(0, v, torch.ones_like(v, dtype=torch.float))
            degrees[ntype] = total_deg
        
        # Process edge features and extract direct conversation context
        processed_edge_features = {}
        conversation_context = None
        
        if edge_features is not None:
            if 'comment' in edge_features:
                processed_edge_features['comment'] = self.edge_projections['comment'](edge_features['comment'])
                g.edges['comment'].data['feat'] = processed_edge_features['comment']
            
            if 'user_comment_user' in edge_features:
                reply_feat = self.edge_projections['user_comment_user'](edge_features['user_comment_user'])
                processed_edge_features['user_comment_user'] = reply_feat
                
                # Extract conversation context if parent data available
                if (self.use_parent_context and 
                    'parent_feat' in g.edges['user_comment_user'].data):
                    
                    parent_comments = g.edges['user_comment_user'].data['parent_feat']
                    
                    # Use ORIGINAL post features (before projection) for conversation context
                    avg_post_content = node_features['post'].mean(dim=0, keepdim=True).expand(parent_comments.size(0), -1)
                    
                    # Combine parent comment + post content for conversation context
                    combined_context = torch.cat([parent_comments, avg_post_content], dim=1)
                    conversation_context = self.conversation_encoder(combined_context)
                    
                    # Apply conversation weighting to edge features
                    enhanced_reply_feat = ((1 - self.conversation_weight) * reply_feat + 
                                         self.conversation_weight * conversation_context)
                    processed_edge_features['user_comment_user'] = enhanced_reply_feat
                
                g.edges['user_comment_user'].data['feat'] = processed_edge_features['user_comment_user']
        
        # Build comprehensive user context from all interactions
        user_context_features = self._build_comprehensive_user_context(
            g, node_features, conversation_context
        )
        
        # Apply conversation-aware RGCN layers
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            h_new_dict = rgcn_layer(g, h, processed_edge_features, user_context_features)
            
            # Reverse message passing in first layer (post → user information flow)
            if i == 0:
                for rel_type, rev_key in [('publish', 'publish_rev'), ('comment', 'comment_rev')]:
                    if g.num_edges(('user', rel_type, 'post')) == 0:
                        continue
                    
                    src_users, dst_posts = g.edges(etype=rel_type)
                    
                    if len(src_users) > 0:
                        post_trans = self.reverse_transforms[rev_key](h['post'])
                        user_rev_msg = torch.zeros_like(h['user'])
                        user_counts = torch.zeros(h['user'].size(0), device=h['user'].device)
                        
                        # Aggregate post information back to users
                        for u, p in zip(src_users.tolist(), dst_posts.tolist()):
                            if u < user_rev_msg.size(0) and p < post_trans.size(0):
                                user_rev_msg[u] += post_trans[p]
                                user_counts[u] += 1
                        
                        # Average for users with multiple posts
                        mask = user_counts > 0
                        if mask.any():
                            user_rev_msg[mask] = user_rev_msg[mask] / user_counts[mask].unsqueeze(1)
                            
                            if 'user' not in h_new_dict:
                                h_new_dict['user'] = []
                            h_new_dict['user'].append(user_rev_msg)
            
            # Combine messages with residual connections
            for ntype in h:
                if ntype in h_new_dict and len(h_new_dict[ntype]) > 0:
                    h_combined = torch.stack(h_new_dict[ntype]).sum(dim=0)
                    h_res = h[ntype] + h_combined
                    h_norm = self.layer_norms[ntype](h_res)
                    h[ntype] = F.dropout(F.relu(h_norm), p=self.dropout, training=self.training)
                    
                    # Handle zero-degree nodes
                    if ntype in degrees:
                        zero_degree_mask = degrees[ntype] == 0
                        if zero_degree_mask.any():
                            h[ntype][zero_degree_mask] = initial_features[ntype][zero_degree_mask]
                            
                            if (~zero_degree_mask).any():
                                global_context = h[ntype][~zero_degree_mask].mean(dim=0, keepdim=True)
                                h[ntype][zero_degree_mask] = (
                                    0.9 * h[ntype][zero_degree_mask] + 
                                    0.1 * global_context
                                )
        
        # Return user node predictions
        return self.classifier(h['user'])