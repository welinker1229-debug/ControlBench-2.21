import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class ConversationAwareSAGEConv(nn.Module):
    """
    SAGE convolution enhanced with conversation context for user-comment-user edges
    """
    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        super(ConversationAwareSAGEConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        
        # Standard SAGE components
        self.sage_conv = dglnn.SAGEConv(in_feats, out_feats, aggregator_type)
        
        # Conversation-aware message function
        self.conversation_message_func = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),  # node_feat + conversation_context
            nn.LayerNorm(out_feats),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Gating mechanism for blending standard and conversation-aware messages
        self.message_gate = nn.Sequential(
            nn.Linear(out_feats * 2, out_feats),
            nn.Sigmoid()
        )
    
    def forward(self, g, h, user_context_features=None):
        """
        Forward pass with user context-aware aggregation
        """
        # Standard SAGE convolution
        standard_output = self.sage_conv(g, h)
        
        # If no user context, return standard output
        if user_context_features is None or g.num_edges() == 0:
            return standard_output
        
        # User context-aware message passing
        src_nodes, dst_nodes = g.edges()
        
        if len(src_nodes) > 0:
            # Create context-enhanced messages using user context
            conv_messages = []
            
            for i, src_idx in enumerate(src_nodes):
                src_feat = h[src_idx]
                
                # Get user context for this source node
                if src_idx < user_context_features.size(0):
                    user_ctx = user_context_features[src_idx]
                    # Combine node features with user context
                    enhanced_feat = torch.cat([src_feat, user_ctx], dim=0)
                    conv_message = self.conversation_message_func(enhanced_feat)
                else:
                    # Fallback to standard message for nodes without context
                    conv_message = torch.zeros(self.out_feats, device=h.device)
                
                conv_messages.append(conv_message)
            
            if conv_messages:
                conv_messages = torch.stack(conv_messages)
                
                # Aggregate conversation messages by destination node
                conv_aggregated = torch.zeros_like(standard_output)
                dst_counts = torch.zeros(h.size(0), device=h.device)
                
                for i, dst_idx in enumerate(dst_nodes):
                    if dst_idx < conv_aggregated.size(0):
                        conv_aggregated[dst_idx] = conv_aggregated[dst_idx] + conv_messages[i]
                        dst_counts[dst_idx] += 1
                
                # Normalize by destination node degree
                mask = dst_counts > 0
                if mask.any():
                    conv_aggregated[mask] = conv_aggregated[mask] / dst_counts[mask].unsqueeze(1)
                
                # Blend standard and conversation-aware outputs
                gate_input = torch.cat([standard_output, conv_aggregated], dim=1)
                gate = self.message_gate(gate_input)
                
                output = gate * standard_output + (1 - gate) * conv_aggregated
                return output
        
        return standard_output

class HinSAGENodeClassifier(nn.Module):
    """
    Conversation-aware HinSAGE: Incorporates Post→Parent Comment→Reply context in GraphSAGE aggregation
    """
    def __init__(self, in_feats, edge_feats, hidden_size, out_classes, dropout=0.5,
                 n_layers=2, aggregator_type='mean', post_feats=None,
                 use_parent_context=True, conversation_weight=0.3):
        super(HinSAGENodeClassifier, self).__init__()
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.hidden_size = hidden_size
        self.out_classes = out_classes
        self.dropout = dropout
        self.n_layers = n_layers
        self.aggregator_type = aggregator_type
        self.use_parent_context = use_parent_context
        self.conversation_weight = conversation_weight
        
        self.post_feats = post_feats if post_feats is not None else in_feats
        
        # Input projections
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(self.post_feats, hidden_size)
        
        # Edge feature projections
        self.comment_edge_proj = nn.Linear(edge_feats, hidden_size)
        self.user_comment_user_edge_proj = nn.Linear(edge_feats, hidden_size)
        
        # Conversation context encoder (Post → Parent Comment → Reply)
        if self.use_parent_context:
            self.conversation_encoder = nn.Sequential(
                nn.Linear(edge_feats + self.post_feats, hidden_size),  # parent_comment + post_content
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # User context encoder for comprehensive context
        self.user_context_encoder = nn.Sequential(
            nn.Linear(self.post_feats, hidden_size),  # For user's own posts and commented posts
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Standard HinSAGE layers (using DGL's built-in HeteroGraphConv)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            conv_dict = {
                'publish': dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type),
                'comment': dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type),
                'user_comment_user': dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type)
            }
            self.layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            'user': nn.LayerNorm(hidden_size),
            'post': nn.LayerNorm(hidden_size)
        })
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
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
                
                # Safely map conversation contexts to users (avoid in-place operations)
                context_count = min(len(src_nodes), conversation_context.size(0))
                context_updates = torch.zeros_like(user_context_features)
                
                for i in range(context_count):
                    src_idx = src_nodes[i].item()
                    if 0 <= src_idx < num_users:
                        context_updates[src_idx] = context_updates[src_idx] + conversation_context[i]
                
                user_context_features = user_context_features + context_updates
                # print(f"Applied conversation context to {context_count} users")
            
            # 2. Post comment context (medium priority)
            if g.num_edges('comment') > 0:
                comment_users, comment_posts = g.edges(etype='comment')
                
                # Map post content to users who commented
                user_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                comment_context_updates = torch.zeros_like(user_context_features)
                
                for user_idx, post_idx in zip(comment_users.tolist(), comment_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < node_features['post'].size(0):
                        # Add post content context (weighted by 0.3)
                        post_context = self.user_context_encoder(node_features['post'][post_idx])
                        comment_context_updates[user_idx] = comment_context_updates[user_idx] + 0.3 * post_context
                        user_post_counts[user_idx] += 1
                
                # Average for users with multiple comments
                valid_users = user_post_counts > 0
                if valid_users.any():
                    comment_context_updates[valid_users] = comment_context_updates[valid_users] / user_post_counts[valid_users].unsqueeze(1)
                
                # Apply updates (avoid in-place)
                user_context_features = user_context_features + comment_context_updates
                # print(f"Applied post comment context to {valid_users.sum().item()} users")
            
            # 3. Published post context (high priority - user's own stance)
            if g.num_edges('publish') > 0:
                publish_users, publish_posts = g.edges(etype='publish')
                
                # Map user's own posts back to them
                user_own_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                publish_context_updates = torch.zeros_like(user_context_features)  
                
                for user_idx, post_idx in zip(publish_users.tolist(), publish_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < node_features['post'].size(0):
                        # Add own post content context (weighted by 0.5)
                        own_post_context = self.user_context_encoder(node_features['post'][post_idx])
                        publish_context_updates[user_idx] = publish_context_updates[user_idx] + 0.5 * own_post_context
                        user_own_post_counts[user_idx] += 1
                
                # Average for users with multiple posts
                valid_publishers = user_own_post_counts > 0
                if valid_publishers.any():
                    publish_context_updates[valid_publishers] = publish_context_updates[valid_publishers] / user_own_post_counts[valid_publishers].unsqueeze(1)
                
                # Apply updates (avoid in-place)
                user_context_features = user_context_features + publish_context_updates
                # print(f"Applied published post context to {valid_publishers.sum().item()} users")
            
        except Exception as e:
            print(f"Warning: Error in context building: {e}")
            # Return zero context if error occurs
            user_context_features = torch.zeros(num_users, self.hidden_size, device=node_features['user'].device)
        
        return user_context_features
    
    def forward(self, g, node_features, edge_features):
        """
        Forward pass with conversation context integration
        """
        # Project node features
        h = {
            'user': F.leaky_relu(self.user_proj(node_features['user'])),
            'post': F.leaky_relu(self.post_proj(node_features['post']))
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
        
        # Process edge features and extract conversation context
        conversation_context = None
        
        if 'user_comment_user' in edge_features:
            # Extract conversation context if parent data available
            if (self.use_parent_context and 
                'parent_feat' in g.edges['user_comment_user'].data):
                
                parent_comments = g.edges['user_comment_user'].data['parent_feat']
                
                # Use ORIGINAL post features (before projection) for conversation context
                avg_post_content = node_features['post'].mean(dim=0, keepdim=True).expand(parent_comments.size(0), -1)
                
                # Combine parent comment + post content for conversation context
                combined_context = torch.cat([parent_comments, avg_post_content], dim=1)
                conversation_context = self.conversation_encoder(combined_context)
        
        # Build comprehensive user context from all interactions
        user_context_features = self._build_comprehensive_user_context(
            g, node_features, conversation_context
        )
        
        # Apply dropout
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        
        # Process through HinSAGE layers with conversation context
        for i, layer in enumerate(self.layers):
            # For the first layer, enhance user features with comprehensive context
            if i == 0 and user_context_features is not None:
                # Enhance user features with conversation context
                conv_weight = self.conversation_weight
                enhanced_user_features = ((1 - conv_weight) * h['user'] + 
                                        conv_weight * user_context_features)
                h['user'] = enhanced_user_features
            
            # Apply HeteroGraphConv layer
            h_new = layer(g, h)
            
            # Handle zero-degree nodes (avoid in-place operations)
            for ntype in h_new:
                if ntype in degrees:
                    zero_degree_mask = degrees[ntype] == 0
                    if zero_degree_mask.any():
                        # Create new tensor instead of modifying in-place
                        enhanced_features = h_new[ntype].clone()
                        enhanced_features[zero_degree_mask] = initial_features[ntype][zero_degree_mask]
                        
                        # Add global context for isolated nodes
                        if (~zero_degree_mask).any():
                            global_context = enhanced_features[~zero_degree_mask].mean(dim=0, keepdim=True)
                            enhanced_features[zero_degree_mask] = (
                                0.9 * enhanced_features[zero_degree_mask] + 
                                0.1 * global_context
                            )
                        
                        h_new[ntype] = enhanced_features
            
            # Apply layer normalization
            h = {ntype: self.layer_norms[ntype](feat) if ntype in self.layer_norms else feat 
                 for ntype, feat in h_new.items()}
            
            # Apply non-linearity and dropout except for last layer
            if i < len(self.layers) - 1:
                h = {k: F.leaky_relu(v) for k, v in h.items()}
                h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        
        # Classify user nodes
        return self.classifier(h['user'])