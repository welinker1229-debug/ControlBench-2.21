import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConversationAwareHGTLayer(nn.Module):
    """
    HGT layer with conversation context integration for user-comment-user edges
    """
    def __init__(self, in_dim, out_dim, num_heads, edge_dim=None, dropout=0.5, use_norm=True):
        super(ConversationAwareHGTLayer, self).__init__()
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim if edge_dim is not None else in_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        
        # Initialize parameters immediately to avoid device issues
        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        # Edge projections
        self.edge_proj = nn.ModuleDict()
        
        # Relation-specific parameters - use ModuleDict to ensure proper device handling
        self.relation_params = nn.ModuleDict()
        
        # Conversation-aware components
        self.conversation_k_proj = nn.Linear(in_dim, out_dim)
        self.conversation_v_proj = nn.Linear(in_dim, out_dim)
        self.conversation_attention_weight = nn.Parameter(torch.FloatTensor([0.3]))
        
        # Feature similarity boost
        self.feature_sim_boost = nn.Parameter(torch.FloatTensor([0.2]))
        
        self.drop = nn.Dropout(dropout)
        
        # Flag to track initialization
        self.initialized = False
    
    def _init_params(self, g):
        """Initialize parameters based on graph structure"""
        if self.initialized:
            return
            
        node_types = g.ntypes
        canonical_etypes = g.canonical_etypes
        
        # Get device from the first parameter
        device = next(self.parameters()).device
        
        # Standard HGT parameters
        for ntype in node_types:
            self.k_linears[ntype] = nn.Linear(self.in_dim, self.out_dim).to(device)
            self.q_linears[ntype] = nn.Linear(self.in_dim, self.out_dim).to(device)
            self.v_linears[ntype] = nn.Linear(self.in_dim, self.out_dim).to(device)
            self.a_linears[ntype] = nn.Linear(self.out_dim, self.out_dim).to(device)
            
            if self.use_norm:
                self.norms[ntype] = nn.LayerNorm(self.out_dim).to(device)
        
        # Relation-specific parameters using ModuleDict for proper device handling
        for src, etype, dst in canonical_etypes:
            # Create a module to hold relation-specific parameters
            rel_module = nn.Module()
            rel_module.k = nn.Parameter(torch.Tensor(self.num_heads, self.d_k, self.d_k))
            rel_module.q = nn.Parameter(torch.Tensor(self.num_heads, self.d_k, self.d_k))
            rel_module.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_k, self.d_k))
            
            # Initialize parameters
            nn.init.xavier_uniform_(rel_module.k)
            nn.init.xavier_uniform_(rel_module.q)
            nn.init.xavier_uniform_(rel_module.v)
            
            self.relation_params[etype] = rel_module.to(device)
        
        # Edge projections
        for _, etype, _ in canonical_etypes:
            if etype in ['comment', 'user_comment_user']:
                self.edge_proj[etype] = nn.Linear(self.edge_dim, self.out_dim).to(device)
        
        self.initialized = True
    
    def forward(self, g, h_dict, edge_h=None, user_context_features=None):
        """
        Forward pass with comprehensive user context integration
        """
        if not self.initialized:
            self._init_params(g)
        
        # Store initial features for zero-degree handling
        initial_h = {ntype: feat.clone() for ntype, feat in h_dict.items()}
        
        # Calculate node degrees
        degrees = {}
        for ntype in g.ntypes:
            total_deg = torch.zeros(g.number_of_nodes(ntype), device=h_dict['user'].device)
            for etype in g.etypes:
                canonical = g.to_canonical_etype(etype)
                if canonical[0] == ntype or canonical[2] == ntype:
                    u, v = g.edges(etype=etype)
                    if canonical[0] == ntype:
                        total_deg.index_add_(0, u, torch.ones_like(u, dtype=torch.float))
                    if canonical[2] == ntype:
                        total_deg.index_add_(0, v, torch.ones_like(v, dtype=torch.float))
            degrees[ntype] = total_deg
        
        # Result dictionary
        result = {ntype: torch.zeros_like(feat) for ntype, feat in h_dict.items()}
        
        # Process each canonical edge type
        for srctype, etype, dsttype in g.canonical_etypes:
            if g.number_of_edges((srctype, etype, dsttype)) == 0:
                continue
                
            src_feat = h_dict[srctype]
            dst_feat = h_dict[dsttype]
            sub_g = g[srctype, etype, dsttype]
            src_idx, dst_idx = sub_g.edges()
            
            # Process edge features
            edge_feat = None
            if edge_h is not None and etype in edge_h and edge_h[etype] is not None:
                if etype in self.edge_proj:
                    edge_feat = self.edge_proj[etype](edge_h[etype])
            
            # Standard K, Q, V projections
            k = self.k_linears[srctype](src_feat)
            q = self.q_linears[dsttype](dst_feat)
            v = self.v_linears[srctype](src_feat)
            
            # Special handling for conversation edges
            if (etype == 'user_comment_user' and user_context_features is not None):
                
                # Create conversation-aware K, V projections
                # Use user context features to enhance K and V
                conv_weight = torch.sigmoid(self.conversation_attention_weight)
                
                # Create new tensors instead of modifying in-place
                k_enhanced = k.clone()
                v_enhanced = v.clone()
                
                # Map user context to source nodes
                for i, src_node_idx in enumerate(src_idx):
                    src_node_idx = src_node_idx.item()
                    if src_node_idx < user_context_features.size(0):
                        user_ctx = user_context_features[src_node_idx]
                        
                        # Enhance K and V with conversation context
                        conv_k = self.conversation_k_proj(user_ctx)
                        conv_v = self.conversation_v_proj(user_ctx)
                        
                        # Blend standard and conversation-aware projections (no in-place)
                        k_enhanced[src_node_idx] = (1 - conv_weight) * k[src_node_idx] + conv_weight * conv_k
                        v_enhanced[src_node_idx] = (1 - conv_weight) * v[src_node_idx] + conv_weight * conv_v
                
                # Replace with enhanced versions
                k = k_enhanced
                v = v_enhanced
            
            # Reshape for multi-head attention
            k = k.view(k.size(0), self.num_heads, self.d_k)
            q = q.view(q.size(0), self.num_heads, self.d_k)
            v = v.view(v.size(0), self.num_heads, self.d_k)
            
            k_src = k[src_idx]
            q_dst = q[dst_idx]
            v_src = v[src_idx]
            
            # Apply relation-specific transformations
            rel_k = self.relation_params[etype].k
            rel_q = self.relation_params[etype].q
            rel_v = self.relation_params[etype].v
            
            k_transformed = torch.bmm(k_src.transpose(0, 1), rel_k).transpose(0, 1)
            q_transformed = torch.bmm(q_dst.transpose(0, 1), rel_q).transpose(0, 1)
            
            # Compute attention scores
            attention = torch.sum(q_transformed * k_transformed, dim=2) / self.sqrt_dk
            
            # Add edge feature contribution
            if edge_feat is not None:
                ef = edge_feat.view(-1, self.num_heads, self.d_k)
                edge_attn = torch.sum(ef * q_transformed, dim=2) / self.sqrt_dk
                attention = attention + edge_attn
            
            # Feature similarity boost for sparse regions
            sparsity_level = (degrees[dsttype] == 0).float().mean().item()
            if edge_feat is None and sparsity_level > 0.3:
                src_norm = F.normalize(src_feat[src_idx], dim=1)
                dst_norm = F.normalize(dst_feat[dst_idx], dim=1)
                feat_sim = torch.sum(src_norm * dst_norm, dim=1, keepdim=True)
                feat_sim = feat_sim.view(-1, 1).expand(-1, self.num_heads)
                attention = attention + self.feature_sim_boost * feat_sim
            
            # Apply softmax grouped by destination node
            unique_dst, dst_inverse = torch.unique(dst_idx, return_inverse=True)
            attention_mask = torch.zeros_like(attention)
            
            for i, dst_node in enumerate(unique_dst):
                idx = (dst_inverse == i)
                if idx.sum() > 0:
                    attention_mask[idx] = F.softmax(attention[idx], dim=0)
            
            attention_mask = self.drop(attention_mask)
            
            # Transform values
            v_transformed = torch.bmm(v_src.transpose(0, 1), rel_v).transpose(0, 1)
            
            # Compute weighted values
            message = v_transformed * attention_mask.unsqueeze(-1)
            
            # Aggregate messages by destination node
            agg_message = torch.zeros((dst_feat.shape[0], self.num_heads, self.d_k), device=dst_feat.device)
            
            for i, dst_node in enumerate(unique_dst):
                idx = (dst_inverse == i)
                if idx.sum() > 0:
                    agg_message[dst_node] = message[idx].sum(dim=0)
            
            # Reshape and apply output transformation
            agg_message = agg_message.reshape(-1, self.out_dim)
            out = self.a_linears[dsttype](agg_message)
            
            if self.use_norm:
                out = self.norms[dsttype](out)
            
            result[dsttype] = result[dsttype] + out
        
        # Handle zero-degree nodes (avoid in-place operations)
        for ntype in result:
            if ntype in degrees:
                zero_degree_mask = degrees[ntype] == 0
                if zero_degree_mask.any():
                    # Create new tensor instead of modifying in-place
                    enhanced_result = result[ntype].clone()
                    enhanced_result[zero_degree_mask] = initial_h[ntype][zero_degree_mask]
                    
                    if (~zero_degree_mask).any():
                        global_context = enhanced_result[~zero_degree_mask].mean(dim=0, keepdim=True)
                        enhanced_result[zero_degree_mask] = (
                            0.9 * enhanced_result[zero_degree_mask] + 
                            0.1 * global_context
                        )
                    
                    result[ntype] = enhanced_result
        
        return result

class HGTNodeClassifier(nn.Module):
    """
    Conversation-aware HGT: Integrates Post→Parent Comment→Reply context in attention mechanisms
    """
    def __init__(self, in_feats, edge_feats, hidden_size, out_classes, n_layers=2, 
                 num_heads=4, dropout=0.2, num_ntypes=2, num_etypes=3, post_feats=None, 
                 use_edge_feats=True, use_norm=True, use_parent_context=True,
                 conversation_weight=0.3):
        super(HGTNodeClassifier, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.in_feats = in_feats
        self.post_feats = post_feats if post_feats is not None else in_feats
        self.edge_feats = edge_feats
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.use_edge_feats = use_edge_feats
        self.use_parent_context = use_parent_context
        self.conversation_weight = conversation_weight
        
        # Node projections
        self.node_projection = nn.ModuleDict({
            'user': nn.Sequential(
                nn.Linear(in_feats, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'post': nn.Sequential(
                nn.Linear(self.post_feats, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        
        # Edge feature projections
        self.edge_projection = nn.ModuleDict({
            'comment': nn.Sequential(
                nn.Linear(edge_feats, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ),
            'user_comment_user': nn.Sequential(
                nn.Linear(edge_feats, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            )
        })
        
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
        
        # Structural encoding for isolated nodes
        self.structural_encoder = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Conversation-aware HGT layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ConversationAwareHGTLayer(
                in_dim=hidden_size,
                out_dim=hidden_size,
                edge_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                use_norm=use_norm
            ))
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Classifier
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
                
                # Safely map conversation contexts to users (avoid in-place operations)
                context_count = min(len(src_nodes), conversation_context.size(0))
                for i in range(context_count):
                    src_idx = src_nodes[i].item()
                    if 0 <= src_idx < num_users:
                        # Create new tensor instead of in-place addition
                        user_context_features = user_context_features.clone()
                        user_context_features[src_idx] = user_context_features[src_idx] + conversation_context[i]
                
                print(f"Applied conversation context to {context_count} users")
            
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
                
                print(f"Applied post comment context to {valid_users.sum().item()} users")
            
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
                
                print(f"Applied published post context to {valid_publishers.sum().item()} users")
            
        except Exception as e:
            print(f"Warning: Error in context building: {e}")
            # Return zero context if error occurs
            user_context_features = torch.zeros(num_users, self.hidden_size, device=node_features['user'].device)
        
        return user_context_features
        
    def forward(self, g, node_feats, edge_feats=None):
        """
        Forward pass with conversation context integration
        """
        # Project node features
        h_dict = {}
        for ntype, feat in node_feats.items():
            if ntype in self.node_projection:
                h_dict[ntype] = self.node_projection[ntype](feat)
        
        # Calculate node degrees
        degrees = {}
        for ntype in g.ntypes:
            total_deg = torch.zeros(g.number_of_nodes(ntype), device=node_feats['user'].device)
            for etype in g.etypes:
                canonical = g.to_canonical_etype(etype)
                if canonical[0] == ntype or canonical[2] == ntype:
                    u, v = g.edges(etype=etype)
                    if canonical[0] == ntype:
                        total_deg.index_add_(0, u, torch.ones_like(u, dtype=torch.float))
                    if canonical[2] == ntype:
                        total_deg.index_add_(0, v, torch.ones_like(v, dtype=torch.float))
            degrees[ntype] = total_deg
        
        # Structural features for isolated nodes
        if 'user' in degrees:
            structural_features = torch.zeros((g.number_of_nodes('user'), 3), device=node_feats['user'].device)
            max_degree = max(1.0, degrees['user'].max().item())
            structural_features[:, 0] = degrees['user'] / max_degree
            
            # Comment and user-comment-user participation
            if 'comment' in g.etypes:
                src, _ = g.edges(etype='comment')
                comment_counts = torch.bincount(src, minlength=g.number_of_nodes('user')).float()
                if comment_counts.max() > 0:
                    structural_features[:, 1] = comment_counts / comment_counts.max()
            
            if 'user_comment_user' in g.etypes:
                src, _ = g.edges(etype='user_comment_user')
                ucu_counts = torch.bincount(src, minlength=g.number_of_nodes('user')).float()
                if ucu_counts.max() > 0:
                    structural_features[:, 2] = ucu_counts / ucu_counts.max()
            
            # Add structural features to isolated nodes (avoid in-place operations)
            zero_degree_mask = degrees['user'] == 0
            if zero_degree_mask.any():
                struct_emb = self.structural_encoder(structural_features)
                
                # Pad to correct dimension
                if struct_emb.size(1) < h_dict['user'].size(1):
                    padding = torch.zeros(struct_emb.size(0), h_dict['user'].size(1) - struct_emb.size(1), 
                                         device=struct_emb.device)
                    struct_emb = torch.cat([struct_emb, padding], dim=1)
                
                # Create new tensor instead of in-place modification
                enhanced_user_features = h_dict['user'].clone()
                enhanced_user_features[zero_degree_mask] = h_dict['user'][zero_degree_mask] + struct_emb[zero_degree_mask]
                h_dict['user'] = enhanced_user_features
        
        # Process edge features and extract conversation context
        edge_h = None
        conversation_context = None
        
        if self.use_edge_feats and edge_feats is not None:
            edge_h = {}
            for etype, feat in edge_feats.items():
                if etype in self.edge_projection:
                    edge_h[etype] = self.edge_projection[etype](feat)
            
            # Extract conversation context for user-comment-user edges
            if (self.use_parent_context and 'user_comment_user' in edge_feats and
                'parent_feat' in g.edges['user_comment_user'].data):
                
                parent_comments = g.edges['user_comment_user'].data['parent_feat']
                
                # Use ORIGINAL post features (before projection) for conversation context
                avg_post_content = node_feats['post'].mean(dim=0, keepdim=True).expand(parent_comments.size(0), -1)
                
                # Combine parent comment + post content for conversation context
                combined_context = torch.cat([parent_comments, avg_post_content], dim=1)
                conversation_context = self.conversation_encoder(combined_context)
                
                # Apply conversation weighting to edge features
                reply_feat = edge_h['user_comment_user']
                enhanced_reply_feat = ((1 - self.conversation_weight) * reply_feat + 
                                     self.conversation_weight * conversation_context)
                edge_h['user_comment_user'] = enhanced_reply_feat
        
        # Build comprehensive user context from all interactions
        user_context_features = self._build_comprehensive_user_context(
            g, node_feats, conversation_context
        )
        
        # Process through conversation-aware HGT layers
        for layer in self.layers:
            h_dict = layer(g, h_dict, edge_h, user_context_features)
            
            # Apply activation between layers
            for ntype in h_dict:
                h_dict[ntype] = F.relu(h_dict[ntype])
        
        # Final normalization
        h_dict['user'] = self.final_norm(h_dict['user'])
        
        # Classify user stances
        return self.classifier(h_dict['user'])