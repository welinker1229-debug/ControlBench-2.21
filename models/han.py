import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv

class ConversationContextEncoder(nn.Module):
    """
    Encodes conversation context: Original Post → Parent Comment → Current Reply
    For predicting user stance based on conversational thread context.
    """
    def __init__(self, post_dim, comment_dim, hidden_dim):
        super(ConversationContextEncoder, self).__init__()
        
        # Project different feature types to common dimension
        self.post_proj = nn.Linear(post_dim, hidden_dim)
        self.comment_proj = nn.Linear(comment_dim, hidden_dim)
        
        # Conversation sequence encoder (Post → Parent → Reply)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # parent + post context
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, post_content, parent_comment):
        """
        Args:
            post_content: Original post content embeddings [batch, post_dim]
            parent_comment: Parent comment embeddings [batch, comment_dim]
        Returns:
            Conversation context vector [batch, hidden_dim]
        """
        # Project to common dimension
        post_ctx = self.post_proj(post_content)  # [batch, hidden_dim]
        parent_ctx = self.comment_proj(parent_comment)  # [batch, hidden_dim]
        
        # Stack for attention: [post, parent_comment]
        context_seq = torch.stack([post_ctx, parent_ctx], dim=1)  # [batch, 2, hidden_dim]
        
        # Apply self-attention to capture post→parent relationship
        attended_ctx, _ = self.context_attention(
            context_seq, context_seq, context_seq
        )  # [batch, 2, hidden_dim]
        
        # Combine post and parent context
        combined_ctx = torch.cat([
            attended_ctx[:, 0, :],  # post context
            attended_ctx[:, 1, :]   # parent context
        ], dim=1)  # [batch, hidden_dim * 2]
        
        # Fuse contexts
        conversation_context = self.context_fusion(combined_ctx)  # [batch, hidden_dim]
        
        return conversation_context

class ConversationAwareGATLayer(nn.Module):
    """
    GAT layer enhanced with conversation context for user-comment-user edges
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0):
        super(ConversationAwareGATLayer, self).__init__()
        self.gat = GATConv(
            in_dim, out_dim, num_heads, 
            dropout, dropout, 
            activation=F.elu,
            allow_zero_in_degree=True
        )
        
        # Project back to consistent dimension
        self.proj = nn.Linear(out_dim * num_heads, in_dim)
        
        # Conversation context integration
        self.context_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),  # node_feat + conversation_context
            nn.Sigmoid()
        )
        
    def forward(self, g, h, conversation_context=None):
        if g.num_edges() == 0:
            return h
            
        # Apply GAT
        h_gat = self.gat(g, h)
        
        # Handle multi-head output
        if len(h_gat.shape) > 2:
            h_gat = h_gat.reshape(h_gat.shape[0], -1)
            
        h_projected = self.proj(h_gat)
        
        # Integrate conversation context if available
        if conversation_context is not None:
            # Ensure conversation context matches node features dimension
            if conversation_context.size(0) == h_projected.size(0):
                # Gating mechanism to blend node features with conversation context
                gate_input = torch.cat([h_projected, conversation_context], dim=1)
                gate = self.context_gate(gate_input)
                
                # Apply gating: gate * node_features + (1-gate) * conversation_context
                h_projected = gate * h_projected + (1 - gate) * conversation_context
        
        return h_projected

class SemanticAttention(nn.Module):
    """
    Semantic attention for combining meta-path embeddings
    """
    def __init__(self, in_size, hidden_size=256):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, degrees=None):
        # z shape: [num_nodes, num_meta_paths, embedding_dim]
        w = self.project(z)  # [num_nodes, num_meta_paths, 1]
        
        # Degree-aware attention boost for isolated nodes
        if degrees is not None:
            norm_degrees = degrees / (degrees.max() + 1e-6)
            degree_boost = torch.exp(-norm_degrees).unsqueeze(1).unsqueeze(2)
            w = w + 0.1 * degree_boost
            
        beta = torch.softmax(w, dim=1)  # [num_nodes, num_meta_paths, 1]
        return (beta * z).sum(1)  # [num_nodes, embedding_dim]

class HANNodeClassifier(nn.Module):
    """
    Conversation-aware HAN: Predicts user stance based on comprehensive user context
    including conversation threads, post interactions, and published content.
    
    Context Hierarchy:
    1. Direct conversations: User A → User B (user_comment_user with parent context)
    2. Post comments: User A → Post X (comment edges)  
    3. Published posts: User A → Post Y (publish edges)
    """
    def __init__(self, in_feats=768, edge_feats=768, hidden_size=256, out_classes=9, 
                 dropout=0.5, n_layers=1, num_heads=8, post_feats=1536,
                 use_parent_context=True, conversation_weight=0.3):
        super(HANNodeClassifier, self).__init__()
        
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.post_feats = post_feats
        self.hidden_size = hidden_size
        self.out_classes = out_classes
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_parent_context = use_parent_context
        self.conversation_weight = conversation_weight
        
        # Node feature projections
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        
        # Edge feature projections  
        self.comment_edge_proj = nn.Linear(edge_feats, hidden_size)
        self.user_comment_user_edge_proj = nn.Linear(edge_feats, hidden_size)
        
        # Conversation context encoder (Post → Parent Comment → Reply)
        if self.use_parent_context:
            self.conversation_encoder = ConversationContextEncoder(
                post_dim=post_feats,
                comment_dim=edge_feats, 
                hidden_dim=hidden_size
            )
        
        # Multi-context encoder for comprehensive user modeling
        self.user_context_encoder = nn.Sequential(
            nn.Linear(post_feats, hidden_size),  # For user's own posts and commented posts
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Meta-paths for conversation-aware processing
        self.meta_paths = [
            [('user', 'user_comment_user', 'user')],  # User replies (conversation)
            [('user', 'comment', 'post')],            # User comments on posts
            [('user', 'publish', 'post')]             # User creates posts
        ]
        
        # Conversation-aware meta-path layers
        self.meta_path_layers = nn.ModuleList()
        for _ in range(len(self.meta_paths)):
            self.meta_path_layers.append(
                ConversationAwareGATLayer(hidden_size, hidden_size // num_heads, num_heads, dropout)
            )
        
        # Semantic attention for combining meta-path embeddings
        self.semantic_attention = SemanticAttention(hidden_size)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
                post_feats = node_features['post']
                
                # Map post content to users who commented
                user_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                
                for user_idx, post_idx in zip(comment_users.tolist(), comment_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < post_feats.size(0):
                        # Add post content context (weighted by 0.3)
                        post_context = self.user_context_encoder(post_feats[post_idx])
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
                post_feats = node_features['post']
                
                # Map user's own posts back to them
                user_own_post_counts = torch.zeros(num_users, device=node_features['user'].device)
                
                for user_idx, post_idx in zip(publish_users.tolist(), publish_posts.tolist()):
                    if 0 <= user_idx < num_users and 0 <= post_idx < post_feats.size(0):
                        # Add own post content context (weighted by 0.5)
                        own_post_context = self.user_context_encoder(post_feats[post_idx])
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
        
    def forward(self, g, node_features, edge_features=None, labels=None):
        # Project node features
        h_dict = {
            'user': F.relu(self.user_proj(node_features['user'])),
            'post': F.relu(self.post_proj(node_features['post']))
        }
        
        user_feats = h_dict['user']
        post_feats = h_dict['post']
        
        # Process edge features and extract conversation context
        conversation_context = None
        
        if edge_features is not None:
            # Process comment edges (user → post)
            if 'comment' in edge_features:
                g.edges['comment'].data['feat'] = self.comment_edge_proj(edge_features['comment'])
                
            # Process user-comment-user edges (user → user replies)
            if 'user_comment_user' in edge_features:
                reply_feat = self.user_comment_user_edge_proj(edge_features['user_comment_user'])
                g.edges['user_comment_user'].data['feat'] = reply_feat
                
                # Extract conversation context if parent data available
                if (self.use_parent_context and 
                    'parent_feat' in g.edges['user_comment_user'].data):
                    
                    # Get parent comment embeddings
                    parent_comments = g.edges['user_comment_user'].data['parent_feat']
                    
                    # Use ORIGINAL post features (before projection) for conversation context
                    # Get average of original post features
                    avg_post_content = node_features['post'].mean(dim=0, keepdim=True).expand(parent_comments.size(0), -1)
                    
                    # Encode conversation context: Post → Parent Comment → Reply
                    conversation_context = self.conversation_encoder(
                        post_content=avg_post_content,
                        parent_comment=parent_comments
                    )
                    
                    # Apply conversation context to edge features
                    enhanced_reply_feat = ((1 - self.conversation_weight) * reply_feat + 
                                         self.conversation_weight * conversation_context)
                    g.edges['user_comment_user'].data['feat'] = enhanced_reply_feat
        
        # Build comprehensive user context from all interactions
        user_context_features = self._build_comprehensive_user_context(
            g, node_features, conversation_context
        )
        
        # Calculate node degrees for attention
        user_degrees = torch.zeros(user_feats.shape[0], device=user_feats.device)
        for etype in g.etypes:
            canonical = g.to_canonical_etype(etype)
            if canonical[0] == 'user':
                u, v = g.edges(etype=etype)
                user_degrees.index_add_(0, u, torch.ones_like(u, dtype=torch.float))
            if canonical[2] == 'user':
                u, v = g.edges(etype=etype)
                user_degrees.index_add_(0, v, torch.ones_like(v, dtype=torch.float))
        
        # Process meta-paths with comprehensive context awareness
        user_meta_path_embeddings = []
        
        for i, meta_path in enumerate(self.meta_paths):
            try:
                if len(meta_path) == 1:
                    etype = meta_path[0]
                    src_type, edge_type, dst_type = etype
                    
                    if g.num_edges(etype) == 0:
                        # Use base user features enhanced with context
                        enhanced_user_feats = user_feats + 0.2 * user_context_features
                        user_meta_path_embeddings.append(enhanced_user_feats)
                        continue
                    
                    if src_type == 'user' and dst_type == 'post':
                        # User-post interactions: Enhanced with post context
                        enhanced_user_feats = user_feats + 0.3 * user_context_features
                        user_meta_path_embeddings.append(enhanced_user_feats)
                        
                    elif src_type == 'user' and dst_type == 'user':
                        # User-user conversations: The key conversation path
                        sg = dgl.edge_type_subgraph(g, [etype])
                        sg = dgl.add_self_loop(sg)
                        
                        # Apply conversation-aware GAT with user context
                        user_mp_embeds = self.meta_path_layers[i](sg, user_feats, user_context_features)
                        user_meta_path_embeddings.append(user_mp_embeds)
                    else:
                        # Fallback with context enhancement
                        enhanced_user_feats = user_feats + 0.2 * user_context_features
                        user_meta_path_embeddings.append(enhanced_user_feats)
                
            except Exception as e:
                print(f"Warning: Error processing meta-path {i}: {e}")
                # Fallback: use enhanced user features
                enhanced_user_feats = user_feats + 0.1 * user_context_features
                user_meta_path_embeddings.append(enhanced_user_feats)
                continue
        
        # Apply semantic attention to combine meta-path embeddings
        if len(user_meta_path_embeddings) > 0:
            # Ensure all embeddings have the same shape
            min_size = min(emb.size(0) for emb in user_meta_path_embeddings)
            user_meta_path_embeddings = [emb[:min_size] for emb in user_meta_path_embeddings]
            
            stacked = torch.stack(user_meta_path_embeddings, dim=1)
            final_h = self.semantic_attention(stacked, user_degrees[:min_size])
        else:
            # Fallback to context-enhanced user features
            final_h = user_feats + 0.2 * user_context_features
        
        # Classify user stances
        return self.classifier(final_h)