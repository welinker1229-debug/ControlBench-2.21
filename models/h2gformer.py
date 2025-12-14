import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import HeteroGraphConv
from dgl.utils import expand_as_pair

class SparseMultiHeadAttentionConv(nn.Module):
    """
    Sparse Graph Transformer Convolution.
    Implements the core logic of H2G-Former's SparseNodeTransformer.
    """
    def __init__(self, in_dim, out_dim, num_heads, edge_dim=None, dropout=0.0):
        super(SparseMultiHeadAttentionConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        
        # Edge Feature Projection
        if edge_dim is not None and edge_dim > 0:
            self.W_e = nn.Linear(edge_dim, num_heads)
            self.use_edge = True
        else:
            self.use_edge = False
            
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, g, h, e_feat=None):
        # h is (h_src, h_dst) or single tensor
        h_src, h_dst = expand_as_pair(h, g)
        
        # Step 1. Project Q, K, V
        # Shape: [N, num_heads, head_dim]
        q = self.W_q(h_dst).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(h_src).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(h_src).view(-1, self.num_heads, self.head_dim)
        
        # Step 2. Calculate Attention Scores
        g.srcdata.update({'k': k, 'v': v})
        g.dstdata.update({'q': q})        
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))
        
        # Step 3. Add Edge Bias if available
        if self.use_edge and e_feat is not None:
            e_bias = self.W_e(e_feat).view(-1, self.num_heads, 1)
            g.edata['score'] = (g.edata['score'] * self.scale) + e_bias
        else:
            g.edata['score'] = g.edata['score'] * self.scale

        # Step 4. Softmax and Dropout
        g.edata['a'] = self.attn_drop(dgl.nn.functional.edge_softmax(g, g.edata['score']))

        # Step 5. Aggregation
        g.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'out'))
        
        out = g.dstdata['out'].view(-1, self.out_dim)
        return self.out_proj(out)

class H2GFormerLayer(nn.Module):
    """
    One layer of H2G-Former.
    Structure: Pre-Norm -> Sparse Attention -> Residual -> Pre-Norm -> FFN -> Residual
    """
    def __init__(self, hidden_dim, num_heads, dropout, ntypes, etypes):
        super(H2GFormerLayer, self).__init__()
        
        # 1. Heterogeneous Sparse Attention
        convs = {}
        for etype in etypes:
            edge_dim = hidden_dim
            convs[etype] = SparseMultiHeadAttentionConv(
                hidden_dim, hidden_dim, num_heads, edge_dim=edge_dim, dropout=dropout
            )
            
        self.attn = HeteroGraphConv(convs, aggregate='sum')
        
        # 2. Feed Forward Network
        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for ntype in ntypes
        })
        
        self.norm1 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm2 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h_dict, edge_feat_dict=None):
        # Layer 1: Sparse Attention
        h_norm = {ntype: self.norm1[ntype](h) for ntype, h in h_dict.items()}
        
        # Attention
        mod_args = {}
        if edge_feat_dict:
            for etype, feat in edge_feat_dict.items():
                # Pass edge features if availables
                if etype in g.etypes:
                    mod_args[etype] = (feat,)
        
        h_attn = self.attn(g, h_norm, mod_args=mod_args)        
        h_dict_new = {}
        for ntype, h in h_dict.items():
            if ntype in h_attn:
                h_dict_new[ntype] = h + self.dropout(h_attn[ntype])
            else:
                h_dict_new[ntype] = h
                
        # Layer 2: FFN
        h_out_dict = {}
        for ntype, h in h_dict_new.items():
            h_in = self.norm2[ntype](h)
            h_ffn = self.ffn[ntype](h_in)
            h_out_dict[ntype] = h + self.dropout(h_ffn)
            
        return h_out_dict

class H2GFormerNodeClassifier(nn.Module):
    """
    H2G-Former Implementation for ControBench.
    """
    def __init__(self, in_feats, edge_feats, hidden_size, out_classes, 
                 dropout=0.5, n_layers=2, num_heads=4, post_feats=1536,
                 use_parent_context=True, conversation_weight=0.0):
        super(H2GFormerNodeClassifier, self).__init__()
        
        self.n_layers = n_layers
        
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj = nn.Linear(edge_feats, hidden_size)
        
        # Define Layers
        # ntypes/etypes are use to initialize the Hetero structures,
        # by default, hardcoded for ControBench schema (can also be passed dynamically)
        ntypes = ['user', 'post']
        etypes = ['publish', 'comment', 'user_comment_user']
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(H2GFormerLayer(
                hidden_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                ntypes=ntypes,
                etypes=etypes
            ))
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_classes)
        )

    def forward(self, g, node_features, edge_features=None):
        # Encode Nodes & Edge
        h_dict = {
            'user': self.user_proj(node_features['user']),
            'post': self.post_proj(node_features['post'])
        }
        e_dict = {}
        if edge_features:
            for etype, feat in edge_features.items():
                e_dict[etype] = self.edge_proj(feat)
                
        # Apply H2G-Former Layers
        for layer in self.layers:
            h_dict = layer(g, h_dict, e_dict)
            
        return self.classifier(h_dict['user'])