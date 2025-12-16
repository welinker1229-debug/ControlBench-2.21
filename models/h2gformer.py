import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import HeteroGraphConv
from dgl.utils import expand_as_pair


class SparseMultiHeadAttentionConv(nn.Module):
    """
    Sparse Graph Transformer Convolution.
    Implements the core logic of H2G‑Former's SparseNodeTransformer.
    """
    def __init__(self, in_dim, out_dim, num_heads, edge_dim=None, dropout=0.0):
        super().__init__()
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

    def forward(self, g: dgl.DGLGraph, h, e_feat=None):
        """Perform sparse multi‑head attention on a heterogeneous graph."""
        h_src, h_dst = expand_as_pair(h, g)

        # Step 1. Project Q, K, V
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
    One layer of the H2G‑Former.

    Structure: Pre‑Norm → k‑hop Sparse Attention → Residual → Pre‑Norm →
    Feed‑Forward → Residual.  The number of hops per layer is
    configurable via `num_hops`.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 ntypes, etypes, num_hops: int = 1):
        super().__init__()
        self.num_hops = max(1, num_hops)

        # 1. Heterogeneous Sparse Attention (per edge type)
        convs = {}
        for etype in etypes:
            edge_dim = hidden_dim  # project edge features to hidden_dim per head
            convs[etype] = SparseMultiHeadAttentionConv(
                hidden_dim, hidden_dim, num_heads, edge_dim=edge_dim, dropout=dropout
            )
        self.attn = HeteroGraphConv(convs, aggregate='sum')

        # 2. Feed Forward Network (type‑specific)
        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for ntype in ntypes
        })

        # Layer Normalisation per node type
        self.norm1 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm2 = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, h_dict: dict, edge_feat_dict: dict = None):
        """
        Forward pass for a single H2G‑Former layer.

        Args:
            g: The heterogeneous graph.
            h_dict: A dictionary of node features keyed by node type.
            edge_feat_dict: Optional dictionary of edge features keyed by edge type.

        Returns:
            A new dictionary of node features after attention and FFN.
        """
        h_norm = {ntype: self.norm1[ntype](h) for ntype, h in h_dict.items()}

        # Prepare edge arguments if edge features are provided
        mod_args = {}
        if edge_feat_dict:
            for etype, feat in edge_feat_dict.items():
                # Only pass edge features for existing edge types
                if etype in g.etypes:
                    mod_args[etype] = (feat,)

        # Multi‑hop attention: repeat attention aggregation `num_hops` times
        h_current = h_norm
        for _ in range(self.num_hops):
            h_attn = self.attn(g, h_current, mod_args=mod_args)
            h_next = {}
            for ntype, h_old in h_current.items():
                if ntype in h_attn:
                    h_next[ntype] = h_old + self.dropout(h_attn[ntype])
                else:
                    h_next[ntype] = h_old
            h_current = h_next

        # Apply second layer norm and type‑specific FFN with residual connection
        h_out = {}
        for ntype, h_val in h_current.items():
            h_in = self.norm2[ntype](h_val)
            h_ffn = self.ffn[ntype](h_in)
            h_out[ntype] = h_val + self.dropout(h_ffn)
        return h_out


class H2GFormerNodeClassifier(nn.Module):
    """
    H2G‑Former implementation for ControBench with optional label embeddings
    and configurable k‑hop attention.

    Args:
        in_feats: Dimension of user node features.
        edge_feats: Dimension of raw edge features.
        hidden_size: Hidden feature dimension used in all layers.
        out_classes: Number of output classes.
        dropout: Dropout rate.
        n_layers: Number of H2G‑Former layers.
        num_heads: Number of attention heads.
        post_feats: Dimension of post (text) features.
        num_classes: Number of distinct labels (for label embedding).  If
            provided and `use_label_emb` is True, labels will be embedded.
        use_label_emb: Whether to add label embeddings to node features.
        num_hops: Number of hops to aggregate within each layer (>=1).
    """
    def __init__(self, in_feats: int, edge_feats: int, hidden_size: int,
                 out_classes: int, dropout: float = 0.5, n_layers: int = 2,
                 num_heads: int = 4, post_feats: int = 1536,
                 num_classes: int = None, use_label_emb: bool = False,
                 num_hops: int = 2, **kwargs):
        super().__init__()

        self.n_layers = n_layers
        self.use_label_emb = use_label_emb and num_classes is not None
        self.num_classes = num_classes if self.use_label_emb else 0
        self.num_hops = max(1, num_hops)

        # Projections for node and edge raw features
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj = nn.Linear(edge_feats, hidden_size)

        # Label embedding: reserve index 0 for unknown/masked labels
        if self.use_label_emb:
            self.label_emb = nn.Embedding(self.num_classes + 1, hidden_size)
            self.label_dropout = nn.Dropout(dropout)
        else:
            self.label_emb = None

        # Node and edge types for ControBench schema (can be parameterised)
        ntypes = ['user', 'post']
        etypes = ['publish', 'comment', 'user_comment_user']

        # Construct H2G‑Former layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(H2GFormerLayer(
                hidden_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                ntypes=ntypes,
                etypes=etypes,
                num_hops=self.num_hops
            ))

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_classes)
        )

    def forward(self, g: dgl.DGLGraph, node_features: dict,
                edge_features: dict = None, labels: dict = None):
        """
        Forward pass for the node classifier.

        Args:
            g: Heterogeneous DGL graph.
            node_features: Dictionary with raw node features keyed by node type.
            edge_features: Optional dictionary of raw edge features keyed by edge type.
            labels: Optional dictionary of node labels keyed by node type.  Labels
                should be integer tensors with −1 for unknown/masked labels.

        Returns:
            Predictions for the 'user' node type.
        """
        # Encode node features
        h_dict = {
            'user': self.user_proj(node_features['user']),
            'post': self.post_proj(node_features['post'])
        }

        # Optionally add label embeddings for user nodes
        if self.use_label_emb and labels is not None and 'user' in labels:
            user_labels = labels['user']
            user_labels = user_labels.to(torch.long)
            embed_idx = torch.where(user_labels >= 0, user_labels + 1, torch.zeros_like(user_labels))
            label_vecs = self.label_emb(embed_idx)
            h_dict['user'] = h_dict['user'] + self.label_dropout(label_vecs)

        # Encode edge features if provided
        e_dict = {}
        if edge_features:
            for etype, feat in edge_features.items():
                e_dict[etype] = self.edge_proj(feat)

        for layer in self.layers:
            h_dict = layer(g, h_dict, e_dict)

        return self.classifier(h_dict['user'])