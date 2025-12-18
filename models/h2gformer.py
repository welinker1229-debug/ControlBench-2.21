import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import HeteroGraphConv, GraphConv
from dgl.utils import expand_as_pair


class SparseMultiHeadAttentionConv(nn.Module):
    """Sparse graph transformer convolution with optional edge bias."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 edge_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        if edge_dim is not None and edge_dim > 0:
            self.W_e = nn.Linear(edge_dim, num_heads)
            self.use_edge = True
        else:
            self.use_edge = False

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor,
                e_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Sparse multi‑head attention on a (sub‑)graph."""
        h_src, h_dst = expand_as_pair(h, g)

        q = self.W_q(h_dst).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(h_src).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(h_src).view(-1, self.num_heads, self.head_dim)

        g.srcdata.update({'k': k, 'v': v})
        g.dstdata.update({'q': q})
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))

        if self.use_edge and e_feat is not None:
            e_bias = self.W_e(e_feat).view(-1, self.num_heads, 1)
            g.edata['score'] = (g.edata['score'] * self.scale) + e_bias
        else:
            g.edata['score'] = g.edata['score'] * self.scale

        g.edata['a'] = self.attn_drop(dgl.nn.functional.edge_softmax(g, g.edata['score']))

        g.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'out'))
        out = g.dstdata['out'].view(-1, self.out_dim)
        return self.out_proj(out)


class H2GFormerLayer(nn.Module):
    """One H2G‑Former block combining local conv, attention, and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 ntypes: list[str], etypes: list[str], num_hops: int = 1) -> None:
        super().__init__()
        self.num_hops = max(1, num_hops)

        local_convs: dict[str, nn.Module] = {}
        for etype in etypes:
            local_convs[etype] = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.local_conv = HeteroGraphConv(local_convs, aggregate='sum')

        attn_convs: dict[str, nn.Module] = {}
        for etype in etypes:
            attn_convs[etype] = SparseMultiHeadAttentionConv(
                hidden_dim, hidden_dim, num_heads, edge_dim=hidden_dim, dropout=dropout
            )
        self.attn = HeteroGraphConv(attn_convs, aggregate='sum')

        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for ntype in ntypes
        })

        self.norm_local = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm_attn = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm_ffn = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, h_dict: dict[str, torch.Tensor],
                edge_feat_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Forward pass for a single H2G‑Former layer."""
        h_norm_local = {ntype: self.norm_local[ntype](h) for ntype, h in h_dict.items()}
        h_local = self.local_conv(g, h_norm_local)
        h_after_local: dict[str, torch.Tensor] = {}
        for ntype, h_orig in h_dict.items():
            if ntype in h_local:
                h_after_local[ntype] = h_orig + self.dropout(h_local[ntype])
            else:
                h_after_local[ntype] = h_orig

        h_norm_attn = {ntype: self.norm_attn[ntype](h) for ntype, h in h_after_local.items()}

        mod_args: dict[str, tuple] = {}
        if edge_feat_dict:
            for etype, feat in edge_feat_dict.items():
                if etype in g.etypes:
                    mod_args[etype] = (feat,)

        h_current = h_norm_attn
        for _ in range(self.num_hops):
            h_attn = self.attn(g, h_current, mod_args=mod_args)
            h_next: dict[str, torch.Tensor] = {}
            for ntype, h_old in h_current.items():
                if ntype in h_attn:
                    h_next[ntype] = h_old + self.dropout(h_attn[ntype])
                else:
                    h_next[ntype] = h_old
            h_current = h_next
        h_global = h_current

        h_out: dict[str, torch.Tensor] = {}
        for ntype, h_val in h_global.items():
            h_in = self.norm_ffn[ntype](h_val)
            h_ffn = self.ffn[ntype](h_in)
            h_out[ntype] = h_val + self.dropout(h_ffn)
        return h_out


class H2GFormerNodeClassifier(nn.Module):
    """H2G‑Former node classifier for ControBench."""

    def __init__(self, in_feats: int, edge_feats: int, hidden_size: int,
                 out_classes: int, dropout: float = 0.5, n_layers: int = 2,
                 num_heads: int = 4, post_feats: int = 1536,
                 num_classes: int | None = None, use_label_emb: bool = False,
                 num_hops: int = 2, **kwargs) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.use_label_emb = bool(use_label_emb and num_classes is not None)
        self.num_classes = num_classes if self.use_label_emb else 0
        self.num_hops = max(1, num_hops)

        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        self.edge_proj = nn.Linear(edge_feats, hidden_size)

        if self.use_label_emb:
            self.label_emb = nn.Embedding(self.num_classes + 1, hidden_size)
            self.label_dropout = nn.Dropout(dropout)
        else:
            self.label_emb = None

        ntypes = ['user', 'post']
        etypes = ['publish', 'comment', 'user_comment_user']

        self.layers = nn.ModuleList([
            H2GFormerLayer(
                hidden_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                ntypes=ntypes,
                etypes=etypes,
                num_hops=self.num_hops
            ) for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_classes)
        )

    def forward(self, g: dgl.DGLGraph, node_features: dict[str, torch.Tensor],
                edge_features: dict[str, torch.Tensor] | None = None,
                labels: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Compute user node predictions."""
        h_dict: dict[str, torch.Tensor] = {
            'user': self.user_proj(node_features['user']),
            'post': self.post_proj(node_features['post'])
        }

        if self.use_label_emb and labels is not None and 'user' in labels:
            user_labels = labels['user'].to(torch.long)
            embed_idx = torch.where(user_labels >= 0, user_labels + 1, torch.zeros_like(user_labels))
            label_vecs = self.label_emb(embed_idx)
            h_dict['user'] = h_dict['user'] + self.label_dropout(label_vecs)

        e_dict: dict[str, torch.Tensor] = {}
        if edge_features:
            for etype, feat in edge_features.items():
                e_dict[etype] = self.edge_proj(feat)

        for layer in self.layers:
            h_dict = layer(g, h_dict, e_dict)

        return self.classifier(h_dict['user'])
