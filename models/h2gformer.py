import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import dgl
from dgl.nn import HeteroGraphConv, GraphConv


class SparseNodeTransformerLayer(nn.Module):
    """
    Sparse Node Transformer layer for DGL heterogeneous graphs.
    Implements edge-level sparse attention with per-node-type projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ntypes: List[str],
        num_etypes: int,
        edge_weight: bool = False,
        attn_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ntypes = list(ntypes)

        self.q_lin = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.ntypes})
        self.k_lin = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.ntypes})
        self.v_lin = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.ntypes})
        self.o_lin = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.ntypes})

        if edge_weight:
            H, D = self.num_heads, self.head_dim
            self.edge_weights = nn.Parameter(torch.empty(num_etypes, H, D, D))
            self.msg_weights = nn.Parameter(torch.empty(num_etypes, H, D, D))
            nn.init.xavier_uniform_(self.edge_weights)
            nn.init.xavier_uniform_(self.msg_weights)
        else:
            self.edge_weights = None
            self.msg_weights = None

        self.dropout_attn = nn.Dropout(attn_dropout if attn_dropout is not None else dropout)

    @staticmethod
    def _scatter_max_per_dst(
        scores: torch.Tensor,  # [H, E]
        dst: torch.Tensor,     # [E]
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute per-head max score for each destination node."""
        H, E = scores.shape
        expanded_dst_nodes = dst.repeat(H, 1)  # [H, E]

        try:
            from torch_scatter import scatter_max  # type: ignore
            max_scores, _ = scatter_max(scores, expanded_dst_nodes, dim=1, dim_size=num_nodes)  # [H, L]
        except Exception:
            max_scores = torch.full((H, num_nodes), float("-inf"), device=scores.device, dtype=scores.dtype)
            max_scores.scatter_reduce_(1, expanded_dst_nodes, scores, reduce="amax", include_self=True)

        return max_scores.gather(1, expanded_dst_nodes)  # [H, E]

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with sparse edge-level attention."""
        homo_g = dgl.to_homogeneous(g, store_type=True)

        node_type_tensor = homo_g.ndata[dgl.NTYPE]  # [N]
        edge_type_tensor = homo_g.edata[dgl.ETYPE] if dgl.ETYPE in homo_g.edata else None

        N = homo_g.num_nodes()
        device = next(iter(h_dict.values())).device
        dtype = next(iter(h_dict.values())).dtype

        q = torch.empty((N, self.hidden_dim), device=device, dtype=dtype)
        k = torch.empty((N, self.hidden_dim), device=device, dtype=dtype)
        v = torch.empty((N, self.hidden_dim), device=device, dtype=dtype)

        for nt in self.ntypes:
            nt_id = g.get_ntype_id(nt)
            mask = (node_type_tensor == nt_id)
            if mask.any():
                x = h_dict[nt]
                q[mask] = self.q_lin[nt](x)
                k[mask] = self.k_lin[nt](x)
                v[mask] = self.v_lin[nt](x)

        H, D = self.num_heads, self.head_dim
        q = q.view(N, H, D).transpose(0, 1).contiguous()  # [H, N, D]
        k = k.view(N, H, D).transpose(0, 1).contiguous()  # [H, N, D]
        v = v.view(N, H, D).transpose(0, 1).contiguous()  # [H, N, D]

        src_nodes, dst_nodes = homo_g.edges()  # each [E]
        E = src_nodes.shape[0]
        L = N

        edge_q = q[:, dst_nodes, :]  # [H, E, D]
        edge_k = k[:, src_nodes, :]  # [H, E, D]
        edge_v = v[:, src_nodes, :]  # [H, E, D]

        if self.edge_weights is not None and edge_type_tensor is not None:
            ew = self.edge_weights[edge_type_tensor]  # [E, H, D, D]
            ew = ew.transpose(0, 1).contiguous()  # [H, E, D, D]
            edge_k = torch.matmul(ew, edge_k.unsqueeze(-1)).squeeze(-1)  # [H, E, D]

        edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D)  # [H, E]
        edge_scores = torch.clamp(edge_scores, min=-5.0, max=5.0)

        max_scores_gathered = self._scatter_max_per_dst(edge_scores, dst_nodes, num_nodes=L)  # [H, E]
        exp_scores = torch.exp(edge_scores - max_scores_gathered)  # [H, E]
        expanded_dst_nodes = dst_nodes.repeat(H, 1)  # [H, E]

        sum_exp_scores = torch.zeros((H, L), device=device, dtype=dtype)  # [H, L]
        sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)

        edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)  # [H, E]
        edge_scores = edge_scores.unsqueeze(-1)  # [H, E, 1]
        edge_scores = self.dropout_attn(edge_scores)

        out = torch.zeros((H, L, D), device=device, dtype=dtype)  # [H, N, D]
        out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, E, D)), edge_scores * edge_v)

        out = out.transpose(0, 1).contiguous().view(L, H * D)  # [N, hidden_dim]

        h_out: Dict[str, torch.Tensor] = {}
        for nt in self.ntypes:
            nt_id = g.get_ntype_id(nt)
            mask = (node_type_tensor == nt_id)
            if mask.any():
                h_out[nt] = self.o_lin[nt](out[mask])
            else:
                h_out[nt] = h_dict[nt]

        return h_out


class H2GFormerLayer(nn.Module):
    """H2GFormer layer combining local GNN, sparse transformer attention, and FFN."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 ntypes: List[str], etypes: List[str], 
                 layers_pre_gt: int = 1, layers_post_gt: int = 1,
                 edge_weight: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ntypes = ntypes
        self.etypes = etypes
        
        self.pre_gnn_layers = nn.ModuleList()
        for _ in range(layers_pre_gt):
            local_convs = {etype: GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True) 
                          for etype in etypes}
            self.pre_gnn_layers.append(HeteroGraphConv(local_convs, aggregate='sum'))
        
        self.sparse_attn = SparseNodeTransformerLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ntypes=ntypes,
            num_etypes=len(etypes),
            edge_weight=edge_weight,
        )

        self.post_gnn_layers = nn.ModuleList()
        for _ in range(layers_post_gt):
            local_convs = {etype: GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True) 
                          for etype in etypes}
            self.post_gnn_layers.append(HeteroGraphConv(local_convs, aggregate='sum'))
        
        self.norm_pre = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm_attn = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm_post = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        self.norm_ffn = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in ntypes})
        
        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ) for ntype in ntypes
        })
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, g: dgl.DGLGraph, h_dict: Dict[str, torch.Tensor],
                edge_feat_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through pre-GNN, sparse attention, post-GNN, and FFN."""
        h_current = h_dict
        for pre_gnn in self.pre_gnn_layers:
            h_normalized = {ntype: self.norm_pre[ntype](h) for ntype, h in h_current.items()}
            h_gnn = pre_gnn(g, h_normalized)
            h_current = {
                ntype: h_current[ntype] + self.dropout_layer(h_gnn.get(ntype, h_current[ntype]))
                for ntype in h_current
            }
        
        h_normalized = {ntype: self.norm_attn[ntype](h) for ntype, h in h_current.items()}
        h_attn = self.sparse_attn(g, h_normalized)
        h_current = {
            ntype: h_current[ntype] + self.dropout_layer(h_attn.get(ntype, h_current[ntype]))
            for ntype in h_current
        }
        
        for post_gnn in self.post_gnn_layers:
            h_normalized = {ntype: self.norm_post[ntype](h) for ntype, h in h_current.items()}
            h_gnn = post_gnn(g, h_normalized)
            h_current = {
                ntype: h_current[ntype] + self.dropout_layer(h_gnn.get(ntype, h_current[ntype]))
                for ntype in h_current
            }
        
        h_out = {}
        for ntype in h_current:
            h_norm = self.norm_ffn[ntype](h_current[ntype])
            h_ffn = self.ffn[ntype](h_norm)
            h_out[ntype] = h_current[ntype] + h_ffn
        
        return h_out


class H2GFormerNodeClassifier(nn.Module):
    """H2GFormer node classifier for ControBench."""
    
    def __init__(self, in_feats: int, edge_feats: int, hidden_size: int,
                 out_classes: int, dropout: float = 0.5, n_layers: int = 3,
                 num_heads: int = 8, post_feats: int = 1536,
                 layers_pre_gt: int = 1, layers_post_gt: int = 1,
                 edge_weight: bool = False,
                 num_classes: Optional[int] = None, use_label_emb: bool = False,
                 **kwargs) -> None:
        super().__init__()
        
        self.n_layers = n_layers
        self.use_label_emb = bool(use_label_emb and num_classes is not None)
        self.num_classes = num_classes if self.use_label_emb else 0
        
        self.user_proj = nn.Linear(in_feats, hidden_size)
        self.post_proj = nn.Linear(post_feats, hidden_size)
        
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
                layers_pre_gt=layers_pre_gt,
                layers_post_gt=layers_post_gt,
                edge_weight=edge_weight
            ) for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_classes)
        )
        
    def forward(self, g: dgl.DGLGraph, node_features: Dict[str, torch.Tensor],
                edge_features: Optional[Dict[str, torch.Tensor]] = None,
                labels: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute user node predictions."""
        h_dict = {
            'user': self.user_proj(node_features['user']),
            'post': self.post_proj(node_features['post'])
        }
        
        if self.use_label_emb and labels is not None and 'user' in labels:
            user_labels = labels['user'].to(torch.long)
            embed_idx = torch.where(user_labels >= 0, user_labels + 1, torch.zeros_like(user_labels))
            label_vecs = self.label_emb(embed_idx)
            h_dict['user'] = h_dict['user'] + self.label_dropout(label_vecs)
        
        for layer in self.layers:
            h_dict = layer(g, h_dict, edge_features)
        
        return self.classifier(h_dict['user'])

