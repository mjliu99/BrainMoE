# networks/gnns.py

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    TransformerConv,
)


def _safe_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_edge_weight_for_gcn_cheb(edge_weight: torch.Tensor):
    """
    GCN/Cheb are sensitive to negative edge weights because degree normalization
    may become invalid. So we use non-negative safe weights here.
    """
    if edge_weight is None:
        return None

    ew = edge_weight.view(-1)
    ew = _safe_tensor(ew)
    ew = ew.abs().clamp_min(1e-6)
    return ew


def _safe_edge_attr_for_transformer(edge_index: torch.Tensor, x: torch.Tensor, edge_weight: torch.Tensor):
    """
    TransformerConv with edge_dim=1 requires edge_attr of shape [E, 1].
    If edge_weight is None, use zeros.
    """
    if edge_weight is not None:
        edge_attr = edge_weight.view(-1, 1)
        edge_attr = _safe_tensor(edge_attr)
    else:
        edge_attr = torch.zeros(
            edge_index.size(1), 1,
            device=x.device,
            dtype=x.dtype,
        )
    return edge_attr


# ============================================================
# Basic backbone layers
# ============================================================

class GCNNetLayer(nn.Module):
    """
    GCN backbone returning node embeddings.
    Uses safe non-negative edge weights.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        cached: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.use_residual = use_residual

        dims = [in_dim] + [hidden_dim] * (num_layers - 1)

        self.gnn_layers = nn.ModuleList([
            GCNConv(d_in, hidden_dim, cached=cached, normalize=True)
            for d_in in dims
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weight=None):
        h = _safe_tensor(x)
        ew = _safe_edge_weight_for_gcn_cheb(edge_weight)

        for conv, norm in zip(self.gnn_layers, self.norms):
            h_res = h

            h = conv(h, edge_index, edge_weight=ew)
            h = _safe_tensor(h)

            h = norm(h)
            h = _safe_tensor(h)

            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = _safe_tensor(h)

            if self.use_residual and h.shape == h_res.shape:
                h = h + h_res
                h = _safe_tensor(h)

        return h


class ChebNetLayer(nn.Module):
    """
    Chebyshev GNN backbone returning node embeddings.
    Uses safe non-negative edge weights.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        K: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_ch = in_dim
        for _ in range(num_layers):
            self.convs.append(ChebConv(in_ch, hidden_dim, K=K, normalization="sym"))
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_ch = hidden_dim

    def forward(self, x, edge_index, edge_weight=None):
        h = _safe_tensor(x)
        ew = _safe_edge_weight_for_gcn_cheb(edge_weight)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h

            h = conv(h, edge_index, edge_weight=ew)
            h = _safe_tensor(h)

            h = norm(h)
            h = _safe_tensor(h)

            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = _safe_tensor(h)

            if self.use_residual and h.shape == h_res.shape:
                h = h + h_res
                h = _safe_tensor(h)

        return h


class GraphTransformerNetLayer(nn.Module):
    """
    TransformerConv backbone returning node embeddings.
    Supports scalar edge weights by converting edge_weight [E] -> edge_attr [E, 1].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.dropout = dropout
        self.edge_dim = edge_dim
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_ch = in_dim
        out_ch = hidden_dim // heads

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=False,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_ch = hidden_dim

    def forward(self, x, edge_index, edge_weight=None):
        h = _safe_tensor(x)
        edge_attr = _safe_edge_attr_for_transformer(edge_index, x, edge_weight)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h

            h = conv(h, edge_index, edge_attr=edge_attr)
            h = _safe_tensor(h)

            h = norm(h)
            h = _safe_tensor(h)

            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = _safe_tensor(h)

            if self.use_residual and h.shape == h_res.shape:
                h = h + h_res
                h = _safe_tensor(h)

        return h


# ============================================================
# Experts for BrainMoE
# ============================================================

class MLPExpert(nn.Module):
    """
    Pure feature expert.
    Unified signature:
        forward(x, edge_index=None, edge_weight=None)
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, edge_weight=None):
        return _safe_tensor(self.net(_safe_tensor(x)))


class GCNExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = GCNNetLayer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cached=False,
            use_residual=True,
        )

    def forward(self, x, edge_index, edge_weight=None):
        return self.backbone(x, edge_index, edge_weight=edge_weight)


class ChebNetExpert(nn.Module):
    def __init__(self, hidden_dim: int, K: int = 3, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = ChebNetLayer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K,
            dropout=dropout,
            use_residual=True,
        )

    def forward(self, x, edge_index, edge_weight=None):
        return self.backbone(x, edge_index, edge_weight=edge_weight)


class GraphTransformerExpert(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = GraphTransformerNetLayer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            use_residual=True,
        )

    def forward(self, x, edge_index, edge_weight=None):
        return self.backbone(x, edge_index, edge_weight=edge_weight)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    num_nodes = 5
    x = torch.randn(num_nodes, 64)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1],
         [1, 2, 3, 4, 0, 3]],
        dtype=torch.long,
    )
    edge_weight = torch.randn(edge_index.size(1))  # include negative weights

    experts = [
        MLPExpert(64),
        GCNExpert(64),
        ChebNetExpert(64),
        GraphTransformerExpert(64),
    ]

    print("[Check with edge_weight]")
    for exp in experts:
        y = exp(x, edge_index, edge_weight)
        print(f"{exp.__class__.__name__:>24s} -> {tuple(y.shape)} | nan={torch.isnan(y).any().item()}")

    print("\n[Check without edge_weight]")
    for exp in experts:
        y = exp(x, edge_index, None)
        print(f"{exp.__class__.__name__:>24s} -> {tuple(y.shape)} | nan={torch.isnan(y).any().item()}")