# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .gnns import GCNExpert, TwoHopGCNExpert, GraphTransformerExpert, ChebNetExpert
# from torch_geometric.nn import global_mean_pool

# class NoisyTopKRouter(nn.Module):
#     """
#     Q(h) = hWq + eps * Softplus(hWn)
#     """
#     def __init__(self, dim: int, num_experts: int):
#         super().__init__()
#         self.Wq = nn.Linear(dim, num_experts, bias=False)
#         self.Wn = nn.Linear(dim, num_experts, bias=False)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         router_logits = self.Wq(h)  # [N,E]
#         if self.training:
#             noise_scale = F.softplus(self.Wn(h))  # [N,E] >= 0
#             eps = torch.randn_like(router_logits)
#             return router_logits + 0.1 * (eps * noise_scale)
#         return router_logits
    
# class BrainMoE(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         llm_dim: int,
#         hidden_dim: int,
#         num_experts: int = 4,
#         top_k: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         assert num_experts == 4, "This example assumes 4 experts: GCN, 2-hop GCN, GraphTransformer, ChebNet."
#         self.num_experts = num_experts
#         self.top_k = top_k

#         self.projection_head_input = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(dropout),
#             # nn.Linear(in_dim, hidden_dim),
#         )

#         self.projection_head_llm = nn.Sequential(
#             nn.Linear(llm_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(dropout),
#             # nn.Linear(llm_dim//2, hidden_dim),
#         )

#         self.dropout = nn.Dropout(dropout)

#         self.router = NoisyTopKRouter(hidden_dim, num_experts)

#         # Expert library
#         self.experts = nn.ModuleList([
#             GCNExpert(hidden_dim),                # Expert 0: GCN
#             TwoHopGCNExpert(hidden_dim),          # Expert 1: 2-hop GCN
#             GraphTransformerExpert(hidden_dim),   # Expert 2: Graph Transformer (light)
#             ChebNetExpert(hidden_dim, K=3),       # Expert 3: ChebNet (light)
#         ])

#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim// 2, hidden_dim // 4),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim// 4, 2),
#         )


#     def _topk_gates(self, fused_logits):
#         """
#         is this node-level moe?
#         """
#         N, E = fused_logits.shape
#         k = min(self.top_k, E)

#         topk_vals, topk_idx = torch.topk(fused_logits, k=k, dim=-1)  # [N,k]
#         topk_w = F.softmax(topk_vals, dim=-1)  # [N,k]

#         gates = torch.zeros(N, E, device=fused_logits.device, dtype=fused_logits.dtype)
#         gates.scatter_(1, topk_idx, topk_w)
#         return gates 

#     def forward(
#         self,
#         x,
#         edge_index,
#         llm_embeddings,
#         batch
#     ):
#         # print("x shape:", x.shape)
#         N = x.size(0)
#         h = F.relu(self.projection_head_input(x))        # node representation h_i
#         llm_h = F.relu(self.projection_head_llm(llm_embeddings))  # LLM representation

#         h = h + llm_h
#         h = self.dropout(h)

#         router_logits = self.router(h)        # Q(h)
#         gates = self._topk_gates(router_logits)  # G(h) in dense form

#         # Run all experts in parallel
#         expert_outs = []
#         for expert in self.experts:
#             expert_outs.append(expert(h, edge_index))  # each [N, hidden_dim]

#         # Weighted sum across experts: sum_e gate[i,e] * E_e(h)[i]
#         # print(gates)
#         out = torch.zeros_like(h)
#         for e, E_out in enumerate(expert_outs):
#             out = out + gates[:, e:e+1] * E_out

#         out = out + h
#         out = self.dropout(out)
#         out = global_mean_pool(out, batch)
#         out = self.classification_head(out)
#         return out, gates

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .gnns import (
    GCNExpert,
    GraphTransformerExpert,
    ChebNetExpert,
    MLPExpert,
)


def inverse_sigmoid(p: float, eps: float = 1e-6) -> float:
    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


class NoisyTopKRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, noise_std: float = 0.1):
        super().__init__()
        self.Wq = nn.Linear(dim, num_experts, bias=False)
        self.Wn = nn.Linear(dim, num_experts, bias=False)
        self.noise_std = noise_std

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.Wq(h)  # [N, E]
        if self.training and self.noise_std > 0:
            noise_scale = F.softplus(self.Wn(h))
            eps = torch.randn_like(logits)
            logits = logits + self.noise_std * eps * noise_scale
        return logits


class BrainMoE(nn.Module):
    """
    Experts: [MLP, Cheb, GT, GCN]
    Node-level routing
    Graph-level LLM prior (stage1) with soft probability mixing
    Graph-level gated fusion (stage2)
    Mean pooling readout
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        identity_dim: int = 0,
        llm_dim: int = 3072,
        llm_hidden_dim: int = 64,
        top_k: int = 2,
        dropout: float = 0.1,
        use_identity: bool = True,
        use_llm_stage1: bool = False,
        use_llm_stage2: bool = False,
        router_noise_std: float = 0.1,
        router_temperature: float = 1.5,
        stage1_scale_init: float = 0.35,
    ):
        super().__init__()

        self.num_experts = 4
        self.top_k = top_k
        self.current_top_k = top_k
        self.router_temperature = router_temperature

        self.use_identity = use_identity
        self.use_llm_stage1 = use_llm_stage1
        self.use_llm_stage2 = use_llm_stage2
        self.hidden_dim = hidden_dim
        self.llm_input_dim = llm_dim
        self.dropout = nn.Dropout(dropout)

        # 1) node feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 2) identity encoder
        if self.use_identity:
            assert identity_dim > 0, "identity_dim must be > 0 when use_identity=True"
            self.identity_encoder = nn.Sequential(
                nn.Linear(identity_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            router_in_dim = hidden_dim * 2
            fuse_in_dim = hidden_dim * 2
        else:
            self.identity_encoder = None
            router_in_dim = hidden_dim
            fuse_in_dim = hidden_dim

        # 3) node-level router
        self.router = NoisyTopKRouter(
            dim=router_in_dim,
            num_experts=self.num_experts,
            noise_std=router_noise_std,
        )

        # 4) node representation fusion before experts
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 5) expert library
        self.experts = nn.ModuleList([
            MLPExpert(hidden_dim, num_layers=2, dropout=dropout),
            ChebNetExpert(hidden_dim, K=2, num_layers=2, dropout=dropout),
            GraphTransformerExpert(hidden_dim, num_layers=2, heads=4, dropout=dropout),
            GCNExpert(hidden_dim, num_layers=2, dropout=dropout),
        ])

        self.expert_post_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_experts)
        ])

        self.expert_scales = nn.Parameter(torch.ones(self.num_experts))

        # 6) stage1: graph-level router prior
        if self.use_llm_stage1:
            self.stage1_proj = nn.Sequential(
                nn.Linear(llm_dim, llm_hidden_dim),
                nn.LayerNorm(llm_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(llm_hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.graph_router_head = nn.Linear(hidden_dim, self.num_experts)

            # raw parameter -> sigmoid(alpha) in forward
            self.graph_router_mix_logit = nn.Parameter(
                torch.tensor(inverse_sigmoid(stage1_scale_init), dtype=torch.float)
            )
        else:
            self.stage1_proj = None
            self.graph_router_head = None
            self.graph_router_mix_logit = None

        # 7) stage2: graph-level gated fusion
        if self.use_llm_stage2:
            self.llm_proj = nn.Sequential(
                nn.Linear(llm_dim, llm_hidden_dim),
                nn.LayerNorm(llm_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(llm_hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.llm_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.llm_proj = None
            self.llm_gate = None

        # 8) classifier
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.logit_bias = nn.Parameter(torch.zeros(2))

    def _topk_gates_from_prob(self, router_prob: torch.Tensor) -> torch.Tensor:
        """
        router_prob: [N, E]
        return: sparse top-k gates [N, E]
        """
        n, e = router_prob.shape
        k = min(self.current_top_k, e)

        topk_vals, topk_idx = torch.topk(router_prob, k=k, dim=-1)
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        gates = torch.zeros_like(router_prob)
        gates.scatter_(1, topk_idx, topk_w)
        return gates

    def _reshape_graph_level_llm(
        self,
        llm_tensor: torch.Tensor,
        batch: torch.Tensor,
        llm_dim: int,
    ) -> torch.Tensor:
        if llm_tensor is None:
            return None

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        if llm_tensor.dim() == 1:
            expected = num_graphs * llm_dim
            if llm_tensor.numel() != expected:
                raise ValueError(
                    f"Flattened LLM tensor has numel={llm_tensor.numel()}, "
                    f"but expected num_graphs({num_graphs}) * llm_dim({llm_dim}) = {expected}"
                )
            llm_tensor = llm_tensor.view(num_graphs, llm_dim)
        elif llm_tensor.dim() == 2:
            pass
        else:
            raise ValueError(
                f"Unexpected llm tensor dim={llm_tensor.dim()}, shape={tuple(llm_tensor.shape)}"
            )

        return llm_tensor

    @staticmethod
    def _graph_mean_from_node_distribution(node_dist: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        node_dist: [N, E]
        batch: [N]
        return: [B, E]
        """
        if batch.numel() == 0:
            return torch.zeros(0, node_dist.size(-1), device=node_dist.device, dtype=node_dist.dtype)

        bsz = int(batch.max().item()) + 1
        out = torch.zeros(bsz, node_dist.size(-1), device=node_dist.device, dtype=node_dist.dtype)
        out.index_add_(0, batch, node_dist)
        counts = torch.bincount(batch, minlength=bsz).to(node_dist.device).clamp_min(1).unsqueeze(-1).float()
        out = out / counts
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor = None,
        node_identity: torch.Tensor = None,
        llm_stage1: torch.Tensor = None,
        llm_stage2: torch.Tensor = None,
    ):
        """
        x            : [N, in_dim]
        edge_index   : [2, E]
        batch        : [N]
        edge_weight  : [E] or None
        node_identity: [N, identity_dim] or None
        llm_stage1   : [B, llm_dim] or flattened [B*llm_dim]
        llm_stage2   : [B, llm_dim] or flattened [B*llm_dim]
        """
        aux = {}

        if self.use_llm_stage1 and llm_stage1 is not None:
            llm_stage1 = self._reshape_graph_level_llm(
                llm_stage1, batch, self.llm_input_dim
            )

        if self.use_llm_stage2 and llm_stage2 is not None:
            llm_stage2 = self._reshape_graph_level_llm(
                llm_stage2, batch, self.llm_input_dim
            )

        # node feature encoding
        h_x = self.feature_encoder(x)  # [N, H]

        # identity-aware routing/input
        if self.use_identity:
            if node_identity is None:
                raise ValueError("node_identity is required when use_identity=True")
            h_id = self.identity_encoder(node_identity)           # [N, H]
            h_router = torch.cat([h_x, h_id], dim=-1)            # [N, 2H]
            h = self.fuse(torch.cat([h_x, h_id], dim=-1))        # [N, H]
        else:
            h_router = h_x
            h = self.fuse(h_x)

        # node-level router logits
        node_logits = self.router(h_router)                      # [N, E]
        aux["node_logits"] = node_logits

        # base node routing probability
        node_prob = F.softmax(node_logits / max(self.router_temperature, 1e-6), dim=-1)

        # stage1 graph-level prior with soft probability mixing
        graph_logits = None
        if self.use_llm_stage1:
            if llm_stage1 is None:
                raise ValueError("llm_stage1 is required when use_llm_stage1=True")

            h_g = self.stage1_proj(llm_stage1)                   # [B, H]
            graph_logits = self.graph_router_head(h_g)          # [B, E]
            graph_prob = F.softmax(graph_logits, dim=-1)        # [B, E]

            mix_alpha = torch.sigmoid(self.graph_router_mix_logit)
            router_prob = (1.0 - mix_alpha) * node_prob + mix_alpha * graph_prob[batch]
            router_prob = router_prob / router_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            aux["graph_logits"] = graph_logits
            aux["graph_prob"] = graph_prob
            aux["stage1_mix_alpha"] = mix_alpha.detach()
            aux["p_net"] = graph_prob
        else:
            router_prob = node_prob

        aux["router_prob"] = router_prob

        # use node logits for z-loss stability
        aux["router_logits"] = node_logits

        # sparse node-level gates from mixed probability
        gates = self._topk_gates_from_prob(router_prob)         # [N, E]
        aux["gates"] = gates

        # graph-level posterior from hard gates
        p_router = self._graph_mean_from_node_distribution(gates, batch)  # [B, E]
        aux["p_router"] = p_router

        h = self.dropout(h)

        # expert forward
        expert_outs = []
        for i, expert in enumerate(self.experts):
            if i == 0:  # MLP expert
                e_out = expert(h, None, None)
            else:
                e_out = expert(h, edge_index, edge_weight)

            e_out = self.expert_post_norms[i](e_out)
            e_out = self.expert_scales[i] * e_out
            expert_outs.append(e_out)

        out = torch.zeros_like(h)
        for e, e_out in enumerate(expert_outs):
            out = out + gates[:, e:e + 1] * e_out

        out = out + h
        out = self.dropout(out)

        graph_emb = global_mean_pool(out, batch)                # [B, H]
        aux["graph_emb_before_stage2"] = graph_emb

        # stage2 gated fusion
        if self.use_llm_stage2:
            if llm_stage2 is None:
                raise ValueError("llm_stage2 is required when use_llm_stage2=True")
            llm2 = self.llm_proj(llm_stage2)                    # [B, H]
            alpha = self.llm_gate(torch.cat([graph_emb, llm2], dim=-1))  # [B, H]
            graph_emb = graph_emb + alpha * llm2
            aux["llm_stage2_proj"] = llm2
            aux["llm_stage2_gate"] = alpha

        aux["graph_emb_after_stage2"] = graph_emb

        logits = self.classification_head(graph_emb) + self.logit_bias
        return logits, gates, aux