# /root/autodl-tmp/BrainMoE-02/main.py
import argparse
import json
import random
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.utils import get_subjects
from networks.BrainMoe import BrainMoE


DATASET_PATH = Path("./data")


# =========================================================
# basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mean_std(metrics_list, key):
    vals = np.array([m[key] for m in metrics_list], dtype=float)
    if len(vals) <= 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))


# =========================================================
# AAL network labels from E column
# =========================================================
def load_aal_net_ids_from_xlsx(aal_xlsx_path: str, n_nodes: int):
    aal_xlsx_path = Path(aal_xlsx_path)
    if not aal_xlsx_path.exists():
        raise FileNotFoundError(f"AAL xlsx not found: {aal_xlsx_path}")

    df = pd.read_excel(aal_xlsx_path, header=0)
    if df.shape[1] < 5:
        raise ValueError(f"AAL.xlsx has only {df.shape[1]} columns; E column required.")

    raw_series = df.iloc[:, 4].iloc[:n_nodes]
    if len(raw_series) < n_nodes:
        raise ValueError(f"AAL E column only has {len(raw_series)} rows, but need {n_nodes}.")

    raw_labels = []
    for v in raw_series.tolist():
        if pd.isna(v):
            raw_labels.append("UNKNOWN")
        else:
            raw_labels.append(str(v).strip())

    unique_labels = []
    for x in raw_labels:
        if x not in unique_labels:
            unique_labels.append(x)

    net_name_to_id = {name: i for i, name in enumerate(unique_labels)}
    net_id = torch.tensor([net_name_to_id[x] for x in raw_labels], dtype=torch.long)
    return net_id, net_name_to_id, raw_labels


def load_aal_roi_names_from_xlsx(aal_xlsx_path: str, n_nodes: int, col_idx: int = 2):
    """
    Default col_idx=2 means column C (0-based indexing).
    """
    aal_xlsx_path = Path(aal_xlsx_path)
    if not aal_xlsx_path.exists():
        raise FileNotFoundError(f"AAL xlsx not found: {aal_xlsx_path}")

    df = pd.read_excel(aal_xlsx_path, header=0)
    if df.shape[1] <= col_idx:
        raise ValueError(
            f"AAL.xlsx has only {df.shape[1]} columns, but need column index {col_idx}."
        )

    raw_series = df.iloc[:, col_idx].iloc[:n_nodes]
    if len(raw_series) < n_nodes:
        raise ValueError(f"AAL column only has {len(raw_series)} rows, but need {n_nodes}.")

    roi_names = []
    for i, v in enumerate(raw_series.tolist()):
        if pd.isna(v):
            roi_names.append(f"ROI_{i}")
        else:
            roi_names.append(str(v).strip())
    return roi_names


# =========================================================
# topo features
# =========================================================
@torch.no_grad()
def topo_features_rich(edge_index, edge_weight, net_id, n_nodes: int, device):
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if edge_weight is None:
        w = torch.ones(src.numel(), device=device)
    else:
        w = edge_weight.to(device).view(-1)

    deg = torch.zeros(n_nodes, device=device).scatter_add_(0, src, torch.ones_like(w))
    strength = torch.zeros(n_nodes, device=device).scatter_add_(0, src, w)

    same = (net_id[src] == net_id[dst]).float()
    within_w = torch.zeros(n_nodes, device=device).scatter_add_(0, src, w * same)
    cross_w = torch.zeros(n_nodes, device=device).scatter_add_(0, src, w * (1 - same))
    within_ratio = within_w / (within_w + cross_w + 1e-8)

    num_nets = int(net_id.max().item() + 1)
    kis = torch.zeros(n_nodes, num_nets, device=device)
    dst_net = net_id[dst]
    kis.index_put_((src, dst_net), w, accumulate=True)
    k_i = kis.sum(dim=1, keepdim=True) + 1e-8
    pcoef = 1.0 - ((kis / k_i) ** 2).sum(dim=1)

    a = torch.zeros(n_nodes, n_nodes, device=device)
    a[src, dst] = 1.0
    a = torch.maximum(a, a.t())

    aw = torch.zeros(n_nodes, n_nodes, device=device)
    aw[src, dst] = w
    aw = torch.maximum(aw, aw.t())

    tri = torch.diag(a @ a @ a) / 2.0
    denom = deg * (deg - 1.0) + 1e-8
    clustering = 2.0 * tri / denom

    neigh_deg_sum = (a * deg.view(1, -1)).sum(dim=1)
    avg_neigh_deg = neigh_deg_sum / (a.sum(dim=1) + 1e-8)

    a2 = (a @ a) > 0
    two_hop = a2.float().sum(dim=1)

    v = torch.ones(n_nodes, device=device) / (n_nodes ** 0.5)
    for _ in range(12):
        v = aw @ v
        v = v / (v.norm() + 1e-8)
    eigencent = v

    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    feats = torch.stack([
        zscore(deg),
        zscore(strength),
        zscore(pcoef),
        zscore(within_ratio),
        zscore(clustering),
        zscore(avg_neigh_deg),
        zscore(two_hop),
        zscore(eigencent),
    ], dim=-1)
    return feats


@torch.no_grad()
def build_node_identity(edge_index, edge_weight, base_net_id, n_nodes: int):
    device = edge_index.device
    net_id = base_net_id.to(device)

    num_nets = int(net_id.max().item() + 1)
    net_onehot = F.one_hot(net_id, num_classes=num_nets).float()
    topo_feats = topo_features_rich(
        edge_index=edge_index,
        edge_weight=edge_weight,
        net_id=net_id,
        n_nodes=n_nodes,
        device=device,
    )
    node_identity = torch.cat([net_onehot, topo_feats], dim=-1)
    return node_identity.cpu()


# =========================================================
# load LLM embeddings
# =========================================================
def load_llm_embeddings(dataset: str, llm_cache_dir: str, use_stage1: bool, use_stage2: bool):
    stage1 = None
    stage2 = None
    llm_dim = 0

    if use_stage1:
        stage1_path = Path(llm_cache_dir) / dataset / "stage1_emb_by_idx.pt"
        if not stage1_path.exists():
            raise FileNotFoundError(f"stage1 embedding not found: {stage1_path}")
        stage1 = torch.load(stage1_path, weights_only=False)
        if not isinstance(stage1, dict):
            raise TypeError(f"Expected dict at {stage1_path}, got {type(stage1)}")

    if use_stage2:
        stage2_path = Path(llm_cache_dir) / dataset / "stage2_emb_by_idx.pt"
        if not stage2_path.exists():
            raise FileNotFoundError(f"stage2 embedding not found: {stage2_path}")
        stage2 = torch.load(stage2_path, weights_only=False)
        if not isinstance(stage2, dict):
            raise TypeError(f"Expected dict at {stage2_path}, got {type(stage2)}")

    ref = stage1 if stage1 is not None else stage2
    if ref is not None:
        keys = sorted(ref.keys())
        if len(keys) == 0:
            raise ValueError("LLM embedding dict is empty.")
        llm_dim = len(ref[keys[0]])

    return stage1, stage2, llm_dim


# =========================================================
# dataset building
# =========================================================
def build_pyg_dataset(
    x_list,
    y_list,
    edge_index_list,
    edge_weight_list,
    use_identity: bool,
    base_net_id=None,
    use_llm_stage1: bool = False,
    stage1_emb=None,
    use_llm_stage2: bool = False,
    stage2_emb=None,
):
    all_data = []

    for idx, (x, y, edge_index, edge_weight) in enumerate(
        zip(x_list, y_list, edge_index_list, edge_weight_list)
    ):
        x_t = torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()

        edge_index_t = (
            edge_index.long()
            if isinstance(edge_index, torch.Tensor)
            else torch.tensor(edge_index, dtype=torch.long)
        )

        edge_attr_t = (
            edge_weight.float()
            if isinstance(edge_weight, torch.Tensor)
            else torch.tensor(edge_weight, dtype=torch.float)
        )

        data_kwargs = {
            "x": x_t,
            "edge_index": edge_index_t,
            "edge_attr": edge_attr_t,
            "y": torch.tensor([int(y)], dtype=torch.long),
        }

        if use_identity:
            if base_net_id is None:
                raise ValueError("base_net_id is required when use_identity=True")
            n_nodes = x_t.size(0)
            node_identity_t = build_node_identity(
                edge_index=edge_index_t,
                edge_weight=edge_attr_t,
                base_net_id=base_net_id,
                n_nodes=n_nodes,
            )
            data_kwargs["node_identity"] = node_identity_t

        if use_llm_stage1:
            if stage1_emb is None:
                raise ValueError("stage1_emb is required when use_llm_stage1=True")
            if idx not in stage1_emb:
                raise KeyError(f"Missing stage1 embedding for idx={idx}")
            data_kwargs["llm_stage1"] = torch.tensor(stage1_emb[idx], dtype=torch.float)

        if use_llm_stage2:
            if stage2_emb is None:
                raise ValueError("stage2_emb is required when use_llm_stage2=True")
            if idx not in stage2_emb:
                raise KeyError(f"Missing stage2 embedding for idx={idx}")
            data_kwargs["llm_stage2"] = torch.tensor(stage2_emb[idx], dtype=torch.float)

        all_data.append(Data(**data_kwargs))

    return all_data


def make_loader(all_data, indices, batch_size, shuffle):
    subset = [all_data[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def class_weight_from_indices(all_data, train_idx, device):
    ys = torch.tensor([int(all_data[i].y.item()) for i in train_idx], dtype=torch.long)
    counts = torch.bincount(ys, minlength=2).float()

    inv = counts.sum() / counts.clamp_min(1.0)
    w = torch.sqrt(inv)
    w = w / w.mean()
    w = w.to(device)

    return w, counts


# =========================================================
# training helpers
# =========================================================
def model_forward(model, data):
    return model(
        x=data.x,
        edge_index=data.edge_index,
        batch=data.batch,
        edge_weight=getattr(data, "edge_attr", None),
        node_identity=getattr(data, "node_identity", None),
        llm_stage1=getattr(data, "llm_stage1", None),
        llm_stage2=getattr(data, "llm_stage2", None),
    )


def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def soft_router_balance_loss(router_prob: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    mean_prob = router_prob.mean(dim=0)
    mean_prob = mean_prob.clamp_min(eps)
    mean_prob = mean_prob / mean_prob.sum().clamp_min(eps)
    entropy = -torch.sum(mean_prob * torch.log(mean_prob))
    return -entropy


def soft_graph_router_balance_loss(router_prob: torch.Tensor, batch: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if batch.numel() == 0:
        return torch.zeros((), device=router_prob.device, dtype=router_prob.dtype)

    bsz = int(batch.max().item()) + 1
    e = router_prob.size(1)

    g_sum = torch.zeros(bsz, e, device=router_prob.device, dtype=router_prob.dtype)
    g_sum.index_add_(0, batch, router_prob)
    counts = torch.bincount(batch, minlength=bsz).to(router_prob.device).clamp_min(1).unsqueeze(-1).float()
    g_graph = g_sum / counts

    g_graph = g_graph.clamp_min(eps)
    g_graph = g_graph / g_graph.sum(dim=-1, keepdim=True).clamp_min(eps)

    entropy = -torch.sum(g_graph * torch.log(g_graph), dim=-1)
    return -entropy.mean()


def expert_min_usage_loss(router_prob: torch.Tensor, min_usage: float = 0.08) -> torch.Tensor:
    mean_prob = router_prob.mean(dim=0)
    deficit = F.relu(min_usage - mean_prob)
    return deficit.mean()


def router_importance_loss(router_prob: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    imp = router_prob.mean(dim=0)
    imp = imp / imp.sum().clamp_min(eps)
    target = torch.full_like(imp, 1.0 / imp.numel())
    return torch.sum((imp - target) ** 2)


def router_load_loss(gates: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    load = (gates > 0).float().mean(dim=0)
    load = load / load.sum().clamp_min(eps)
    target = torch.full_like(load, 1.0 / load.numel())
    return torch.sum((load - target) ** 2)


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    return (router_logits ** 2).mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    class_weight=None,
    lambda_prior=0.0,
    lambda_soft_graph_balance=0.0,
    lambda_soft_expert_balance=0.0,
    lambda_min_usage=0.0,
    lambda_importance=0.0,
    lambda_load=0.0,
    lambda_z=0.0,
    min_usage=0.08,
):
    model.train()
    ce = nn.CrossEntropyLoss(weight=class_weight if class_weight is not None else None)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    total_ce = 0.0
    total_kl = 0.0
    total_sgbal = 0.0
    total_sebal = 0.0
    total_minu = 0.0
    total_imp = 0.0
    total_load = 0.0
    total_z = 0.0

    for data in loader:
        data = data.to(device)
        y = data.y.view(-1).long()

        optimizer.zero_grad()
        logits, gates, aux = model_forward(model, data)

        loss_ce = ce(logits, y)

        loss_kl = torch.zeros((), device=device)
        if "p_net" in aux and "p_router" in aux:
            loss_kl = sym_kl(aux["p_net"], aux["p_router"])

        router_prob = aux["router_prob"]
        router_logits = aux["router_logits"]

        loss_sgbal = torch.zeros((), device=device)
        if lambda_soft_graph_balance > 0:
            loss_sgbal = soft_graph_router_balance_loss(router_prob, data.batch)

        loss_sebal = torch.zeros((), device=device)
        if lambda_soft_expert_balance > 0:
            loss_sebal = soft_router_balance_loss(router_prob)

        loss_minu = torch.zeros((), device=device)
        if lambda_min_usage > 0:
            loss_minu = expert_min_usage_loss(router_prob, min_usage=min_usage)

        loss_imp = torch.zeros((), device=device)
        if lambda_importance > 0:
            loss_imp = router_importance_loss(router_prob)

        loss_load = torch.zeros((), device=device)
        if lambda_load > 0:
            loss_load = router_load_loss(gates)

        loss_z = torch.zeros((), device=device)
        if lambda_z > 0:
            loss_z = router_z_loss(router_logits)

        loss = (
            loss_ce
            + lambda_prior * loss_kl
            + lambda_soft_graph_balance * loss_sgbal
            + lambda_soft_expert_balance * loss_sebal
            + lambda_min_usage * loss_minu
            + lambda_importance * loss_imp
            + lambda_load * loss_load
            + lambda_z * loss_z
        )

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_kl += loss_kl.item() * bs
        total_sgbal += loss_sgbal.item() * bs
        total_sebal += loss_sebal.item() * bs
        total_minu += loss_minu.item() * bs
        total_imp += loss_imp.item() * bs
        total_load += loss_load.item() * bs
        total_z += loss_z.item() * bs

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += bs

    avg = max(total_examples, 1)
    return (
        total_loss / avg,
        total_correct / avg,
        {
            "loss_ce": total_ce / avg,
            "loss_kl": total_kl / avg,
            "loss_soft_graph_balance": total_sgbal / avg,
            "loss_soft_expert_balance": total_sebal / avg,
            "loss_min_usage": total_minu / avg,
            "loss_importance": total_imp / avg,
            "loss_load": total_load / avg,
            "loss_z": total_z / avg,
        },
    )


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    ys, probs = [], []

    for data in loader:
        data = data.to(device)
        y = data.y.view(-1).long()

        logits, _, _ = model_forward(model, data)
        p1 = torch.softmax(logits, dim=1)[:, 1]

        ys.append(y.detach().cpu())
        probs.append(p1.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    return y_true, y_prob


@torch.no_grad()
def collect_gate_usage(model, loader, device):
    model.eval()

    gate_sum = None
    select_sum = None
    total_nodes = 0

    graph_entropy_sum = 0.0
    total_graphs = 0

    router_prob_sum = None

    for data in loader:
        data = data.to(device)
        _, gates, aux = model_forward(model, data)

        if gate_sum is None:
            gate_sum = gates.sum(dim=0).detach().cpu()
            select_sum = (gates > 0).float().sum(dim=0).detach().cpu()
            router_prob_sum = aux["router_prob"].sum(dim=0).detach().cpu()
        else:
            gate_sum += gates.sum(dim=0).detach().cpu()
            select_sum += (gates > 0).float().sum(dim=0).detach().cpu()
            router_prob_sum += aux["router_prob"].sum(dim=0).detach().cpu()

        total_nodes += gates.size(0)

        bsz = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 0
        if bsz > 0:
            g_sum = torch.zeros(bsz, gates.size(1), device=gates.device, dtype=gates.dtype)
            g_sum.index_add_(0, data.batch, gates)
            counts = torch.bincount(data.batch, minlength=bsz).to(gates.device).clamp_min(1).unsqueeze(-1).float()
            g_graph = g_sum / counts
            g_graph = g_graph / g_graph.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            graph_entropy = -torch.sum(g_graph.clamp_min(1e-9) * torch.log(g_graph.clamp_min(1e-9)), dim=-1)
            graph_entropy_sum += graph_entropy.sum().item()
            total_graphs += bsz

    if gate_sum is None or total_nodes == 0:
        return None

    mean_gate = (gate_sum / total_nodes).numpy().tolist()
    select_rate = (select_sum / total_nodes).numpy().tolist()
    mean_router_prob = (router_prob_sum / total_nodes).numpy().tolist()
    avg_graph_entropy = graph_entropy_sum / max(total_graphs, 1)

    return {
        "mean_gate_weight": mean_gate,
        "selection_rate": select_rate,
        "mean_router_prob": mean_router_prob,
        "avg_graph_gate_entropy": float(avg_graph_entropy),
    }


@torch.no_grad()
def collect_node_expert_attribution(model, loader, device, n_nodes: int, label_filter=None):
    model.eval()

    router_prob_sum = None
    gate_sum = None
    select_sum = None
    graph_count = 0

    for data in loader:
        data = data.to(device)
        logits, gates, aux = model_forward(model, data)

        graph_y = data.y.view(-1).long()
        batch_vec = data.batch
        router_prob = aux["router_prob"]

        num_graphs_in_batch = graph_y.size(0)

        for g in range(num_graphs_in_batch):
            if label_filter is not None and int(graph_y[g].item()) != int(label_filter):
                continue

            node_mask = (batch_vec == g)
            rp_g = router_prob[node_mask]
            gate_g = gates[node_mask]

            if rp_g.size(0) != n_nodes:
                raise ValueError(
                    f"Graph has {rp_g.size(0)} nodes, expected {n_nodes}. "
                    "Node-expert attribution assumes fixed node count/order."
                )

            if router_prob_sum is None:
                num_experts = rp_g.size(1)
                router_prob_sum = torch.zeros(n_nodes, num_experts, dtype=torch.float64)
                gate_sum = torch.zeros(n_nodes, num_experts, dtype=torch.float64)
                select_sum = torch.zeros(n_nodes, num_experts, dtype=torch.float64)

            router_prob_sum += rp_g.detach().cpu().double()
            gate_sum += gate_g.detach().cpu().double()
            select_sum += (gate_g > 0).float().detach().cpu().double()
            graph_count += 1

    if graph_count == 0:
        return None

    mean_router_prob = (router_prob_sum / graph_count).numpy()
    mean_gate_weight = (gate_sum / graph_count).numpy()
    selection_rate = (select_sum / graph_count).numpy()

    return {
        "mean_router_prob": mean_router_prob,
        "mean_gate_weight": mean_gate_weight,
        "selection_rate": selection_rate,
        "num_graphs": int(graph_count),
    }


@torch.no_grad()
def collect_subject_expert_usage(model, loader, device):
    model.eval()

    usage_soft_list = []
    usage_hard_list = []
    label_list = []
    prob_list = []
    pred_list = []

    for data in loader:
        data = data.to(device)

        logits, gates, aux = model_forward(model, data)
        y = data.y.view(-1).long()

        router_prob = aux["router_prob"]   # [N, E]
        batch_vec = data.batch             # [N]

        bsz = y.size(0)
        num_experts = router_prob.size(1)

        # soft subject-level expert usage
        usage_soft = torch.zeros(
            bsz, num_experts, device=router_prob.device, dtype=router_prob.dtype
        )
        usage_soft.index_add_(0, batch_vec, router_prob)
        counts = torch.bincount(batch_vec, minlength=bsz).to(router_prob.device).clamp_min(1).unsqueeze(-1).float()
        usage_soft = usage_soft / counts
        usage_soft = usage_soft / usage_soft.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # hard subject-level expert usage
        usage_hard = torch.zeros(
            bsz, num_experts, device=gates.device, dtype=gates.dtype
        )
        usage_hard.index_add_(0, batch_vec, gates)
        usage_hard = usage_hard / counts
        usage_hard = usage_hard / usage_hard.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # prediction
        p1 = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        usage_soft_list.append(usage_soft.detach().cpu())
        usage_hard_list.append(usage_hard.detach().cpu())
        label_list.append(y.detach().cpu())
        prob_list.append(p1.detach().cpu())
        pred_list.append(pred.detach().cpu())

    if len(usage_soft_list) == 0:
        return None

    return {
        "usage_soft": torch.cat(usage_soft_list, dim=0),   # [num_subjects, 4]
        "usage_hard": torch.cat(usage_hard_list, dim=0),   # [num_subjects, 4]
        "labels": torch.cat(label_list, dim=0),            # [num_subjects]
        "probs": torch.cat(prob_list, dim=0),              # [num_subjects]
        "preds": torch.cat(pred_list, dim=0),              # [num_subjects]
    }


@torch.no_grad()
def collect_subject_node_expert_usage(model, loader, device, n_nodes: int):
    model.eval()

    soft_list = []
    hard_list = []
    label_list = []
    prob_list = []
    pred_list = []

    for data in loader:
        data = data.to(device)

        logits, gates, aux = model_forward(model, data)
        y = data.y.view(-1).long()
        p1 = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        router_prob = aux["router_prob"]   # [N_total, E]
        batch_vec = data.batch             # [N_total]
        bsz = y.size(0)
        num_experts = router_prob.size(1)

        soft_batch = torch.zeros(bsz, n_nodes, num_experts, device=device, dtype=router_prob.dtype)
        hard_batch = torch.zeros(bsz, n_nodes, num_experts, device=device, dtype=gates.dtype)

        for g in range(bsz):
            node_mask = (batch_vec == g)
            rp_g = router_prob[node_mask]   # [n_nodes, E]
            gt_g = gates[node_mask]         # [n_nodes, E]

            if rp_g.size(0) != n_nodes:
                raise ValueError(
                    f"Graph has {rp_g.size(0)} nodes, expected {n_nodes}. "
                    "This function assumes fixed node order/count across subjects."
                )

            soft_batch[g] = rp_g
            hard_batch[g] = gt_g

        soft_list.append(soft_batch.detach().cpu())
        hard_list.append(hard_batch.detach().cpu())
        label_list.append(y.detach().cpu())
        prob_list.append(p1.detach().cpu())
        pred_list.append(pred.detach().cpu())

    if len(soft_list) == 0:
        return None

    return {
        "soft_node_usage": torch.cat(soft_list, dim=0),   # [num_subjects, n_nodes, E]
        "hard_node_usage": torch.cat(hard_list, dim=0),   # [num_subjects, n_nodes, E]
        "labels": torch.cat(label_list, dim=0),           # [num_subjects]
        "probs": torch.cat(prob_list, dim=0),             # [num_subjects]
        "preds": torch.cat(pred_list, dim=0),             # [num_subjects]
    }

def save_node_expert_attribution_csv(attr_dict, roi_names, save_path_prefix, expert_names=None):
    if attr_dict is None:
        return

    if expert_names is None:
        expert_names = ["mlp", "cheb", "gt", "gcn"]

    n_nodes = len(roi_names)

    for key in ["mean_router_prob", "mean_gate_weight", "selection_rate"]:
        arr = attr_dict[key]
        df = pd.DataFrame(arr, columns=expert_names)
        df.insert(0, "roi_idx", np.arange(n_nodes))
        df.insert(1, "roi_name", roi_names)
        df["top_expert"] = [expert_names[i] for i in arr.argmax(axis=1)]
        out_path = f"{save_path_prefix}_{key}.csv"
        df.to_csv(out_path, index=False)

    meta = {
        "num_graphs": attr_dict["num_graphs"],
        "expert_names": expert_names,
    }
    with open(f"{save_path_prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

@torch.no_grad()
def collect_probs_with_feature_noise(model, loader, device, noise_std: float = 0.0, noise_repeat: int = 1):
    model.eval()
    ys, probs = [], []

    for data in loader:
        data = data.to(device)
        y = data.y.view(-1).long()

        prob_accum = []

        for _ in range(noise_repeat):
            data_noisy = data.clone()

            if noise_std > 0:
                noise = torch.randn_like(data_noisy.x) * noise_std
                data_noisy.x = data_noisy.x + noise

            logits, _, _ = model_forward(model, data_noisy)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            prob_accum.append(p1.detach().cpu())

        prob_accum = torch.stack(prob_accum, dim=0).mean(dim=0)

        ys.append(y.detach().cpu())
        probs.append(prob_accum)

    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    return y_true, y_prob

# =========================================================
# metrics
# =========================================================
def accuracy_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return accuracy_score(y_true, y_pred)


def balanced_acc_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def f1_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return f1_score(y_true, y_pred, zero_division=0)


def precision_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def recall_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return recall_score(y_true, y_pred, zero_division=0)


def metric_at_threshold(y_true, y_prob, thr: float, metric_name: str):
    if metric_name == "accuracy":
        return accuracy_at_threshold(y_true, y_prob, thr)
    elif metric_name == "balanced_acc":
        return balanced_acc_at_threshold(y_true, y_prob, thr)
    elif metric_name == "f1":
        return f1_at_threshold(y_true, y_prob, thr)
    elif metric_name == "precision":
        return precision_at_threshold(y_true, y_prob, thr)
    elif metric_name == "recall":
        return recall_at_threshold(y_true, y_prob, thr)
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")


def select_threshold_by_metric(
    y_true,
    y_prob,
    metric_name="accuracy",
    grid=None,
    thr_min=0.35,
    thr_max=0.65,
    thr_steps=61,
):
    if grid is None:
        grid = np.linspace(thr_min, thr_max, thr_steps)

    best_thr, best_score = 0.5, -1.0
    for thr in grid:
        score = metric_at_threshold(y_true, y_prob, float(thr), metric_name)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def evaluate_metric(y_true, y_prob, metric_name: str, thr: float = 0.5):
    if metric_name == "roc_auc":
        if len(np.unique(y_true)) > 1:
            return float(roc_auc_score(y_true, y_prob))
        return 0.0
    return float(metric_at_threshold(y_true, y_prob, thr, metric_name))


def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal = balanced_acc_at_threshold(y_true, y_prob, thr)

    auc = 0.0
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "balanced_acc": float(bal),
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "threshold": float(thr),
    }


# =========================================================
# router schedule
# =========================================================
def set_router_schedule(model, epoch: int, final_top_k: int):
    """
    More ACC-oriented routing schedule:
      epoch  1-6  : topk=4, temp=2.5
      epoch  7-15 : topk=3, temp=1.8
      epoch >=16  : topk=final_top_k, temp=1.2
    """
    if epoch <= 6:
        model.current_top_k = model.num_experts
        model.router_temperature = 2.5
    elif epoch <= 15:
        model.current_top_k = min(3, model.num_experts)
        model.router_temperature = 1.8
    else:
        model.current_top_k = min(final_top_k, model.num_experts)
        model.router_temperature = 1.2


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("BrainMoE main")

    parser.add_argument("--dataset", type=str, default="ADHD", help="ADHD or ABIDE")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--llm_hidden_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_prior", type=float, default=0.0)
    parser.add_argument("--lambda_soft_graph_balance", type=float, default=0.001)
    parser.add_argument("--lambda_soft_expert_balance", type=float, default=0.001)
    parser.add_argument("--lambda_min_usage", type=float, default=0.0)
    parser.add_argument("--min_usage", type=float, default=0.08)

    parser.add_argument("--router_noise_std", type=float, default=0.02)
    parser.add_argument("--router_temperature", type=float, default=2.5)
    parser.add_argument("--router_warmup_epochs", type=int, default=16)
    parser.add_argument("--stage1_scale_init", type=float, default=0.35)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--lambda_importance", type=float, default=0.002)
    parser.add_argument("--lambda_load", type=float, default=0.002)
    parser.add_argument("--lambda_z", type=float, default=1e-4)

    parser.add_argument(
        "--selection_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "balanced_acc", "f1", "precision", "recall", "roc_auc"],
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "balanced_acc", "f1", "precision", "recall"],
    )
    parser.add_argument("--thr_min", type=float, default=0.35)
    parser.add_argument("--thr_max", type=float, default=0.65)
    parser.add_argument("--thr_steps", type=int, default=61)

    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=8)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=18)
    parser.add_argument("--early_stop_use_moving_avg", action="store_true")
    parser.add_argument("--moving_avg_window", type=int, default=3)

    parser.add_argument("--use_identity", action="store_true")
    parser.add_argument("--use_llm_stage1", action="store_true")
    parser.add_argument("--use_llm_stage2", action="store_true")
    parser.add_argument("--use_neuro_bias", action="store_true")
    parser.add_argument("--neuro_bias_scale_init", type=float, default=0.3)

    parser.add_argument(
        "--aal_xlsx",
        type=str,
        default="/home/xinyangzhao/Mujie/BrainMoe/isdt/AAL.xlsx",
        help="AAL.xlsx path; E column used as network labels",
    )
    parser.add_argument(
        "--llm_cache_dir",
        type=str,
        default="/home/xinyangzhao/Mujie/BrainMoe/data/abnormal_llm_cache",
        help="Contains <dataset>/stage1_emb_by_idx.pt and stage2_emb_by_idx.pt",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_brainmoe_identity",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)
    print(
        f"[ARGS] dataset={args.dataset} seed={args.seed} "
        f"use_identity={args.use_identity} "
        f"use_llm_stage1={args.use_llm_stage1} "
        f"use_llm_stage2={args.use_llm_stage2} "
        f"use_class_weight={args.use_class_weight} "
        f"selection_metric={args.selection_metric} "
        f"threshold_metric={args.threshold_metric}"
    )

    # -----------------------------------------------------
    # load cached subjects
    # -----------------------------------------------------
    cache_path = DATASET_PATH / f"preprocessed_subjects/{args.dataset}_cached.pt"
    if cache_path.exists():
        print(f"[DATA] Loading cached dataset from: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        x_list = data["x_list"]
        y_list = data["y_list"]
        edge_index_list = data["edge_index_list"]
        edge_weight_list = data["edge_weight_list"]
    else:
        print(f"[DATA] Cache not found. Building from raw get_subjects({args.dataset}) ...")
        (
            id_list,
            x_list,
            y_list,
            sex_list,
            age_list,
            handedness_list,
            viq_list,
            piq_list,
            fiq_list,
            edge_index_list,
            edge_weight_list,
            timeseries_list,
            ts_len_list,
            subjects_json_list,
        ) = get_subjects(args.dataset)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "id_list": id_list,
                "x_list": x_list,
                "y_list": y_list,
                "sex_list": sex_list,
                "age_list": age_list,
                "handedness_list": handedness_list,
                "viq_list": viq_list,
                "piq_list": piq_list,
                "fiq_list": fiq_list,
                "edge_index_list": edge_index_list,
                "edge_weight_list": edge_weight_list,
                "timeseries_list": timeseries_list,
                "ts_len_list": ts_len_list,
                "subjects_json_list": subjects_json_list,
            },
            cache_path,
        )
        print(f"[DATA] Saved cache to: {cache_path}")

    first_x = x_list[0] if isinstance(x_list[0], torch.Tensor) else torch.tensor(x_list[0])
    n_nodes = first_x.size(0)
    num_subjects = len(y_list)
    print(f"[DATA] subjects={num_subjects} | num_nodes={n_nodes}")

    labels = np.array([int(y) for y in y_list], dtype=int)
    majority_acc = max((labels == 0).mean(), (labels == 1).mean())
    print(f"[BASELINE] majority_class_acc={majority_acc:.4f}")

    # -----------------------------------------------------
    # identity
    # -----------------------------------------------------
    base_net_id = None
    net_name_to_id = {}
    num_nets = 0

    if args.use_identity:
        base_net_id, net_name_to_id, _ = load_aal_net_ids_from_xlsx(args.aal_xlsx, n_nodes)
        num_nets = int(base_net_id.max().item() + 1)
        print(f"[AAL] loaded {num_nets} network labels from E column")
        print(f"[AAL] network mapping: {net_name_to_id}")

    roi_names = load_aal_roi_names_from_xlsx(args.aal_xlsx, n_nodes, col_idx=2)

    # -----------------------------------------------------
    # llm embeddings
    # -----------------------------------------------------
    stage1_emb, stage2_emb, llm_dim = load_llm_embeddings(
        dataset=args.dataset,
        llm_cache_dir=args.llm_cache_dir,
        use_stage1=args.use_llm_stage1,
        use_stage2=args.use_llm_stage2,
    )
    if args.use_llm_stage1 or args.use_llm_stage2:
        print(f"[LLM] loaded embeddings | dim={llm_dim}")

    # -----------------------------------------------------
    # build dataset
    # -----------------------------------------------------
    all_data = build_pyg_dataset(
        x_list=x_list,
        y_list=y_list,
        edge_index_list=edge_index_list,
        edge_weight_list=edge_weight_list,
        use_identity=args.use_identity,
        base_net_id=base_net_id,
        use_llm_stage1=args.use_llm_stage1,
        stage1_emb=stage1_emb,
        use_llm_stage2=args.use_llm_stage2,
        stage2_emb=stage2_emb,
    )

    sample = all_data[0]
    in_dim = sample.x.size(1)
    identity_dim = sample.node_identity.size(1) if hasattr(sample, "node_identity") else 0
    print(f"[DIMS] in_dim={in_dim} | identity_dim={identity_dim} | llm_dim={llm_dim}")

    # -----------------------------------------------------
    # cv
    # -----------------------------------------------------
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_tag = "BrainMoE"
    model_tag += "_node_identity" if args.use_identity else "_no_identity"
    model_tag += "_with_stage1" if args.use_llm_stage1 else ""
    model_tag += "_with_stage2" if args.use_llm_stage2 else ""
    model_tag += f"_sel-{args.selection_metric}_thr-{args.threshold_metric}"

    fold_metrics = []
    fold_gate_usages = []
    fold_robustness = []
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        y_trainval = labels[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=y_trainval,
        )

        print("\n" + "=" * 90)
        print(f"[FOLD {fold}/{args.n_splits}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        train_loader = make_loader(all_data, train_idx, args.batch_size, shuffle=True)
        val_loader = make_loader(all_data, val_idx, args.batch_size, shuffle=False)
        test_loader = make_loader(all_data, test_idx, args.batch_size, shuffle=False)

        cw, cc = class_weight_from_indices(all_data, train_idx, device)
        print(f"[CLASS] counts={cc.tolist()} | weights={cw.tolist()}")

        model = BrainMoE(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            identity_dim=identity_dim,
            llm_dim=llm_dim if (args.use_llm_stage1 or args.use_llm_stage2) else 3072,
            llm_hidden_dim=args.llm_hidden_dim,
            top_k=args.top_k,
            dropout=args.dropout,
            use_identity=args.use_identity,
            use_llm_stage1=args.use_llm_stage1,
            use_llm_stage2=args.use_llm_stage2,
            router_noise_std=args.router_noise_std,
            router_temperature=args.router_temperature,
            stage1_scale_init=args.stage1_scale_init,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )

        best_state = None
        best_val_score = -1.0
        best_val_thr = 0.5
        best_epoch = 0
        epochs_no_improve = 0

        recent_scores = deque(maxlen=max(1, args.moving_avg_window))

        for epoch in range(1, args.num_epochs + 1):
            set_router_schedule(model, epoch, args.top_k)

            cw_to_use = cw if args.use_class_weight else None
            tr_loss, tr_acc, tr_loss_dict = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                class_weight=cw_to_use,
                lambda_prior=args.lambda_prior,
                lambda_soft_graph_balance=args.lambda_soft_graph_balance,
                lambda_soft_expert_balance=args.lambda_soft_expert_balance,
                lambda_min_usage=args.lambda_min_usage,
                lambda_importance=args.lambda_importance,
                lambda_load=args.lambda_load,
                lambda_z=args.lambda_z,
                min_usage=args.min_usage,
            )

            yv, pv = collect_probs(model, val_loader, device)
            thr, _ = select_threshold_by_metric(
                yv,
                pv,
                metric_name=args.threshold_metric,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                thr_steps=args.thr_steps,
            )
            val_metrics = compute_metrics(yv, pv, thr)
            val_select_score = evaluate_metric(
                y_true=yv,
                y_prob=pv,
                metric_name=args.selection_metric,
                thr=thr,
            )

            recent_scores.append(val_select_score)
            if args.early_stop_use_moving_avg:
                score_for_early_stop = float(np.mean(recent_scores))
            else:
                score_for_early_stop = val_select_score

            scheduler_metric = val_metrics["roc_auc"]
            scheduler.step(scheduler_metric)

            if score_for_early_stop > best_val_score:
                best_val_score = score_for_early_stop
                best_val_thr = thr
                best_epoch = epoch
                best_state = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "top_k": model.current_top_k,
                    "temperature": model.router_temperature,
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            current_lr = optimizer.param_groups[0]["lr"]
            if args.selection_metric == "roc_auc":
                select_msg = f"val_roc_auc={val_select_score*100:.2f}%"
            else:
                select_msg = f"val_{args.selection_metric}={val_select_score*100:.2f}%"

            print(
                f"[Fold {fold}] epoch={epoch:03d} "
                f"topk={model.current_top_k} "
                f"temp={model.router_temperature:.2f} "
                f"lr={current_lr:.6g} "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% "
                f"{select_msg} "
                f"thr={thr:.2f} "
                f"(ce={tr_loss_dict['loss_ce']:.4f}, "
                f"kl={tr_loss_dict['loss_kl']:.4f}, "
                f"sgbal={tr_loss_dict['loss_soft_graph_balance']:.4f}, "
                f"sebal={tr_loss_dict['loss_soft_expert_balance']:.4f}, "
                f"minu={tr_loss_dict['loss_min_usage']:.4f}, "
                f"imp={tr_loss_dict['loss_importance']:.4f}, "
                f"load={tr_loss_dict['loss_load']:.4f}, "
                f"z={tr_loss_dict['loss_z']:.6f})"
            )

            if epochs_no_improve >= args.early_stop_patience:
                print(f"[Fold {fold}] Early stopping at epoch={epoch}")
                break

        print(
            f"[Fold {fold}] best_epoch={best_epoch} "
            f"best_val_{args.selection_metric}={best_val_score*100:.2f}% "
            f"best_thr={best_val_thr:.2f}"
        )

        if best_state is not None:
            model.load_state_dict(best_state["model"])
            model.current_top_k = best_state["top_k"]
            model.router_temperature = best_state["temperature"]

        yt, pt = collect_probs(model, test_loader, device)
        test_m = compute_metrics(yt, pt, best_val_thr)
        gate_usage = collect_gate_usage(model, test_loader, device)

        # -------------------------------------------------
        # robustness: feature noise vs AUC
        # -------------------------------------------------
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        noise_repeat = 5   # 每个noise重复几次取平均，减少随机波动

        robustness_rows = []
        for noise_std in noise_levels:
            yn, pn = collect_probs_with_feature_noise(
                model=model,
                loader=test_loader,
                device=device,
                noise_std=noise_std,
                noise_repeat=noise_repeat,
            )

            auc_n = 0.0
            if len(np.unique(yn)) > 1:
                auc_n = roc_auc_score(yn, pn)

            robustness_rows.append({
                "fold": fold,
                "noise_std": float(noise_std),
                "auc": float(auc_n),
            })

        df_robust = pd.DataFrame(robustness_rows)
        df_robust.to_csv(
            results_dir / f"{args.dataset}_{model_tag}_fold{fold}_feature_noise_robustness.csv",
            index=False,
        )

        # -------------------------------------------------
        # subject-level node expert usage for network violin
        # -------------------------------------------------
        subject_node_usage = collect_subject_node_expert_usage(
            model=model,
            loader=test_loader,
            device=device,
            n_nodes=n_nodes,
        )

        if subject_node_usage is not None:
            torch.save(
                {
                    "soft_node_usage": subject_node_usage["soft_node_usage"],   # [N_test, n_nodes, 4]
                    "hard_node_usage": subject_node_usage["hard_node_usage"],   # [N_test, n_nodes, 4]
                    "labels": subject_node_usage["labels"],                     # [N_test]
                    "probs": subject_node_usage["probs"],                       # [N_test]
                    "preds": subject_node_usage["preds"],                       # [N_test]
                    "roi_names": roi_names,
                    "expert_names": ["mlp", "cheb", "gt", "gcn"],
                    "fold": fold,
                    "dataset": args.dataset,
                    "model_tag": model_tag,
                    "threshold": best_val_thr,
                },
                results_dir / f"{args.dataset}_{model_tag}_fold{fold}_subject_node_expert_usage.pt",
            )

        subject_usage = collect_subject_expert_usage(model, test_loader, device)

        if subject_usage is not None:
            torch.save(
                {
                    "usage_soft": subject_usage["usage_soft"],   # [N_test, 4]
                    "usage_hard": subject_usage["usage_hard"],   # [N_test, 4]
                    "labels": subject_usage["labels"],           # [N_test]
                    "probs": subject_usage["probs"],             # [N_test]
                    "preds": subject_usage["preds"],             # [N_test]
                    "expert_names": ["mlp", "cheb", "gt", "gcn"],
                    "fold": fold,
                    "dataset": args.dataset,
                    "model_tag": model_tag,
                    "threshold": best_val_thr,
                },
                results_dir / f"{args.dataset}_{model_tag}_fold{fold}_subject_expert_usage.pt",
            )

        if subject_usage is not None:
            usage_soft_np = subject_usage["usage_soft"].numpy()
            usage_hard_np = subject_usage["usage_hard"].numpy()
            labels_np = subject_usage["labels"].numpy()
            probs_np = subject_usage["probs"].numpy()
            preds_np = subject_usage["preds"].numpy()

            df_usage = pd.DataFrame({
                "subject_idx_in_fold": np.arange(len(labels_np)),
                "label": labels_np,
                "prob_1": probs_np,
                "pred": preds_np,
                "soft_mlp": usage_soft_np[:, 0],
                "soft_cheb": usage_soft_np[:, 1],
                "soft_gt": usage_soft_np[:, 2],
                "soft_gcn": usage_soft_np[:, 3],
                "hard_mlp": usage_hard_np[:, 0],
                "hard_cheb": usage_hard_np[:, 1],
                "hard_gt": usage_hard_np[:, 2],
                "hard_gcn": usage_hard_np[:, 3],
            })
            df_usage.to_csv(
                results_dir / f"{args.dataset}_{model_tag}_fold{fold}_subject_expert_usage.csv",
                index=False,
            )

        expert_names = ["mlp", "cheb", "gt", "gcn"]
        gate_usage_named = {}
        if gate_usage is not None:
            gate_usage_named = {
                "mean_gate_weight": {
                    expert_names[i]: float(gate_usage["mean_gate_weight"][i]) for i in range(len(expert_names))
                },
                "selection_rate": {
                    expert_names[i]: float(gate_usage["selection_rate"][i]) for i in range(len(expert_names))
                },
                "mean_router_prob": {
                    expert_names[i]: float(gate_usage["mean_router_prob"][i]) for i in range(len(expert_names))
                },
                "avg_graph_gate_entropy": float(gate_usage["avg_graph_gate_entropy"]),
            }

        print(
            f"[TEST][fold {fold}] "
            f"ACC={test_m['accuracy']*100:.2f}% | "
            f"BAL_ACC={test_m['balanced_acc']*100:.2f}% | "
            f"AUC={test_m['roc_auc']*100:.2f}% | "
            f"F1={test_m['f1_score']*100:.2f}% | "
            f"PRE={test_m['precision']*100:.2f}% | "
            f"REC={test_m['recall']*100:.2f}%"
        )
        if gate_usage_named:
            print(f"[GATE][fold {fold}] mean_gate_weight={gate_usage_named['mean_gate_weight']}")
            print(f"[GATE][fold {fold}] selection_rate={gate_usage_named['selection_rate']}")
            print(f"[GATE][fold {fold}] mean_router_prob={gate_usage_named['mean_router_prob']}")
            print(f"[GATE][fold {fold}] avg_graph_gate_entropy={gate_usage_named['avg_graph_gate_entropy']:.4f}")

        # -------------------------------------------------
        # node-expert attribution maps
        # -------------------------------------------------
        expert_names = ["mlp", "cheb", "gt", "gcn"]

        attr_all = collect_node_expert_attribution(
            model=model,
            loader=test_loader,
            device=device,
            n_nodes=n_nodes,
            label_filter=None,
        )
        save_node_expert_attribution_csv(
            attr_dict=attr_all,
            roi_names=roi_names,
            save_path_prefix=str(results_dir / f"{args.dataset}_{model_tag}_fold{fold}_all"),
            expert_names=expert_names,
        )

        attr_c0 = collect_node_expert_attribution(
            model=model,
            loader=test_loader,
            device=device,
            n_nodes=n_nodes,
            label_filter=0,
        )
        save_node_expert_attribution_csv(
            attr_dict=attr_c0,
            roi_names=roi_names,
            save_path_prefix=str(results_dir / f"{args.dataset}_{model_tag}_fold{fold}_class0"),
            expert_names=expert_names,
        )

        attr_c1 = collect_node_expert_attribution(
            model=model,
            loader=test_loader,
            device=device,
            n_nodes=n_nodes,
            label_filter=1,
        )
        save_node_expert_attribution_csv(
            attr_dict=attr_c1,
            roi_names=roi_names,
            save_path_prefix=str(results_dir / f"{args.dataset}_{model_tag}_fold{fold}_class1"),
            expert_names=expert_names,
        )

        fold_record = {
            "fold": fold,
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "threshold_metric": args.threshold_metric,
            f"best_val_{args.selection_metric}": best_val_score,
            "best_threshold": best_val_thr,
            "test_metrics": test_m,
            "gate_usage": gate_usage_named,
        }

        # -----------------------------------------------------
        # robustness summary
        # -----------------------------------------------------
        if len(fold_robustness) > 0:
            df_robust_all = pd.concat(fold_robustness, axis=0, ignore_index=True)
            robust_summary = (
                df_robust_all.groupby("noise_std")["auc"]
                .agg(["mean", "std"])
                .reset_index()
            )
            robust_summary.to_csv(
                results_dir / f"{args.dataset}_{model_tag}_feature_noise_robustness_5fold_summary.csv",
                index=False,
            )
        else:
            robust_summary = None

        save_json(
            fold_record,
            results_dir / f"{args.dataset}_{model_tag}_fold{fold}.json",
        )

        fold_metrics.append(test_m)
        fold_gate_usages.append(gate_usage_named)
        fold_robustness.append(df_robust.copy())

    # -----------------------------------------------------
    # summary
    # -----------------------------------------------------
    bal_m, bal_s = mean_std(fold_metrics, "balanced_acc")
    acc_m, acc_s = mean_std(fold_metrics, "accuracy")
    pre_m, pre_s = mean_std(fold_metrics, "precision")
    rec_m, rec_s = mean_std(fold_metrics, "recall")
    f1_m, f1_s = mean_std(fold_metrics, "f1_score")
    auc_m, auc_s = mean_std(fold_metrics, "roc_auc")

    expert_names = ["mlp", "cheb", "gt", "gcn"]

    gate_summary = {
        "mean_gate_weight": {},
        "selection_rate": {},
        "mean_router_prob": {},
        "avg_graph_gate_entropy": {"mean": 0.0, "std": 0.0},
    }

    for key_name in ["mean_gate_weight", "selection_rate", "mean_router_prob"]:
        for name in expert_names:
            vals = [
                g[key_name][name]
                for g in fold_gate_usages
                if g and key_name in g and name in g[key_name]
            ]
            if len(vals) > 0:
                gate_summary[key_name][name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }

    entropy_vals = [
        g["avg_graph_gate_entropy"]
        for g in fold_gate_usages
        if g and "avg_graph_gate_entropy" in g
    ]
    if len(entropy_vals) > 0:
        gate_summary["avg_graph_gate_entropy"] = {
            "mean": float(np.mean(entropy_vals)),
            "std": float(np.std(entropy_vals, ddof=1)) if len(entropy_vals) > 1 else 0.0,
        }

    print("\n" + "=" * 90)
    print(f"[FINAL {args.n_splits}-FOLD SUMMARY] {args.dataset} | {model_tag}")
    print(f"balanced_acc : {bal_m:.4f} ± {bal_s:.4f}")
    print(f"acc          : {acc_m:.4f} ± {acc_s:.4f}")
    print(f"auc          : {auc_m:.4f} ± {auc_s:.4f}")
    print(f"f1           : {f1_m:.4f} ± {f1_s:.4f}")
    print(f"precision    : {pre_m:.4f} ± {pre_s:.4f}")
    print(f"recall       : {rec_m:.4f} ± {rec_s:.4f}")
    print(f"gate_usage   : {gate_summary}")

    summary = {
        "dataset": args.dataset,
        "model": model_tag,
        "selection_metric": args.selection_metric,
        "threshold_metric": args.threshold_metric,
        "experts": ["mlp", "cheb", "gt", "gcn"],
        "router_level": "node",
        "graph_prior_level": "graph_soft_mixing",
        "use_identity": args.use_identity,
        "use_llm_stage1": args.use_llm_stage1,
        "use_llm_stage2": args.use_llm_stage2,
        "lambda_importance": args.lambda_importance,
        "lambda_load": args.lambda_load,
        "lambda_z": args.lambda_z,
        "feature_noise_robustness": (
            robust_summary.to_dict(orient="records") if robust_summary is not None else None
        ),
        "neuro_guided_bias": {
            "use_neuro_bias": args.use_neuro_bias,
            "neuro_bias_scale_init": args.neuro_bias_scale_init,
        },
        "identity_definition": {
            "structural_prior_identity": "AAL.xlsx column E" if args.use_identity else None,
            "topological_role_identity": [
                "degree",
                "strength",
                "participation_coefficient",
                "within_ratio",
                "clustering",
                "avg_neigh_deg",
                "two_hop",
                "eigencent",
            ] if args.use_identity else None,
        },
        "num_nets": num_nets if args.use_identity else None,
        "network_mapping": net_name_to_id if args.use_identity else None,
        "llm_dim": llm_dim if (args.use_llm_stage1 or args.use_llm_stage2) else None,
        "loss_weights": {
            "lambda_prior": args.lambda_prior,
            "lambda_soft_graph_balance": args.lambda_soft_graph_balance,
            "lambda_soft_expert_balance": args.lambda_soft_expert_balance,
            "lambda_min_usage": args.lambda_min_usage,
            "min_usage": args.min_usage,
        },
        "router_settings": {
            "top_k": args.top_k,
            "router_noise_std": args.router_noise_std,
            "router_temperature_init": args.router_temperature,
            "schedule": "epochs 1-6: topk=4,temp=2.5; epochs 7-15: topk=3,temp=1.8; epochs>=16: topk=final,temp=1.2",
            "stage1_prior_mixing": True,
        },
        "threshold_search": {
            "thr_min": args.thr_min,
            "thr_max": args.thr_max,
            "thr_steps": args.thr_steps,
        },
        "metrics_mean_std": {
            "balanced_acc": {"mean": bal_m, "std": bal_s},
            "accuracy": {"mean": acc_m, "std": acc_s},
            "precision": {"mean": pre_m, "std": pre_s},
            "recall": {"mean": rec_m, "std": rec_s},
            "f1_score": {"mean": f1_m, "std": f1_s},
            "roc_auc": {"mean": auc_m, "std": auc_s},
        },
        "gate_usage_mean_std": gate_summary,
        "majority_class_acc": float(majority_acc),
        "args": vars(args),
    }

    summary_path = results_dir / f"{args.dataset}_{model_tag}_5fold_summary.json"
    save_json(summary, summary_path)
    print(f"\n[SAVED] {summary_path}")


if __name__ == "__main__":
    main()