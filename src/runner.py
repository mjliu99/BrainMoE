# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, roc_auc_score
# import numpy as np

# def train_epoch(model, data_loader, optimizer, device, class_weight=None):
#     model.train()
#     ce = nn.CrossEntropyLoss(weight=class_weight)

#     total_loss = 0.0
#     total_correct = 0
#     total_examples = 0

#     for data in data_loader:
#         data = data.to(device)
#         y = data.y.view(-1).long()

#         optimizer.zero_grad()
#         logits, _ = model(data.x, data.edge_index, data.llm_embeddings, data.batch)

#         loss = ce(logits, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * y.size(0)
#         preds = logits.argmax(dim=1)
#         total_correct += (preds == y).sum().item()
#         total_examples += y.size(0)

#     return total_loss / total_examples, total_correct / total_examples


# @torch.no_grad()
# def eval_epoch(model, data_loader, device):
#     model.eval()
#     ce = nn.CrossEntropyLoss()

#     total_loss = 0.0
#     total_correct = 0
#     total_examples = 0
#     all_y = []
#     all_preds = []
#     all_probs = []

#     with torch.no_grad():
#         for data in data_loader:
#             data = data.to(device)
#             y = data.y.view(-1).long()

#             logits, _ = model(data.x, data.edge_index, data.llm_embeddings, data.batch)
#             loss = ce(logits, y)

#             total_loss += loss.item() * y.size(0)
#             preds = logits.argmax(dim=1)
#             total_correct += (preds == y).sum().item()
#             total_examples += y.size(0)

#             all_y.append(y.detach().cpu())
#             all_preds.append(preds.detach().cpu())
#             all_probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu())

#     y_true = torch.cat(all_y).numpy()
#     y_pred = torch.cat(all_preds).numpy()
#     y_prob = torch.cat(all_probs).numpy()

#     roc_auc = 0.0   
#     if len(np.unique(y_true)) > 1:
#         roc_auc = roc_auc_score(y_true, y_prob)

#     metrics = {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall": recall_score(y_true, y_pred, zero_division=0),
#         "f1_score": f1_score(y_true, y_pred, zero_division=0),
#         "roc_auc": roc_auc,
#     }
#     return total_loss / total_examples, total_correct / total_examples, metrics

# ============================================================
# runner.py
# ============================================================

import torch
import torch.nn as nn
import numpy as np


def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    0.5 * (KL(p||q) + KL(q||p))
    p,q: [N,E]
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def graph_level_gate_entropy_loss(gates: torch.Tensor, batch: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    A+B: graph-mean gates + entropy regularizer
    """
    N, E = gates.shape
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if B <= 0:
        return torch.zeros((), device=gates.device, dtype=gates.dtype)

    g_sum = torch.zeros(B, E, device=gates.device, dtype=gates.dtype)
    g_sum.index_add_(0, batch, gates)
    counts = torch.bincount(batch, minlength=B).to(gates.device).clamp_min(1).unsqueeze(-1).float()
    g_graph = g_sum / counts

    g_graph = g_graph.clamp_min(eps)
    g_graph = g_graph / g_graph.sum(dim=-1, keepdim=True).clamp_min(eps)

    entropy = -torch.sum(g_graph * torch.log(g_graph), dim=-1)
    return -entropy.mean()


def train_epoch(
    model,
    data_loader,
    optimizer,
    device,
    class_weight=None,
    lambda_prior: float = 0.05,
    lambda_balance: float = 0.001,
):
    model.train()
    ce = nn.CrossEntropyLoss(weight=class_weight)

    total_loss, total_correct, total_examples = 0.0, 0, 0
    total_ce, total_kl, total_bal = 0.0, 0.0, 0.0

    for data in data_loader:
        data = data.to(device)
        y = data.y.view(-1).long()

        optimizer.zero_grad()
        logits, gates, aux = model(data.x, data.edge_index, data.llm_embeddings, data.batch)

        loss_ce = ce(logits, y)
        loss_kl = sym_kl(aux["p_net"], aux["p_router"])
        loss_bal = graph_level_gate_entropy_loss(gates, data.batch)

        loss = loss_ce + lambda_prior * loss_kl + lambda_balance * loss_bal
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_kl += loss_kl.item() * bs
        total_bal += loss_bal.item() * bs

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += bs

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        {"loss_ce": total_ce / total_examples, "loss_kl": total_kl / total_examples, "loss_bal": total_bal / total_examples},
    )