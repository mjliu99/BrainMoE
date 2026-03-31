# ============================================================
# pretrain_isdt_enhanced.py
# Enhanced ISDT pretraining (CPU-friendly, stable, richer loss)
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq import VectorQuantize
from dataset_loader import create_dataloader


# ---------------- AAL (CBL) ----------------
def load_aal_network_onehot(aal_xlsx: str, N: int = 116):
    df = pd.read_excel(aal_xlsx)
    col = "Unnamed: 4" if "Unnamed: 4" in df.columns else df.columns[4]
    net_str = df[col].astype(str).values[:N]

    net_names = ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"]
    name2id = {n: i for i, n in enumerate(net_names)}

    cleaned = [str(s).strip().upper() for s in net_str]
    net_id = torch.tensor([name2id[s] for s in cleaned], dtype=torch.long)

    net_onehot = torch.zeros(N, len(net_names), dtype=torch.float32)
    net_onehot[torch.arange(N), net_id] = 1.0
    return net_id, net_onehot, net_names, col


# ---------------- helper: get x_node as [N,N] ----------------
def get_x_node_2d(data):
    x = data.x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # accept [N,N] or [1,N,N]
    if x.dim() == 2:
        pass
    elif x.dim() == 3:
        x = x[0]
    else:
        raise RuntimeError(f"Unexpected data.x shape: {tuple(x.shape)}")
    return x.float()


def iter_graphs_from_loader(loader):
    for batch in loader:
        if isinstance(batch, list):
            data_list = batch
        elif hasattr(batch, "to_data_list"):
            data_list = batch.to_data_list()
        else:
            data_list = [batch]
        for data in data_list:
            yield data


# ---------------- topo features (richer, torch only) ----------------
@torch.no_grad()
def topo_features_rich(edge_index, edge_weight, net_id, N: int, device):
    """
    Return: [N, D_topo] z-scored
    D_topo ~ 10
    """
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if edge_weight is None:
        w = torch.ones(src.numel(), device=device)
    else:
        w = edge_weight.to(device).view(-1)

    # degree / strength
    deg = torch.zeros(N, device=device).scatter_add_(0, src, torch.ones_like(w))
    strength = torch.zeros(N, device=device).scatter_add_(0, src, w)

    # within / cross ratio by atlas network
    same = (net_id[src] == net_id[dst]).float()
    within_w = torch.zeros(N, device=device).scatter_add_(0, src, w * same)
    cross_w  = torch.zeros(N, device=device).scatter_add_(0, src, w * (1 - same))
    within_ratio = within_w / (within_w + cross_w + 1e-8)

    # participation coefficient (weighted)
    num_nets = int(net_id.max().item() + 1)
    kis = torch.zeros(N, num_nets, device=device)
    dst_net = net_id[dst]
    kis.index_put_((src, dst_net), w, accumulate=True)
    k_i = kis.sum(dim=1, keepdim=True) + 1e-8
    pcoef = 1.0 - ((kis / k_i) ** 2).sum(dim=1)

    # build sym adjacency (binary + weighted)
    A = torch.zeros(N, N, device=device)
    A[src, dst] = 1.0
    A = torch.maximum(A, A.t())  # undirected binary

    Aw = torch.zeros(N, N, device=device)
    Aw[src, dst] = w
    Aw = torch.maximum(Aw, Aw.t())  # undirected weighted

    # clustering (binary triangles)
    tri = torch.diag(A @ A @ A) / 2.0
    denom = deg * (deg - 1.0) + 1e-8
    clustering = 2.0 * tri / denom

    # avg neighbor degree (binary)
    neigh_deg_sum = (A * deg.view(1, -1)).sum(dim=1)
    avg_neigh_deg = neigh_deg_sum / (A.sum(dim=1) + 1e-8)

    # 2-hop reach (binary): how many nodes within 2 steps
    A2 = (A @ A) > 0
    two_hop = A2.float().sum(dim=1)

    # eigenvector centrality (power iteration on weighted adjacency)
    v = torch.ones(N, device=device) / (N ** 0.5)
    for _ in range(12):
        v = Aw @ v
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
    ], dim=-1)  # [N,8]

    return feats


# ---------------- losses ----------------
def edge_contrastive_loss(edge_index, H, N, num_neg=1):
    """
    Simple edge contrastive: maximize dot on edges vs random negatives.
    H: [N,D]
    """
    src, dst = edge_index[0], edge_index[1]
    pos = (H[src] * H[dst]).sum(dim=-1)

    # negatives
    neg_u = torch.randint(0, N, (src.numel() * num_neg,), device=H.device)
    neg_v = torch.randint(0, N, (src.numel() * num_neg,), device=H.device)
    neg = (H[neg_u] * H[neg_v]).sum(dim=-1)

    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def build_pseudo_id_labels(topo_feats, num_classes=6):
    """
    topo_feats: [N, D], use degree z-score (feat 0) + within_ratio (feat 3) to build pseudo classes.
    This is only for pretraining structure discovery (not clinical meaning).
    """
    degz = topo_feats[:, 0]
    within = topo_feats[:, 3]
    # binning
    deg_bin = torch.bucketize(degz, torch.tensor([-1.0, -0.2, 0.2, 1.0], device=degz.device))  # 0..4
    within_bin = (within > within.median()).long()  # 0/1
    y = deg_bin * 2 + within_bin  # 0..9
    y = y.clamp_max(num_classes - 1)
    return y


def apply_mask(h0, mask_ratio=0.15):
    """
    h0: [N, Din]
    return masked_h0, mask_idx (bool), mask_token
    """
    N, Din = h0.shape
    m = torch.rand(N, device=h0.device) < mask_ratio
    masked = h0.clone()
    masked[m] = 0.0
    return masked, m


# ---------------- ISDT (enhanced pretrain) ----------------
class ISDT(nn.Module):
    def __init__(self, in_dim, hid_dim=128, codebook_size=256, top_m=32,
                 num_id_classes=6, num_nets=9, vq_commit=0.25):
        super().__init__()
        self.top_m = top_m
        self.num_id_classes = num_id_classes

        # stable MLP encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )

        self.Wm = nn.Linear(hid_dim, hid_dim)
        self.Wt = nn.Linear(hid_dim, hid_dim)
        self.Wp = nn.Linear(hid_dim, hid_dim)

        self.vq_m = VectorQuantize(dim=hid_dim, codebook_size=codebook_size, decay=0.8,
                                   commitment_weight=vq_commit, use_cosine_sim=True)
        self.vq_t = VectorQuantize(dim=hid_dim, codebook_size=codebook_size, decay=0.8,
                                   commitment_weight=vq_commit, use_cosine_sim=True)
        self.vq_p = VectorQuantize(dim=hid_dim, codebook_size=codebook_size, decay=0.8,
                                   commitment_weight=vq_commit, use_cosine_sim=True)

        # key node scorer
        self.key_scorer = nn.Linear(hid_dim, 1)

        # heads for auxiliary tasks
        self.id_head = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_id_classes)
        )
        self.net_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_nets)
        )
        self.mask_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_id_classes)
        )

    def forward(self, h0, argmax_code=False):
        """
        h0: [N, Din]
        """
        H = self.enc(h0)  # [N,hid]
        z_m, z_t, z_p = self.Wm(H), self.Wt(H), self.Wp(H)

        q_m, _, cm, dist_m, _ = self.vq_m(z_m)
        q_t, _, ct, dist_t, _ = self.vq_t(z_t)
        q_p, _, cp, dist_p, _ = self.vq_p(z_p)

        # code indices
        if argmax_code:
            k_m = dist_m.argmax(dim=-1)
            k_t = dist_t.argmax(dim=-1)
            k_p = dist_p.argmax(dim=-1)
        else:
            k_m = dist_m.argmin(dim=-1)
            k_t = dist_t.argmin(dim=-1)
            k_p = dist_p.argmin(dim=-1)

        codes = torch.stack([k_m, k_t, k_p], dim=-1)  # [N,3]

        # key nodes
        alpha = torch.sigmoid(self.key_scorer(H)).squeeze(-1)  # [N]
        M = min(self.top_m, alpha.numel())
        key_idx = torch.topk(alpha, k=M, largest=True).indices  # [M]

        token_emb = torch.cat([q_m, q_t, q_p], dim=-1)  # [N, 3*hid]

        L_vq = cm + ct + cp
        return H, codes, key_idx, token_emb, L_vq


# ---------------- training ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ABIDE", choices=["ABIDE", "ADHD"])
    ap.add_argument("--aal_xlsx", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--save_dir", default="isdt/isdt_ckpt_abide")
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--top_m", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--argmax_code", action="store_true")

    # save strategy
    ap.add_argument("--save_every", type=int, default=0, help="0=disable, e.g. 10 means save every 10 epochs")
    ap.add_argument("--save_best", action="store_true", help="save best dev loss to isdt_best.pt")

    # loss weights
    ap.add_argument("--lam_vq", type=float, default=1.0)
    ap.add_argument("--lam_edge", type=float, default=1.0)
    ap.add_argument("--lam_id", type=float, default=0.5)
    ap.add_argument("--lam_net", type=float, default=0.5)
    ap.add_argument("--lam_mask", type=float, default=0.5)
    ap.add_argument("--mask_ratio", type=float, default=0.15)

    ap.add_argument("--num_id_classes", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print_every", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, dev_loader, test_loader = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        seed=0,
        use_cache=True,
        topk=args.topk,
        use_5fold=False,
        fold_id=0,
    )

    net_id, net_onehot, net_names, col = load_aal_network_onehot(args.aal_xlsx, N=116)
    print(f"[AAL] col={col} nets={net_names}")

    D_topo = 8  # topo_features_rich output dims
    in_dim = 116 + len(net_names) + D_topo  # 116 + 9 + 8 = 133

    model = ISDT(
        in_dim=in_dim,
        hid_dim=128,
        codebook_size=args.codebook_size,
        top_m=args.top_m,
        num_id_classes=args.num_id_classes,
        num_nets=len(net_names)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total = 0.0
        n_graph = 0

        for data in iter_graphs_from_loader(loader):
            # -------- build h0 --------
            x_node = get_x_node_2d(data).to(device)  # [N,N]
            N = x_node.size(0)

            edge_index = data.edge_index.long().to(device)
            edge_weight = None
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                edge_weight = data.edge_attr.float().to(device)

            nid = net_id[:N].to(device)
            noh = net_onehot[:N].to(device)

            topo = topo_features_rich(edge_index, edge_weight, nid, N=N, device=device)  # [N,8]
            h0 = torch.cat([x_node, noh, topo], dim=-1)  # [N,133]

            # -------- masked modeling --------
            h0_masked, m_idx = apply_mask(h0, mask_ratio=args.mask_ratio)

            # -------- forward (masked) --------
            Hm, codes, key_idx, token_emb, L_vq = model(h0_masked, argmax_code=args.argmax_code)

            # -------- losses --------
            # (1) edge contrastive on masked H
            L_edge = edge_contrastive_loss(edge_index, Hm, N)

            # (2) pseudo-id classification on key nodes
            y_pseudo = build_pseudo_id_labels(topo, num_classes=args.num_id_classes)
            logits_id = model.id_head(token_emb[key_idx])
            L_id = F.cross_entropy(logits_id, y_pseudo[key_idx])

            # (3) network classification on key nodes (predict atlas network)
            logits_net = model.net_head(Hm[key_idx])
            L_net = F.cross_entropy(logits_net, nid[key_idx])

            # (4) masked-node prediction: for masked nodes, predict pseudo id from H
            if m_idx.any():
                logits_mask = model.mask_head(Hm[m_idx])
                L_mask = F.cross_entropy(logits_mask, y_pseudo[m_idx])
            else:
                L_mask = torch.tensor(0.0, device=device)

            loss = (
                args.lam_vq * L_vq +
                args.lam_edge * L_edge +
                args.lam_id * L_id +
                args.lam_net * L_net +
                args.lam_mask * L_mask
            )

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            total += loss.item()
            n_graph += 1

        return total / max(n_graph, 1)

    best_dev = 1e18

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        dv = run_epoch(dev_loader, train=False)

        if ep % args.print_every == 0:
            print(f"[Epoch {ep:03d}] train={tr:.4f} dev={dv:.4f} | "
                  f"(lam_vq={args.lam_vq}, edge={args.lam_edge}, id={args.lam_id}, net={args.lam_net}, mask={args.lam_mask})")

        # save best
        if args.save_best and dv < best_dev:
            best_dev = dv
            ckpt = {
                "epoch": ep,
                "best_dev": best_dev,
                "model": model.state_dict(),
                "args": vars(args),
                "net_names": net_names,
                "topo_dim": D_topo,
                "in_dim": in_dim,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "isdt_best.pt"))
            print(f"  [SAVE] best -> isdt_best.pt (dev={best_dev:.4f})")

        # optional periodic save
        if args.save_every > 0 and (ep % args.save_every == 0):
            ckpt = {
                "epoch": ep,
                "dev": dv,
                "model": model.state_dict(),
                "args": vars(args),
            }
            outp = os.path.join(args.save_dir, f"isdt_epoch_{ep:03d}.pt")
            torch.save(ckpt, outp)
            print(f"  [SAVE] {outp}")

    print("✅ Enhanced ISDT pretraining finished.")


if __name__ == "__main__":
    main()
