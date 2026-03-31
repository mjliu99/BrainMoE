# ============================================================
# export_token_trained.py
# Export tokens using a TRAINED ISDT checkpoint (isdt_best.pt)
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vq import VectorQuantize
from dataset_loader import create_dataloader


# ---------------- AAL (CBL) ----------------
def load_aal_network_onehot(aal_xlsx: str, N: int = 116, net_names=None):
    df = pd.read_excel(aal_xlsx)
    col = "Unnamed: 4" if "Unnamed: 4" in df.columns else df.columns[4]
    net_str = df[col].astype(str).values[:N]

    if net_names is None:
        net_names = ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"]

    name2id = {n: i for i, n in enumerate(net_names)}
    cleaned = [str(s).strip().upper() for s in net_str]
    net_id = torch.tensor([name2id[s] for s in cleaned], dtype=torch.long)

    net_onehot = torch.zeros(N, len(net_names), dtype=torch.float32)
    net_onehot[torch.arange(N), net_id] = 1.0
    return net_id, net_onehot, net_names, col


# ---------------- get x_node as [N,N] ----------------
def get_x_node_2d(data):
    x = data.x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

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


# ---------------- topo features: match your training script ----------------
@torch.no_grad()
def topo_features_rich(edge_index, edge_weight, net_id, N: int, device):
    """
    Must match the topo_features_rich used in your pretraining.
    Output: [N, D_topo], D_topo=8
    """
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if edge_weight is None:
        w = torch.ones(src.numel(), device=device)
    else:
        w = edge_weight.to(device).view(-1)

    deg = torch.zeros(N, device=device).scatter_add_(0, src, torch.ones_like(w))
    strength = torch.zeros(N, device=device).scatter_add_(0, src, w)

    same = (net_id[src] == net_id[dst]).float()
    within_w = torch.zeros(N, device=device).scatter_add_(0, src, w * same)
    cross_w  = torch.zeros(N, device=device).scatter_add_(0, src, w * (1 - same))
    within_ratio = within_w / (within_w + cross_w + 1e-8)

    num_nets = int(net_id.max().item() + 1)
    kis = torch.zeros(N, num_nets, device=device)
    dst_net = net_id[dst]
    kis.index_put_((src, dst_net), w, accumulate=True)
    k_i = kis.sum(dim=1, keepdim=True) + 1e-8
    pcoef = 1.0 - ((kis / k_i) ** 2).sum(dim=1)

    # binary + weighted adjacency
    A = torch.zeros(N, N, device=device)
    A[src, dst] = 1.0
    A = torch.maximum(A, A.t())

    Aw = torch.zeros(N, N, device=device)
    Aw[src, dst] = w
    Aw = torch.maximum(Aw, Aw.t())

    tri = torch.diag(A @ A @ A) / 2.0
    denom = deg * (deg - 1.0) + 1e-8
    clustering = 2.0 * tri / denom

    neigh_deg_sum = (A * deg.view(1, -1)).sum(dim=1)
    avg_neigh_deg = neigh_deg_sum / (A.sum(dim=1) + 1e-8)

    A2 = (A @ A) > 0
    two_hop = A2.float().sum(dim=1)

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


# ---------------- helpers: force shapes ----------------
def _force_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        if x.size(0) == 1:
            return x.squeeze(0)
        if x.size(1) == 1:
            return x.squeeze(1)
    raise RuntimeError(f"Cannot force tensor to 2D [N,D], got shape {tuple(x.shape)}")


def _force_dist_2d(dist: torch.Tensor) -> torch.Tensor:
    if dist.dim() == 2:
        return dist
    if dist.dim() == 3:
        if dist.size(0) == 1:
            return dist.squeeze(0)
        if dist.size(1) == 1:
            return dist.squeeze(1)
    raise RuntimeError(f"Cannot force dist to 2D [N,K], got shape {tuple(dist.shape)}")


# ---------------- ISDT (must match training architecture) ----------------
class ISDT(nn.Module):
    def __init__(self, in_dim, hid_dim=128, codebook_size=256, top_m=32, vq_commit=0.25):
        super().__init__()
        self.top_m = top_m

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

        self.key_scorer = nn.Linear(hid_dim, 1)

    @torch.no_grad()
    def forward(self, h0, argmax_code=False, debug=False):
        h0 = _force_2d(h0)
        H = _force_2d(self.enc(h0))

        z_m, z_t, z_p = self.Wm(H), self.Wt(H), self.Wp(H)

        _, _, _, dist_m, _ = self.vq_m(z_m)
        _, _, _, dist_t, _ = self.vq_t(z_t)
        _, _, _, dist_p, _ = self.vq_p(z_p)

        dist_m = _force_dist_2d(dist_m)
        dist_t = _force_dist_2d(dist_t)
        dist_p = _force_dist_2d(dist_p)

        if debug:
            print("[DBG] h0", tuple(h0.shape), "H", tuple(H.shape),
                  "| dist_m", tuple(dist_m.shape))

        if argmax_code:
            k_m = dist_m.argmax(dim=-1)
            k_t = dist_t.argmax(dim=-1)
            k_p = dist_p.argmax(dim=-1)
        else:
            k_m = dist_m.argmin(dim=-1)
            k_t = dist_t.argmin(dim=-1)
            k_p = dist_p.argmin(dim=-1)

        k_m = k_m.view(-1)
        k_t = k_t.view(-1)
        k_p = k_p.view(-1)

        codes = torch.stack([k_m, k_t, k_p], dim=-1)  # [N,3]
        N = codes.size(0)

        alpha = torch.sigmoid(self.key_scorer(H)).squeeze(-1).view(-1)  # [N]
        M = min(self.top_m, N)
        key_idx = torch.topk(alpha, k=M, largest=True).indices
        key_idx = key_idx.clamp(0, N - 1)

        return codes, key_idx, codes[key_idx]


# ---------------- export split ----------------
@torch.no_grad()
def export_split(model, loader, net_id, net_onehot, out_path,
                 topo_dim, argmax_code=False, max_graphs=None, debug=False, device="cpu"):
    model.eval()
    codes_all, key_all, skey_all, h0_all = [], [], [], []
    n = 0

    for data in iter_graphs_from_loader(loader):
        x_node = get_x_node_2d(data).to(device)  # [N,N]
        N = x_node.size(0)

        edge_index = data.edge_index.long().to(device)
        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.float().to(device)

        nid = net_id[:N].to(device)
        noh = net_onehot[:N].to(device)

        # use SAME topo used in training
        topo = topo_features_rich(edge_index, edge_weight, nid, N=N, device=device)  # [N,8] by default

        if topo.size(1) != topo_dim:
            raise RuntimeError(f"Topo dim mismatch: topo={topo.size(1)} but ckpt topo_dim={topo_dim}. "
                               f"Make sure export uses the SAME topo_features as training.")

        h0 = torch.cat([x_node, noh, topo], dim=-1)  # [N, in_dim]

        codes, key_idx, S_key = model(h0, argmax_code=argmax_code, debug=debug)

        codes_all.append(codes.cpu())
        key_all.append(key_idx.cpu())
        skey_all.append(S_key.cpu())
        h0_all.append(h0.cpu())

        n += 1
        if max_graphs is not None and n >= max_graphs:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "codes": codes_all,
        "key_idx": key_all,
        "S_key": skey_all,
        "h0": h0_all,   # keep for prototype analysis
    }, out_path)

    print(f"[SAVED] {out_path} graphs={len(codes_all)} size={os.path.getsize(out_path)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ABIDE", choices=["ABIDE", "ADHD"])
    ap.add_argument("--aal_xlsx", required=True)

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--save_dir", default="isdt_ckpt_abide")
    ap.add_argument("--ckpt", default=None, help="Path to isdt_best.pt (default: <save_dir>/isdt_best.pt)")

    ap.add_argument("--argmax_code", action="store_true")
    ap.add_argument("--max_graphs", type=int, default=-1, help="-1 means all")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cpu")

    # override if your ckpt doesn't store these keys
    ap.add_argument("--hid_dim", type=int, default=128)
    ap.add_argument("--codebook_size", type=int, default=256)
    ap.add_argument("--top_m", type=int, default=32)
    ap.add_argument("--vq_commit", type=float, default=0.25)

    args = ap.parse_args()
    device = torch.device(args.device)

    ckpt_path = args.ckpt if args.ckpt is not None else os.path.join(args.save_dir, "isdt_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # auto read dims from ckpt if available
    topo_dim = ckpt.get("topo_dim", 8)          # default 8 (from topo_features_rich)
    in_dim = ckpt.get("in_dim", None)

    # If ckpt doesn't store in_dim, fall back to 116 + 9 + topo_dim
    net_names = ckpt.get("net_names", ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"])
    if in_dim is None:
        in_dim = 116 + len(net_names) + topo_dim

    print(f"[CKPT] loaded: {ckpt_path}")
    print(f"[CKPT] in_dim={in_dim}, topo_dim={topo_dim}, nets={len(net_names)}")

    # loaders
    train_loader, dev_loader, test_loader = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        seed=0,
        use_cache=True,
        topk=args.topk,
        use_5fold=False,
        fold_id=0,
    )

    net_id, net_onehot, net_names2, col = load_aal_network_onehot(args.aal_xlsx, N=116, net_names=net_names)
    print(f"[AAL] col={col} nets={net_names2}")

    # build model and load weights
    model = ISDT(
        in_dim=in_dim,
        hid_dim=args.hid_dim,
        codebook_size=args.codebook_size,
        top_m=args.top_m,
        vq_commit=args.vq_commit,
    ).to(device)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[LOAD] done.")
    if missing:
        print("  missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    if unexpected:
        print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    max_graphs = None if args.max_graphs < 0 else args.max_graphs

    export_split(model, train_loader, net_id, net_onehot,
                 out_path=os.path.join(args.save_dir, "tokens_train.pt"),
                 topo_dim=topo_dim,
                 argmax_code=args.argmax_code,
                 max_graphs=max_graphs,
                 debug=args.debug,
                 device=device)

    export_split(model, dev_loader, net_id, net_onehot,
                 out_path=os.path.join(args.save_dir, "tokens_dev.pt"),
                 topo_dim=topo_dim,
                 argmax_code=args.argmax_code,
                 max_graphs=max_graphs,
                 debug=args.debug,
                 device=device)

    export_split(model, test_loader, net_id, net_onehot,
                 out_path=os.path.join(args.save_dir, "tokens_test.pt"),
                 topo_dim=topo_dim,
                 argmax_code=args.argmax_code,
                 max_graphs=max_graphs,
                 debug=args.debug,
                 device=device)


if __name__ == "__main__":
    main()
