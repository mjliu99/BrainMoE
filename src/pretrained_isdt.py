# ============================================================
# pretrained_isdt.py
# Combined: export_token + build_token_dataset + print analysis
# All in-memory, no .pt file I/O
# ============================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "isdt"))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from tqdm import tqdm
from isdt.vq import VectorQuantize


# ======================== Helpers ========================

def load_aal_network_onehot(aal_xlsx, N=116, net_names=None):
    import pandas as pd
    df = pd.read_excel(aal_xlsx)
    col = "Unnamed: 4" if "Unnamed: 4" in df.columns else df.columns[4]
    net_str = df[col].astype(str).values[:N]
    roi_col = "Unnamed: 2" if "Unnamed: 2" in df.columns else df.columns[2]
    roi_names = df[roi_col].astype(str).values[:N].tolist()
    if net_names is None:
        net_names = ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"]
    name2id = {n: i for i, n in enumerate(net_names)}
    cleaned = [str(s).strip().upper() for s in net_str]
    net_id = torch.tensor([name2id[s] for s in cleaned], dtype=torch.long)
    net_onehot = torch.zeros(N, len(net_names), dtype=torch.float32)
    net_onehot[torch.arange(N), net_id] = 1.0
    return net_id, net_onehot, net_names, roi_names


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


def _force_2d(x):
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        if x.size(0) == 1:
            return x.squeeze(0)
        if x.size(1) == 1:
            return x.squeeze(1)
    raise RuntimeError(f"Cannot force tensor to 2D, got shape {tuple(x.shape)}")


def _force_dist_2d(dist):
    if dist.dim() == 2:
        return dist
    if dist.dim() == 3:
        if dist.size(0) == 1:
            return dist.squeeze(0)
        if dist.size(1) == 1:
            return dist.squeeze(1)
    raise RuntimeError(f"Cannot force dist to 2D, got shape {tuple(dist.shape)}")


# ======================== Topo Features ========================

@torch.no_grad()
def topo_features_rich(edge_index, edge_weight, net_id, N, device):
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if edge_weight is None:
        w = torch.ones(src.numel(), device=device)
    else:
        w = edge_weight.to(device).view(-1)

    deg = torch.zeros(N, device=device).scatter_add_(0, src, torch.ones_like(w))
    strength = torch.zeros(N, device=device).scatter_add_(0, src, w)

    same = (net_id[src] == net_id[dst]).float()
    within_w = torch.zeros(N, device=device).scatter_add_(0, src, w * same)
    cross_w = torch.zeros(N, device=device).scatter_add_(0, src, w * (1 - same))
    within_ratio = within_w / (within_w + cross_w + 1e-8)

    num_nets = int(net_id.max().item() + 1)
    kis = torch.zeros(N, num_nets, device=device)
    dst_net = net_id[dst]
    kis.index_put_((src, dst_net), w, accumulate=True)
    k_i = kis.sum(dim=1, keepdim=True) + 1e-8
    pcoef = 1.0 - ((kis / k_i) ** 2).sum(dim=1)

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
        zscore(deg), zscore(strength), zscore(pcoef), zscore(within_ratio),
        zscore(clustering), zscore(avg_neigh_deg), zscore(two_hop), zscore(eigencent),
    ], dim=-1)
    return feats


# ======================== ISDT Model (inference) ========================

class ISDT(nn.Module):
    def __init__(self, in_dim, hid_dim=128, codebook_size=256, top_m=32, vq_commit=0.25):
        super().__init__()
        self.top_m = top_m
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
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
    def forward(self, h0):
        h0 = _force_2d(h0)
        H = _force_2d(self.enc(h0))
        z_m, z_t, z_p = self.Wm(H), self.Wt(H), self.Wp(H)

        _, _, _, dist_m, _ = self.vq_m(z_m)
        _, _, _, dist_t, _ = self.vq_t(z_t)
        _, _, _, dist_p, _ = self.vq_p(z_p)

        dist_m = _force_dist_2d(dist_m)
        dist_t = _force_dist_2d(dist_t)
        dist_p = _force_dist_2d(dist_p)

        k_m = dist_m.argmin(dim=-1).view(-1)
        k_t = dist_t.argmin(dim=-1).view(-1)
        k_p = dist_p.argmin(dim=-1).view(-1)

        codes = torch.stack([k_m, k_t, k_p], dim=-1)  # [N, 3]
        N = codes.size(0)

        alpha = torch.sigmoid(self.key_scorer(H)).squeeze(-1).view(-1)
        M = min(self.top_m, N)
        key_idx = torch.topk(alpha, k=M, largest=True).indices.clamp(0, N - 1)

        return codes, key_idx, codes[key_idx]


# ======================== Token Dataset Building ========================

def pad_stack_skey(skey_list, L=32):
    G = len(skey_list)
    tokens = torch.zeros(G, L, 3, dtype=torch.long)
    mask = torch.zeros(G, L, dtype=torch.bool)
    for i, sk in enumerate(skey_list):
        sk = sk.long()
        Mi = min(sk.size(0), L)
        tokens[i, :Mi] = sk[:Mi]
        mask[i, :Mi] = True
    return tokens, mask


def triple_to_single_id(tokens, K=256):
    m = tokens[..., 0]
    t = tokens[..., 1]
    p = tokens[..., 2]
    return (m * (K * K) + t * K + p).long()


# ======================== Print Analysis ========================

def print_token_analysis(tokens, attn_mask, codes_all, key_idx_all, h0_all,
                         top_m=32, x_dim=116, net_dim=9, topo_dim=8, graph_id=0):
    NET_SLICE = slice(x_dim, x_dim + net_dim)
    TOPO_SLICE = slice(x_dim + net_dim, x_dim + net_dim + topo_dim)

    # token-level freq (key nodes only)
    token_freq = {"M": Counter(), "T": Counter(), "P": Counter()}
    G, T, _ = tokens.shape
    for g in range(G):
        for t in range(T):
            if not attn_mask[g, t].item():
                continue
            m, tt, p = tokens[g, t].tolist()
            token_freq["M"][int(m)] += 1
            token_freq["T"][int(tt)] += 1
            token_freq["P"][int(p)] += 1

    # node-level freq (all nodes)
    node_freq = {"M": Counter(), "T": Counter(), "P": Counter()}
    for codes in codes_all:
        for node_id in range(codes.size(0)):
            m, tt, p = codes[node_id].tolist()
            node_freq["M"][int(m)] += 1
            node_freq["T"][int(tt)] += 1
            node_freq["P"][int(p)] += 1

    # proto -> grounding nodes
    proto2nodes = defaultdict(list)
    for g_id, codes in enumerate(codes_all):
        for node_id in range(codes.size(0)):
            m, tt, p = codes[node_id].tolist()
            proto2nodes[("M", int(m))].append((g_id, node_id))
            proto2nodes[("T", int(tt))].append((g_id, node_id))
            proto2nodes[("P", int(p))].append((g_id, node_id))

    def prototype_semantics(proto_type, proto_id):
        nodes = proto2nodes.get((proto_type, proto_id), [])
        if not nodes:
            return None
        feats = torch.stack([h0_all[g][n] for g, n in nodes])
        net_mean = feats[:, NET_SLICE].mean(dim=0)
        topo_mean = feats[:, TOPO_SLICE].mean(dim=0)
        return net_mean, topo_mean, len(nodes)

    graph_id = min(graph_id, G - 1)
    print("=" * 96)
    print(f"Graph #{graph_id} | Top-{top_m} tokens + global prototype grounding")
    print("=" * 96)

    key_idx = key_idx_all[graph_id]
    for pos in range(min(top_m, T)):
        if not attn_mask[graph_id, pos].item():
            continue

        roi_id = int(key_idx[pos].item())
        m, tt, p = tokens[graph_id, pos].tolist()
        m, tt, p = int(m), int(tt), int(p)

        print(f"\n[Token pos {pos:02d}] ROI id = {roi_id:03d} | Triplet = (M={m}, T={tt}, P={p})")

        for proto_type, proto_id in [("M", m), ("T", tt), ("P", p)]:
            tf = token_freq[proto_type][proto_id]
            nf = node_freq[proto_type][proto_id]
            print(f"  |-- {proto_type}-Proto {proto_id:03d} | token_freq={tf} | node_freq={nf}")

            sem = prototype_semantics(proto_type, proto_id)
            if sem is None:
                print("     * No grounding nodes found.")
                continue

            net_mean, topo_mean, n_nodes = sem
            print(f"     * Grounding nodes (global): {n_nodes}")
            print(f"     * Network mean ({net_dim}-d): {net_mean.numpy().round(3)}")
            print(f"     * Topology mean ({topo_dim}-d): {topo_mean.numpy().round(3)}")

    print("\n[DONE]")


# ======================== Checkpoint Loader ========================

def load_isdt_from_checkpoint(ckpt_path, device="cpu", hid_dim=128,
                              codebook_size=256, top_m=32, vq_commit=0.25):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    topo_dim = ckpt.get("topo_dim", 8)
    in_dim = ckpt.get("in_dim", None)
    net_names = ckpt.get("net_names", ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"])
    if in_dim is None:
        in_dim = 116 + len(net_names) + topo_dim

    model = ISDT(in_dim=in_dim, hid_dim=hid_dim, codebook_size=codebook_size,
                 top_m=top_m, vq_commit=vq_commit).to(device)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [LOAD] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  [LOAD] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    return model, {"topo_dim": topo_dim, "in_dim": in_dim, "net_names": net_names}


# ======================== Main Pipeline ========================

@torch.no_grad()
def extract_pretrained_isdt(loader, dataset="ABIDE", aal_xlsx=None, top_m=32,
                            codebook_size=256, print_analysis=True, graph_id=0,
                            use_cache=True):
    N = 116
    TOPO_DIM = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # ---- Use dummy loader if none provided ----
    if loader is None:
        loader = create_dummy_loader(num_graphs=500, N=N, batch_size=32)

    # ---- Load checkpoint or create dummy model ----
    ckpt_path = os.path.join("isdt", f"isdt_ckpt_{dataset.lower()}", "isdt_best.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        topo_dim = ckpt.get("topo_dim", TOPO_DIM)
        in_dim = ckpt.get("in_dim", None)
        net_names = ckpt.get("net_names",
                             ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"])
        if in_dim is None:
            in_dim = N + len(net_names) + topo_dim

        model = ISDT(in_dim=in_dim, hid_dim=128,
                     codebook_size=codebook_size, top_m=top_m)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] loaded: {ckpt_path} | in_dim={in_dim}, topo_dim={topo_dim}")
    else:
        net_names = ["SMN", "VN", "DAN", "VAN", "LIN", "FPN", "DMN", "CBL", "SBN"]
        topo_dim = TOPO_DIM
        in_dim = N + len(net_names) + topo_dim
        model = ISDT(in_dim=in_dim, hid_dim=128,
                     codebook_size=codebook_size, top_m=top_m)
        print(f"[WARN] No checkpoint at {ckpt_path}, using random model for testing")

    num_nets = len(net_names)

    # ---- Build net_id / net_onehot ----
    if aal_xlsx is not None:
        net_id, net_onehot, net_names, roi_names = load_aal_network_onehot(
            aal_xlsx, N=N, net_names=net_names)
        print(f"[AAL] loaded: {aal_xlsx} | num_nets={num_nets} | net_names={net_names}")
    else:
        net_id = torch.randint(0, num_nets, (N,))
        net_onehot = torch.zeros(N, num_nets)
        net_onehot[torch.arange(N), net_id] = 1.0
        roi_names = [f"ROI_{i}" for i in range(N)]

    model.eval()
    model.to(device)

    codes_all, key_idx_all, skey_all, h0_all = [], [], [], []

    # ---- Step 1: Export tokens (in memory) ----
    for data in tqdm(list(iter_graphs_from_loader(loader)), desc="[Export] graphs"):
        x_node = get_x_node_2d(data).to(device)
        N_cur = x_node.size(0)

        edge_index = data.edge_index.long().to(device)
        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.float().to(device)

        nid = net_id[:N_cur].to(device)
        noh = net_onehot[:N_cur].to(device)

        topo = topo_features_rich(edge_index, edge_weight, nid, N=N_cur, device=device)
        assert topo.size(1) == topo_dim, \
            f"Topo dim mismatch: got {topo.size(1)}, expected {topo_dim}"

        h0 = torch.cat([x_node, noh, topo], dim=-1)

        codes, key_idx, S_key = model(h0)

        codes_all.append(codes.cpu())
        key_idx_all.append(key_idx.cpu())
        skey_all.append(S_key.cpu())
        h0_all.append(h0.cpu())

    print(f"[Export] {len(codes_all)} graphs processed")

    # ---- Step 2: Build token dataset (pad + mask + single IDs) ----
    tokens, attn_mask = pad_stack_skey(skey_all, L=top_m)
    token_ids = triple_to_single_id(tokens, K=codebook_size)

    print(f"[Build]  tokens={tuple(tokens.shape)}  mask={tuple(attn_mask.shape)}  "
          f"ids={tuple(token_ids.shape)}")

    x_dim = h0_all[0].size(1) - net_onehot.size(1) - topo_dim if h0_all else N
    net_dim = net_onehot.size(1)

    # ---- Step 3: Print analysis ----
    if print_analysis and len(codes_all) > 0:
        print_token_analysis(
            tokens, attn_mask, codes_all, key_idx_all, h0_all,
            top_m=top_m, x_dim=x_dim, net_dim=net_dim,
            topo_dim=topo_dim, graph_id=graph_id,
        )

    # ---- Step 4: Build isdt_json_list (Structural Symbol Input) ----
    NET_SLICE = slice(x_dim, x_dim + net_dim)
    TOPO_SLICE = slice(x_dim + net_dim, x_dim + net_dim + topo_dim)
    G, L, _ = tokens.shape

    # token-level freq (key nodes only)
    token_freq = {"M": Counter(), "T": Counter(), "P": Counter()}
    for idx in attn_mask.nonzero(as_tuple=False):
        m, tt, p = tokens[idx[0], idx[1]].tolist()
        token_freq["M"][int(m)] += 1
        token_freq["T"][int(tt)] += 1
        token_freq["P"][int(p)] += 1

    # node-level freq + proto grounding
    node_freq = {"M": Counter(), "T": Counter(), "P": Counter()}
    proto2nodes = defaultdict(list)
    for g_id, codes in enumerate(codes_all):
        for node_id in range(codes.size(0)):
            m, tt, p = codes[node_id].tolist()
            for ptype, pid in [("M", int(m)), ("T", int(tt)), ("P", int(p))]:
                node_freq[ptype][pid] += 1
                proto2nodes[(ptype, pid)].append((g_id, node_id))

    # pre-compute proto cache with string means (like print output)
    proto_cache = {}
    for (ptype, pid), nodes in proto2nodes.items():
        feats = torch.stack([h0_all[g][n] for g, n in nodes])
        proto_cache[(ptype, pid)] = {
            "id": pid,
            "token_freq": token_freq[ptype][pid],
            "node_freq": node_freq[ptype][pid],
            "network_mean": str(feats[:, NET_SLICE].mean(0).numpy().round(3)),
            "topology_mean": str(feats[:, TOPO_SLICE].mean(0).numpy().round(3)),
        }

    # build per-subject json (flat lookup, no nested proto loop)
    isdt_json_list = []
    for g in tqdm(range(G), desc="[Build] subjects"):
        subject_tokens = []
        key_idx = key_idx_all[g]
        for pos in range(L):
            if not attn_mask[g, pos].item():
                continue
            roi_id = int(key_idx[pos].item())
            m, tt, p = [int(v) for v in tokens[g, pos].tolist()]
            subject_tokens.append({
                "pos": pos,
                "roi_name": roi_names[roi_id] if roi_id < len(roi_names) else f"ROI_{roi_id}",
                "triplet": {"M": m, "T": tt, "P": p},
                "proto": {
                    "M": proto_cache.get(("M", m), {}),
                    "T": proto_cache.get(("T", tt), {}),
                    "P": proto_cache.get(("P", p), {}),
                },
            })
        isdt_json_list.append({"structural_symbol_input_of_brain_network": subject_tokens})

    print(f"[JSON] built {len(isdt_json_list)} structural_symbol_input from pretrained ISDT")

    if use_cache:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsons")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{dataset}_isdt_json.pt")
        torch.save(isdt_json_list, cache_file)
        print(f"[Cache] saved {cache_file}")

    return isdt_json_list


# ======================== Dummy Loader for Testing ========================

def create_dummy_loader(num_graphs=5, N=116, batch_size=1):
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    data_list = []
    for _ in range(num_graphs):
        corr = torch.randn(N, N)
        corr = (corr + corr.t()) / 2
        corr.fill_diagonal_(1.0)

        # sparse edges via threshold
        mask = corr.abs() > 0.3
        mask.fill_diagonal_(False)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_weight = corr[edge_index[0], edge_index[1]]

        data = Data(
            x=corr,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor([0], dtype=torch.long),
        )
        data_list.append(data)

    return DataLoader(data_list, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    # run full pipeline (dummy: no loader, no checkpoint, no aal_xlsx)
    isdt_json_list = extract_pretrained_isdt(
        loader=None,
        aal_xlsx="./isdt/AAL.xlsx",
        dataset="ABIDE",
        print_analysis=True,
        graph_id=0,
    )

    print(f"\n--- Summary ---")
    print(f"subjects: {len(isdt_json_list)}")
    if isdt_json_list:
        print(f"tokens per subject[0]: {len(isdt_json_list[0]['structural_symbol_input_of_brain_network'])}")
