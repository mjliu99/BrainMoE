import torch
from collections import defaultdict, Counter
import numpy as np

# =====================
# Config
# =====================
SAVE_DIR = "./isdt/isdt_ckpt_abide"
GRAPH_ID = 0
TOP_M = 32

# h0 layout (must match export_token.py / ckpt meta)
X_DIM = 116
NET_DIM = 9
# topo_dim = in_dim - (116 + 9) = 133 - 125 = 8
TOPO_DIM = 8

NET_SLICE  = slice(X_DIM, X_DIM + NET_DIM)             # 116:125
TOPO_SLICE = slice(X_DIM + NET_DIM, X_DIM + NET_DIM + TOPO_DIM)  # 125:133

# topo feature names (按你 export_token.py 的 topo_features_from_pyg 返回顺序来填)
# 你现在返回的是 8 维（从你输出看），所以这里给 8 个名字占位：
TOPO_NAMES = ["deg", "strength", "pcoef", "within_ratio", "clustering",
              "topo5", "topo6", "topo7"]  # TODO: 改成你真实的8维含义


# =====================
# Load data
# =====================
token_data = torch.load(f"{SAVE_DIR}/token_dataset_train.pt")
tokens = token_data["tokens"]       # [G, 32, 3]
attn_mask = token_data["attn_mask"] # [G, 32]

node_data = torch.load(f"{SAVE_DIR}/tokens_train.pt")
codes_all = node_data["codes"]      # list of [N,3]
h0_all    = node_data["h0"]         # list of [N,133]
key_all   = node_data["key_idx"]    # list of [32] for each graph

assert len(codes_all) == tokens.size(0), "Mismatch: num_graphs in tokens vs codes_all"
assert len(h0_all) == tokens.size(0), "Mismatch: num_graphs in tokens vs h0_all"
assert len(key_all) == tokens.size(0), "Mismatch: num_graphs in tokens vs key_idx"

# =====================
# Recompute frequencies (to avoid stale prototype_frequency.pt)
# =====================
token_freq = {"M": Counter(), "T": Counter(), "P": Counter()}
node_freq  = {"M": Counter(), "T": Counter(), "P": Counter()}

# token-level freq (Top-32 only)
G, T, _ = tokens.shape
for g in range(G):
    for t in range(T):
        if attn_mask[g, t].item() == 0:
            continue
        m, tt, p = tokens[g, t].tolist()
        token_freq["M"][int(m)] += 1
        token_freq["T"][int(tt)] += 1
        token_freq["P"][int(p)] += 1

# node-level freq (all nodes)
for g_id, codes in enumerate(codes_all):
    for node_id in range(codes.size(0)):
        m, tt, p = codes[node_id].tolist()
        node_freq["M"][int(m)] += 1
        node_freq["T"][int(tt)] += 1
        node_freq["P"][int(p)] += 1


# =====================
# Build proto -> node index for grounding
# =====================
proto2nodes = defaultdict(list)
for g_id, codes in enumerate(codes_all):
    for node_id in range(codes.size(0)):
        m, tt, p = codes[node_id].tolist()
        proto2nodes[("M", int(m))].append((g_id, node_id))
        proto2nodes[("T", int(tt))].append((g_id, node_id))
        proto2nodes[("P", int(p))].append((g_id, node_id))


def prototype_semantics(proto_type: str, proto_id: int):
    """
    Return global mean semantics for this prototype by aggregating all nodes across all graphs.
    """
    nodes = proto2nodes.get((proto_type, proto_id), [])
    if len(nodes) == 0:
        return None

    feats = torch.stack([h0_all[g][n] for (g, n) in nodes])  # [num_nodes, in_dim]
    net_mean = feats[:, NET_SLICE].mean(dim=0)               # [9]
    topo_mean = feats[:, TOPO_SLICE].mean(dim=0)             # [8]

    return net_mean, topo_mean, len(nodes)


# =====================
# Print Top-32 tokens for a specific graph
# =====================
print("=" * 96)
print(f"Graph #{GRAPH_ID} | Top-{TOP_M} tokens (subject-specific) + global prototype grounding")
print("=" * 96)

key_idx = key_all[GRAPH_ID]  # [32], each is a real ROI id in 0..N-1

for pos in range(TOP_M):
    if attn_mask[GRAPH_ID, pos].item() == 0:
        continue

    roi_id = int(key_idx[pos].item())  # IMPORTANT: real ROI index
    m, tt, p = tokens[GRAPH_ID, pos].tolist()
    m, tt, p = int(m), int(tt), int(p)

    print(f"\n[Token pos {pos:02d}] ROI id = {roi_id:03d} | Triplet = (M={m}, T={tt}, P={p})")

    for proto_type, proto_id in [("M", m), ("T", tt), ("P", p)]:
        tf = token_freq[proto_type][proto_id]
        nf = node_freq[proto_type][proto_id]

        print(f"  └─ {proto_type}-Proto {proto_id:03d} | token_freq={tf} | node_freq={nf}")

        sem = prototype_semantics(proto_type, proto_id)
        if sem is None:
            print("     • No grounding nodes found in codes_all (unexpected).")
            continue

        net_mean, topo_mean, n_nodes = sem

        print(f"     • Grounding nodes (global): {n_nodes}")
        print(f"     • Network mean (9-d): {net_mean.numpy().round(3)}")
        print(f"     • Topology mean ({TOPO_DIM}-d): {topo_mean.numpy().round(3)}")
        # 如果你想更可读：
        # for name, val in zip(TOPO_NAMES, topo_mean.numpy()):
        #     print(f"       - {name}: {val:.3f}")

print("\n[DONE]")
