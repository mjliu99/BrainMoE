# import pandas as pd
# import numpy as np
# from pathlib import Path
# import torch
# # from torch_geometric.utils import dense_to_sparse, remove_self_loops
# # from torch_geometric.data import Data
# from sklearn.model_selection import train_test_split, StratifiedKFold
# # try:
# #     from torch_geometric.loader import DataLoader  # PyG >= 2.0
# # except Exception:
# #     from torch_geometric.data import DataLoader    # PyG < 2.0

# from dataclasses import dataclass
# from torch.utils.data import DataLoader

# def dense_to_sparse_torch(A: torch.Tensor):
#     # A: [N,N]
#     idx = (A != 0).nonzero(as_tuple=False).t().contiguous()  # [2,E]
#     val = A[idx[0], idx[1]]
#     return idx, val

# def remove_self_loops_torch(edge_index, edge_weight=None):
#     mask = edge_index[0] != edge_index[1]
#     edge_index = edge_index[:, mask]
#     if edge_weight is not None:
#         edge_weight = edge_weight[mask]
#     return edge_index, edge_weight

# @dataclass
# class SimpleData:
#     x: torch.Tensor
#     edge_index: torch.Tensor
#     edge_attr: torch.Tensor
#     y: torch.Tensor
#     y_a: torch.Tensor
#     y_s: torch.Tensor
#     timeseries: torch.Tensor

# # =========================
# # Path
# # =========================
# # DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets"

# DATASET_PATH = Path(
#     "D:\OneDrive - Federation University Australia (1)\Python Projects\BrainMoe\datasets"
# )


# # =========================
# # Utils
# # =========================
# def load_csv(file_path, header=None):
#     return pd.read_csv(file_path, sep=",", header=header, engine="python")


# # =========================
# # Sparse graph construction (TOP-K)
# # =========================
# def corr_to_edge_index_topk(corr, k=10, abs_val=True):
#     """
#     corr: numpy [N, N]
#     k:    number of neighbors per node
#     """
#     A = torch.from_numpy(corr).float()
#     N = A.size(0)

#     # remove self-loops
#     A.fill_diagonal_(0)

#     score = A.abs() if abs_val else A
#     mask = torch.zeros_like(A)

#     for i in range(N):
#         idx = torch.topk(score[i], k=k).indices
#         mask[i, idx] = 1.0

#     # symmetrize
#     mask = torch.maximum(mask, mask.T)
#     A = A * mask

#     # edge_index, edge_weight = dense_to_sparse(A)
#     # edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
#     edge_index, edge_weight = dense_to_sparse_torch(A)
#     edge_index, edge_weight = remove_self_loops_torch(edge_index, edge_weight)
#     return edge_index, edge_weight


# # =========================
# # Load raw dataset
# # =========================
# def load_dataset(dataset_name, use_cache=True, topk=10):
#     cache_path = DATASET_PATH / f"{dataset_name}_cached_topk{topk}.pt"

#     if use_cache and cache_path.exists():
#         print(f"[CACHE] Loading cached {dataset_name} (topk={topk})")
#         return torch.load(cache_path, weights_only=False)

#     print(f"[LOAD] Loading raw {dataset_name} dataset...")

#     if dataset_name == "ADHD":
#         dataset_folder = DATASET_PATH / "ADHD200_packed"
#         meta_df = load_csv(DATASET_PATH / "ADHD.csv", header=0).set_index("ScanDir ID")
#         sex_key, age_key = "Gender", "Age"
#     elif dataset_name == "ABIDE":
#         dataset_folder = DATASET_PATH / "ABIDE_packed"
#         meta_df = load_csv(DATASET_PATH / "ABIDE.csv", header=0).set_index("SUB_ID")
#         sex_key, age_key = "SEX", "AGE_AT_SCAN"
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     x, y, y_a, y_s = [], [], [], []
#     edge_index, edge_weight = [], []
#     timeseries = []

#     for folder in sorted(dataset_folder.iterdir()):
#         if not folder.is_dir():
#             continue

#         sid = int(folder.name)

#         corr = load_csv(folder / "corr.csv").to_numpy()
#         ts = load_csv(folder / "timeseries.csv").to_numpy()
#         label = load_csv(folder / "label.txt").to_numpy().reshape(-1)

#         # ---- labels ----
#         if dataset_name == "ADHD":
#             label = 1 if label > 0 else 0
#             sex = meta_df.loc[sid][sex_key]
#             if pd.isna(sex):
#                 continue
#             sex = int(sex)
#         else:
#             label = int(label) - 1
#             sex = int(meta_df.loc[sid][sex_key]) - 1

#         age = meta_df.loc[sid][age_key]

#         ei, ew = corr_to_edge_index_topk(corr, k=topk)

#         x.append(corr)
#         y.append(label)
#         y_s.append(sex)
#         y_a.append(age)
#         edge_index.append(ei)
#         edge_weight.append(ew)
#         timeseries.append(ts)

#     # ---- age binning (4 classes) ----
#     y_a = np.asarray(y_a)
#     bins = np.percentile(y_a, [25, 50, 75])
#     y_a = np.digitize(y_a, bins)

#     data = dict(
#         x=x,
#         y=y,
#         y_a=y_a.tolist(),
#         y_s=y_s,
#         edge_index=edge_index,
#         edge_weight=edge_weight,
#         timeseries=timeseries,
#     )

#     if use_cache:
#         torch.save(data, cache_path)

#     return data


# # =========================
# # Build PyG Data objects
# # =========================
# def build_pyg_data(data_dict):
#     return [
#         SimpleData(
#             x=torch.from_numpy(xi).float(),
#             edge_index=ei.long(),
#             edge_attr=ew.float(),
#             y=torch.tensor(yi, dtype=torch.long),
#             y_a=torch.tensor(yai, dtype=torch.long),
#             y_s=torch.tensor(ysi, dtype=torch.long),
#             timeseries=torch.from_numpy(ti).float(),
#         )
#         for xi, yi, yai, ysi, ei, ew, ti in zip(
#             data_dict["x"],
#             data_dict["y"],
#             data_dict["y_a"],
#             data_dict["y_s"],
#             data_dict["edge_index"],
#             data_dict["edge_weight"],
#             data_dict["timeseries"],
#         )
#     ]


# # =========================
# # Create DataLoader
# # =========================
# def create_dataloader(
#     dataset_name,
#     batch_size=8,
#     seed=0,
#     use_cache=True,
#     topk=10,
#     use_5fold=False,
#     fold_id=0,
# ):
#     data = load_dataset(dataset_name, use_cache=use_cache, topk=topk)
#     labels = np.array(data["y"])

#     if not use_5fold:
#         # ===== Single split (70/20/10) =====
#         idx = np.arange(len(labels))
#         idx_tr, idx_tmp = train_test_split(
#             idx, test_size=0.3, random_state=seed, stratify=labels
#         )
#         idx_dev, idx_te = train_test_split(
#             idx_tmp,
#             test_size=1 / 3,
#             random_state=seed,
#             stratify=labels[idx_tmp],
#         )

#     else:
#         # ===== 5-fold stratified CV =====
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#         splits = list(skf.split(np.zeros(len(labels)), labels))
#         idx_tr, idx_te = splits[fold_id]

#         idx_tr, idx_dev = train_test_split(
#             idx_tr,
#             test_size=0.1,
#             random_state=seed,
#             stratify=labels[idx_tr],
#         )

#     def select(idx):
#         return {k: [v[i] for i in idx] for k, v in data.items()}

#     train_data = build_pyg_data(select(idx_tr))
#     dev_data   = build_pyg_data(select(idx_dev))
#     test_data  = build_pyg_data(select(idx_te))

#     print(f"[DATA] Train/Dev/Test = {len(train_data)}/{len(dev_data)}/{len(test_data)}")

#     def collate_list(batch):
#         return batch  # 直接返回 list[SimpleData]

#     return (
#         DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_list),
#         DataLoader(dev_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_list),
#         DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_list),
#     )

# if __name__ == "__main__":
#     train_loader, dev_loader, test_loader = create_dataloader("ABIDE", batch_size=16)
#     print(f"Train batches: {len(train_loader)}")
#     print(f"Dev batches: {len(dev_loader)}")
#     print(f"Test batches: {len(test_loader)}")

#     train_loader, dev_loader, test_loader = create_dataloader("ADHD", batch_size=16)
#     print(f"Train batches: {len(train_loader)}")
#     print(f"Dev batches: {len(dev_loader)}")
#     print(f"Test batches: {len(test_loader)}")

import pandas as pd
import numpy as np
from pathlib import Path
import torch
# from torch_geometric.utils import dense_to_sparse, remove_self_loops
# from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
# try:
#     from torch_geometric.loader import DataLoader  # PyG >= 2.0
# except Exception:
#     from torch_geometric.data import DataLoader    # PyG < 2.0

from dataclasses import dataclass
from torch.utils.data import DataLoader

def dense_to_sparse_torch(A: torch.Tensor):
    # A: [N,N]
    idx = (A != 0).nonzero(as_tuple=False).t().contiguous()  # [2,E]
    val = A[idx[0], idx[1]]
    return idx, val

def remove_self_loops_torch(edge_index, edge_weight=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    return edge_index, edge_weight

@dataclass
class SimpleData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    y_a: torch.Tensor
    y_s: torch.Tensor
    timeseries: torch.Tensor

# =========================
# Path
# =========================
# DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets"

DATASET_PATH = Path(
    "D:\OneDrive - Federation University Australia (1)\Python Projects\BrainMoe\datasets"
)


# =========================
# Utils
# =========================
def load_csv(file_path, header=None):
    return pd.read_csv(file_path, sep=",", header=header, engine="python")


# =========================
# Sparse graph construction (TOP-K)
# =========================
def corr_to_edge_index_topk(corr, k=10, abs_val=True):
    """
    corr: numpy [N, N]
    k:    number of neighbors per node
    """
    A = torch.from_numpy(corr).float()
    N = A.size(0)

    # remove self-loops
    A.fill_diagonal_(0)

    score = A.abs() if abs_val else A
    mask = torch.zeros_like(A)

    for i in range(N):
        idx = torch.topk(score[i], k=k).indices
        mask[i, idx] = 1.0

    # symmetrize
    mask = torch.maximum(mask, mask.T)
    A = A * mask

    # edge_index, edge_weight = dense_to_sparse(A)
    # edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = dense_to_sparse_torch(A)
    edge_index, edge_weight = remove_self_loops_torch(edge_index, edge_weight)
    return edge_index, edge_weight


# =========================
# Load raw dataset
# =========================
def load_dataset(dataset_name, use_cache=True, topk=10):
    cache_path = DATASET_PATH / f"{dataset_name}_cached_topk{topk}.pt"

    if use_cache and cache_path.exists():
        print(f"[CACHE] Loading cached {dataset_name} (topk={topk})")
        return torch.load(cache_path, weights_only=False)

    print(f"[LOAD] Loading raw {dataset_name} dataset...")

    if dataset_name == "ADHD":
        dataset_folder = DATASET_PATH / "ADHD200_packed"
        meta_df = load_csv(DATASET_PATH / "ADHD.csv", header=0).set_index("ScanDir ID")
        sex_key, age_key = "Gender", "Age"
    elif dataset_name == "ABIDE":
        dataset_folder = DATASET_PATH / "ABIDE_packed"
        meta_df = load_csv(DATASET_PATH / "ABIDE.csv", header=0).set_index("SUB_ID")
        sex_key, age_key = "SEX", "AGE_AT_SCAN"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    x, y, y_a, y_s = [], [], [], []
    edge_index, edge_weight = [], []
    timeseries = []

    for folder in sorted(dataset_folder.iterdir()):
        if not folder.is_dir():
            continue

        sid = int(folder.name)

        corr = load_csv(folder / "corr.csv").to_numpy()
        ts = load_csv(folder / "timeseries.csv").to_numpy()
        label = load_csv(folder / "label.txt").to_numpy().reshape(-1)

        # ---- labels ----
        if dataset_name == "ADHD":
            label = 1 if label > 0 else 0
            sex = meta_df.loc[sid][sex_key]
            if pd.isna(sex):
                continue
            sex = int(sex)
        else:
            label = int(label) - 1
            sex = int(meta_df.loc[sid][sex_key]) - 1

        age = meta_df.loc[sid][age_key]

        ei, ew = corr_to_edge_index_topk(corr, k=topk)

        x.append(corr)
        y.append(label)
        y_s.append(sex)
        y_a.append(age)
        edge_index.append(ei)
        edge_weight.append(ew)
        timeseries.append(ts)

    # ---- age binning (4 classes) ----
    y_a = np.asarray(y_a)
    bins = np.percentile(y_a, [25, 50, 75])
    y_a = np.digitize(y_a, bins)

    data = dict(
        x=x,
        y=y,
        y_a=y_a.tolist(),
        y_s=y_s,
        edge_index=edge_index,
        edge_weight=edge_weight,
        timeseries=timeseries,
    )

    if use_cache:
        torch.save(data, cache_path)

    return data


# =========================
# Build PyG Data objects
# =========================
def build_pyg_data(data_dict):
    return [
        SimpleData(
            x=torch.from_numpy(xi).float(),
            edge_index=ei.long(),
            edge_attr=ew.float(),
            y=torch.tensor(yi, dtype=torch.long),
            y_a=torch.tensor(yai, dtype=torch.long),
            y_s=torch.tensor(ysi, dtype=torch.long),
            timeseries=torch.from_numpy(ti).float(),
        )
        for xi, yi, yai, ysi, ei, ew, ti in zip(
            data_dict["x"],
            data_dict["y"],
            data_dict["y_a"],
            data_dict["y_s"],
            data_dict["edge_index"],
            data_dict["edge_weight"],
            data_dict["timeseries"],
        )
    ]


# =========================
# Create DataLoader
# =========================
def create_dataloader(
    dataset_name,
    batch_size=8,
    seed=0,
    use_cache=True,
    topk=10,
    use_5fold=False,
    fold_id=0,
):
    data = load_dataset(dataset_name, use_cache=use_cache, topk=topk)
    labels = np.array(data["y"])

    if not use_5fold:
        # ===== Single split (70/20/10) =====
        idx = np.arange(len(labels))
        idx_tr, idx_tmp = train_test_split(
            idx, test_size=0.3, random_state=seed, stratify=labels
        )
        idx_dev, idx_te = train_test_split(
            idx_tmp,
            test_size=1 / 3,
            random_state=seed,
            stratify=labels[idx_tmp],
        )

    else:
        # ===== 5-fold stratified CV =====
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(labels)), labels))
        idx_tr, idx_te = splits[fold_id]

        idx_tr, idx_dev = train_test_split(
            idx_tr,
            test_size=0.1,
            random_state=seed,
            stratify=labels[idx_tr],
        )

    def select(idx):
        return {k: [v[i] for i in idx] for k, v in data.items()}

    train_data = build_pyg_data(select(idx_tr))
    dev_data   = build_pyg_data(select(idx_dev))
    test_data  = build_pyg_data(select(idx_te))

    print(f"[DATA] Train/Dev/Test = {len(train_data)}/{len(dev_data)}/{len(test_data)}")

    def collate_list(batch):
        return batch  # 直接返回 list[SimpleData]

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_list),
        DataLoader(dev_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_list),
        DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_list),
    )

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = create_dataloader("ABIDE", batch_size=16)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")

    train_loader, dev_loader, test_loader = create_dataloader("ADHD", batch_size=16)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
