import json
from pathlib import Path
import torch
import pandas as pd
from torch_geometric.utils import dense_to_sparse, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

DATASET_PATH = Path("/root/autodl-tmp/BrainMoE-02/datasets") # CHANGE INTO YOUR OWN PATH

def corr_to_edge_index(corr, thresh=0.0):
    A = torch.from_numpy(corr).float()

    # drop self-loops
    try:
        A.fill_diagonal_(0)
    except AttributeError:
        idx = torch.arange(A.size(0))
        A[idx, idx] = 0

    if thresh > 0:
        A = torch.where(A.abs() >= thresh, A, torch.zeros_like(A))
    edge_index, edge_weight = dense_to_sparse(A)  # [2, E], [E]
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    return edge_index, edge_weight

def load_csv(file_path, header=None):
    return pd.read_csv(file_path, sep=",", header=header, engine="python")

def load_dataset(dataset_name):
    
    print(f"Loading raw {dataset_name} dataset...")
    if dataset_name == "ADHD":
        dataset_folder = Path.joinpath(DATASET_PATH, "ADHD200_packed")
        meta_data_df = load_csv(Path.joinpath(DATASET_PATH, "ADHD.csv"), header=0)
        meta_data_df = meta_data_df.set_index("ScanDir ID")
        sex_key = "Gender"
        age_key = "Age"
        handedness_key = "Handedness" #left = 0, right = 1
        viq = 'Verbal IQ'
        piq = "Performance IQ"
        fiq = "Full4 IQ"
        # adhd_index = "ADHD Index"
        # adhd_measure = "ADHD Measure"
        # innatentive = "Inattentive"
        # hyperactive = "Hyper/Impulsive"

    elif dataset_name == "ABIDE":
        dataset_folder = Path.joinpath(DATASET_PATH, "ABIDE_packed")
        meta_data_df = load_csv(Path.joinpath(DATASET_PATH, "ABIDE.csv"), header=0)
        meta_data_df = meta_data_df.set_index("SUB_ID")
        sex_key = "SEX"
        age_key = "AGE_AT_SCAN"
        handedness_key = "HANDEDNESS_CATEGORY"
        viq = 'VIQ'
        piq = "PIQ"
        fiq = "FIQ"
    else:
        raise ValueError("Unknown dataset name")
    

    id_list = []
    x_list = []
    y_list = []
    sex_list = []
    age_list = []
    handedness_list = []
    viq_list = []   
    piq_list = []
    fiq_list = []
    edge_index_list = []
    edge_weight_list = []
    timeseries_list = []
    ts_len_list = []

    for folder in dataset_folder.iterdir():
        if folder.is_dir():
            corr = load_csv(folder/"corr.csv").to_numpy()
            label = load_csv(folder / "label.txt").to_numpy().reshape(-1)
            timeseries = load_csv(folder / "timeseries.csv").to_numpy()
            edge_index, edge_weight = corr_to_edge_index(corr, 0)

            if dataset_name == "ADHD":
                label = 1 if label > 0 else 0
                sex = "male" if meta_data_df.loc[int(folder.name), sex_key] == 1 else ("female" if meta_data_df.loc[int(folder.name), sex_key] == 0 else "undefined")
                handedness = "Right" if meta_data_df.loc[int(folder.name), handedness_key] == 1 else ("Left" if meta_data_df.loc[int(folder.name), handedness_key] == 0 else ("Undefined" if "NA" in str(meta_data_df.loc[int(folder.name), handedness_key]) else "Mixed"))

            elif dataset_name == "ABIDE":
                label = label - 1
                sex = "male" if meta_data_df.loc[int(folder.name), sex_key] == 1 else ("female" if meta_data_df.loc[int(folder.name), sex_key] == 2 else "undefined")
                handedness = "Right" if meta_data_df.loc[int(folder.name), handedness_key] == "R" else ("Left" if meta_data_df.loc[int(folder.name), handedness_key] == "L" else ("Mixed" if meta_data_df.loc[int(folder.name), handedness_key] == "Ambi" or "Mixed" else "undefined"))
            
            age = meta_data_df.loc[int(folder.name), age_key]
            viq_score = meta_data_df.loc[int(folder.name), viq]
            piq_score = meta_data_df.loc[int(folder.name), piq]
            fiq_score = meta_data_df.loc[int(folder.name), fiq]

            id_list.append(folder.name)
            x_list.append(corr)
            y_list.append(label)
            sex_list.append(sex)
            age_list.append(age)
            handedness_list.append(handedness)
            viq_list.append(viq_score)
            piq_list.append(piq_score)
            fiq_list.append(fiq_score)
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            timeseries_list.append(timeseries)
            ts_len_list.append(timeseries.shape[0])
    
    return (
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
    )

def get_subjects(dataset_name):
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
    ) = load_dataset(dataset_name)

    task_json = "ADHD" if dataset_name == "ADHD" else "ASD"   
    node_identity_json = json.load(open(Path("jsons").joinpath(f"node_identity.json"), "r"))
    expert_identity_json = json.load(open(Path("jsons").joinpath(f"expert_identity.json"), "r"))

    if dataset_name == "ADHD":
        knowledge_summary_json = json.load(open(Path("jsons").joinpath(f"ADHD_knowledge_summary.json"), "r"))
    elif dataset_name == "ABIDE":
        knowledge_summary_json = json.load(open(Path("jsons").joinpath(f"ABIDE_knowledge_summary.json"), "r")) 

    problem_prompt_json = json.load(open(Path("jsons").joinpath(f"problem_prompt.json"), "r"))

    subjects_json_list = []
    for i in range(len(x_list)):
        demographic_info_json = {
                "gender": sex_list[i],
                "age": age_list[i],
                "handedness": handedness_list[i],
                "verbal_iq": viq_list[i],
                "performance_iq": piq_list[i],
                "full_scale_iq": fiq_list[i]
                # Optional clinical measures (uncomment if available):
                # "inattentive": inattentive,
                # "hyperactive": hyperactive,
                # "adhd_index": adhd_index,
                # "adhd_measure": adhd_measure,
            }
        subject_json = {
            "detection_task": task_json,
            "phenotype_summary": demographic_info_json,
            "node_identity": node_identity_json,
            "expert_identity": expert_identity_json,
            "knowledge_summary": knowledge_summary_json,
            "problem_prompt": problem_prompt_json,
        }

        subjects_json_list.append(subject_json)
    
    return (
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
    )

def data_loader(x_list, y_list, edge_index_list, edge_weight_list, llm_vectors, batch_size):

    all_data = []
    for x, y, edge_index, edge_weight, llm_vec in zip(
        x_list, y_list, edge_index_list, edge_weight_list, llm_vectors
    ):
        x_t = torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()
        edge_index_t = edge_index.long() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        edge_attr_t = edge_weight.float() if isinstance(edge_weight, torch.Tensor) else torch.tensor(edge_weight, dtype=torch.float)
        llm_t = llm_vec.float() if isinstance(llm_vec, torch.Tensor) else torch.tensor(llm_vec, dtype=torch.float)
        llm_t = llm_t.view(-1)
        llm_node = llm_t.unsqueeze(0).repeat(x_t.size(0), 1)

        all_data.append(
            Data(
                x=x_t,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y=torch.tensor([int(y)], dtype=torch.long),
                llm_embeddings=llm_node,
            )
        )

    labels = np.array(y_list)
    idx = np.arange(len(all_data))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.3, random_state=0, stratify=labels)
    idx_dev, idx_te = train_test_split(idx_tmp, test_size=1 / 3, random_state=0, stratify=labels[idx_tmp])

    train_data = [all_data[i] for i in idx_tr]
    val_data = [all_data[i] for i in idx_dev]
    test_data = [all_data[i] for i in idx_te]

    print(f"[DATA] Train/Dev/Test = {len(train_data)}/{len(val_data)}/{len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

