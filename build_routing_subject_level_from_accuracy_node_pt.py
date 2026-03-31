import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_network_labels_from_aal(aal_xlsx: str, n_roi: int):
    df = pd.read_excel(aal_xlsx, header=0)
    if df.shape[1] < 5:
        raise ValueError("AAL.xlsx must have at least 5 columns; column E is required.")
    labels = (
        df.iloc[:n_roi, 4]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .tolist()
    )
    if len(labels) != n_roi:
        raise ValueError(f"Expected {n_roi} network labels, got {len(labels)}")
    return labels


def find_fold_files(results_dir: str, dataset: str):
    """
    Only use scheme A:
      sel-accuracy_thr-accuracy_fold*_subject_node_expert_usage.pt
    """
    results_dir = Path(results_dir)
    pattern = re.compile(
        rf"^{re.escape(dataset)}_.*sel-accuracy_thr-accuracy_fold(\d+)_subject_node_expert_usage\.pt$"
    )

    matched = []
    for p in results_dir.glob("*.pt"):
        m = pattern.match(p.name)
        if m:
            fold = int(m.group(1))
            matched.append((fold, p))

    matched = sorted(matched, key=lambda x: x[0])
    if not matched:
        raise FileNotFoundError(
            f"No scheme-A node usage pt files found under {results_dir}"
        )
    return matched


def build_subject_level_csv(
    matched_files,
    aal_xlsx,
    out_csv,
    dataset,
    use_soft=True,
):
    rows = []
    global_subject_counter = 0

    all_networks_seen = None
    expert_names_ref = None
    total_correct = 0
    fold_group_counts = {}

    for fold, pt_path in matched_files:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)

        usage_key = "soft_node_usage" if use_soft else "hard_node_usage"
        if usage_key not in obj:
            raise KeyError(f"{usage_key} not found in {pt_path}")

        node_usage = obj[usage_key]   # [N, ROI, E]
        labels = obj["labels"]        # [N]
        preds = obj["preds"]          # [N]
        probs = obj["probs"]          # [N]
        roi_names = obj["roi_names"]
        expert_names = list(obj["expert_names"])

        if expert_names_ref is None:
            expert_names_ref = expert_names
        else:
            if expert_names_ref != expert_names:
                raise ValueError(
                    f"Expert names mismatch in {pt_path}: {expert_names} vs {expert_names_ref}"
                )

        if torch.is_tensor(node_usage):
            node_usage = node_usage.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()

        num_subjects, num_roi, num_experts = node_usage.shape
        network_labels = load_network_labels_from_aal(aal_xlsx, num_roi)

        network_names = list(dict.fromkeys(network_labels))
        if all_networks_seen is None:
            all_networks_seen = network_names
        else:
            if network_names != all_networks_seen:
                raise ValueError(
                    f"Network order mismatch in {pt_path}: {network_names} vs {all_networks_seen}"
                )

        network_to_indices = {}
        for i, net in enumerate(network_labels):
            network_to_indices.setdefault(net, []).append(i)

        fold_group_counts[fold] = {"HC": 0, "ADHD": 0}

        for s in range(num_subjects):
            label = int(labels[s])
            pred = int(preds[s])
            prob_1 = float(probs[s])
            correct = int(label == pred)

            if correct != 1:
                continue

            group = "HC" if label == 0 else "ADHD"
            fold_group_counts[fold][group] += 1
            total_correct += 1

            subj_id = f"accsel_fold{fold}_subj{global_subject_counter:05d}"
            global_subject_counter += 1

            subj_node_usage = node_usage[s]  # [ROI, E]

            for net in network_names:
                roi_idx = network_to_indices[net]
                net_mean = subj_node_usage[roi_idx, :].mean(axis=0)   # [E]

                for e_idx, expert in enumerate(expert_names):
                    rows.append({
                        "subject_id": subj_id,
                        "fold": fold,
                        "group": group,
                        "correct": 1,
                        "network": net,
                        "expert": expert,
                        "mean_prob": float(net_mean[e_idx]),
                        "label": label,
                        "pred": pred,
                        "prob_1": prob_1,
                        "source_file": pt_path.name,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"[Saved] {out_csv}")
    print(df.head())
    print("\n[Summary]")
    print("Unique subjects:", df['subject_id'].nunique())
    print("Total rows     :", len(df))
    print("Networks       :", all_networks_seen)
    print("Experts        :", expert_names_ref)

    print("\n[Fold x Group counts]")
    fg = (
        df[["subject_id", "fold", "group"]]
        .drop_duplicates()
        .groupby(["fold", "group"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(fg)

    print("\n[Duplicate check]")
    dup = df.duplicated(subset=["subject_id", "network", "expert"]).sum()
    print("Duplicated (subject_id, network, expert) rows:", dup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ADHD")
    parser.add_argument("--aal_xlsx", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--use_soft", action="store_true")
    parser.add_argument("--use_hard", action="store_true")
    args = parser.parse_args()

    if args.use_soft and args.use_hard:
        raise ValueError("Use only one of --use_soft / --use_hard")
    use_soft = True
    if args.use_hard:
        use_soft = False

    matched = find_fold_files(args.results_dir, args.dataset)

    print("[Matched files]")
    for fold, p in matched:
        print(f"fold {fold}: {p}")

    build_subject_level_csv(
        matched_files=matched,
        aal_xlsx=args.aal_xlsx,
        out_csv=args.out_csv,
        dataset=args.dataset,
        use_soft=use_soft,
    )


if __name__ == "__main__":
    main()