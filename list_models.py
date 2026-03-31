from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.abnormal_report import (
    load_aal_names,
    fit_control_reference,
    build_abnormal_report,
    report_to_prompt_text,
)

from llm.llm_utils import process_rows, all_text_to_vector


def _to_numpy_corr(corr):
    if isinstance(corr, torch.Tensor):
        return corr.detach().cpu().numpy()
    return corr


def _save_pt(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _load_pt(path: Path):
    return torch.load(path, weights_only=False)


def make_tar_gz(src_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return out_path


def build_stage1_payload_isdt_demo(isdt_json: dict, subject_json: dict):
    stage1_isdt = dict(isdt_json)
    stage1_isdt["_stage"] = "A_router"

    stage1_subject = dict(subject_json)
    stage1_subject["_stage"] = "A_router"

    return stage1_isdt, stage1_subject


def build_stage2_payload_isdt_abnormal_stage1(
    isdt_json,
    abnormal_prompt,
    abnormal_report,
    stage1_llm_out,
    subject_json,
):
    merged_isdt = dict(isdt_json)
    merged_isdt["_stage"] = "B_final"
    merged_isdt["graph_abnormal_prompt"] = abnormal_prompt
    merged_isdt["graph_abnormal_report"] = abnormal_report
    merged_isdt["stage1_router_output"] = stage1_llm_out

    subj = dict(subject_json)
    subj["_stage"] = "B_final"

    return merged_isdt, subj


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="ADHD")
    ap.add_argument("--cached_pt", required=True)
    ap.add_argument("--aal_xlsx", required=True)

    ap.add_argument("--out_dir", default="data/abnormal_llm_cache")
    ap.add_argument("--llm_workers", type=int, default=1)
    ap.add_argument("--emb_workers", type=int, default=2)

    ap.add_argument("--topk_edges", type=int, default=20)
    ap.add_argument("--topk_rois", type=int, default=10)
    ap.add_argument("--max_samples", type=int, default=None, help="Only run first N subjects for quick testing")
    ap.add_argument("--global_mode", action="store_true")
    ap.add_argument("--reuse_fold1", action="store_true")

    args = ap.parse_args()

    cache = torch.load(args.cached_pt, weights_only=False)

    x_list = cache["x_list"]
    y_list = cache["y_list"]
    labels = np.array(y_list)

    subjects_json_list = cache["subjects_json_list"]
    isdt_json_list = cache["isdt_json_list"]

    if args.max_samples is not None:
        n = min(int(args.max_samples), len(x_list))
        x_list = x_list[:n]
        y_list = y_list[:n]
        labels = labels[:n]
        subjects_json_list = subjects_json_list[:n]
        isdt_json_list = isdt_json_list[:n]
        print(f"[TEST MODE] Only running first {n} subjects")

    roi_names = load_aal_names(args.aal_xlsx)

    out_root = Path(args.out_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    task_name = "ADHD vs Control" if args.dataset == "ADHD" else "ASD vs Control"

    if args.global_mode:
        global_dir = out_root / "global"
        global_dir.mkdir(exist_ok=True)

        ref_path = out_root / "global_ref.pt"

        if not ref_path.exists():
            raise RuntimeError("Please generate global_ref.pt first")

        ref = _load_pt(ref_path)

        N = len(x_list)
        print("Total subjects:", N)

        # -------------------------
        # Outputs to cache
        # -------------------------
        stage1_input_by_idx = {}
        stage1_llm_by_idx = {}
        stage1_emb_by_idx = {}

        abnormal_by_idx = {}

        stage2_input_by_idx = {}
        stage2_llm_by_idx = {}
        stage2_emb_by_idx = {}


        # -------------------------
        # Load existing global cache if present
        # -------------------------
        if (global_dir / "stage1_input_by_idx.pt").exists():
            stage1_input_by_idx = _load_pt(global_dir / "stage1_input_by_idx.pt")
        if (global_dir / "stage1_llm_by_idx.pt").exists():
            stage1_llm_by_idx = _load_pt(global_dir / "stage1_llm_by_idx.pt")
        if (global_dir / "stage1_emb_by_idx.pt").exists():
            stage1_emb_by_idx = _load_pt(global_dir / "stage1_emb_by_idx.pt")

        if (global_dir / "abnormal_by_idx.pt").exists():
            abnormal_by_idx = _load_pt(global_dir / "abnormal_by_idx.pt")

        if (global_dir / "stage2_input_by_idx.pt").exists():
            stage2_input_by_idx = _load_pt(global_dir / "stage2_input_by_idx.pt")
        if (global_dir / "stage2_llm_by_idx.pt").exists():
            stage2_llm_by_idx = _load_pt(global_dir / "stage2_llm_by_idx.pt")
        if (global_dir / "stage2_emb_by_idx.pt").exists():
            stage2_emb_by_idx = _load_pt(global_dir / "stage2_emb_by_idx.pt")

        print(
            "Loaded existing cache:",
            len(stage1_llm_by_idx),
            len(stage1_emb_by_idx),
            len(stage2_llm_by_idx),
            len(stage2_emb_by_idx),
        )
        # -------------------------
        # Optional fold1 reuse
        # -------------------------
        if args.reuse_fold1:
            fold1 = out_root / "fold1"

            if (fold1 / "idx_llm.pt").exists():
                idx = _load_pt(fold1 / "idx_llm.pt")

                s1 = _load_pt(fold1 / "stage1_llm_outputs.pt")
                s1e = _load_pt(fold1 / "stage1_embeddings.pt")

                s2 = _load_pt(fold1 / "stage2_llm_outputs.pt")
                s2e = _load_pt(fold1 / "stage2_embeddings.pt")

                # optional inputs / abnormal cache if they exist
                s1_in = _load_pt(fold1 / "stage1_inputs.pt") if (fold1 / "stage1_inputs.pt").exists() else None
                s2_in = _load_pt(fold1 / "stage2_inputs.pt") if (fold1 / "stage2_inputs.pt").exists() else None
                abn_in = _load_pt(fold1 / "abnormal_reports.pt") if (fold1 / "abnormal_reports.pt").exists() else None

                for split in ["train", "val", "test"]:
                    ids = idx[f"{split}_idx_llm"]

                    for j, i in enumerate(ids):
                        stage1_llm_by_idx[i] = s1[split][j]
                        stage1_emb_by_idx[i] = s1e[split][j]

                        stage2_llm_by_idx[i] = s2[split][j]
                        stage2_emb_by_idx[i] = s2e[split][j]

                        if s1_in is not None:
                            stage1_input_by_idx[i] = s1_in[split][j]
                        if s2_in is not None:
                            stage2_input_by_idx[i] = s2_in[split][j]
                        if abn_in is not None:
                            abnormal_by_idx[i] = abn_in[split][j]

                print("Reuse fold1 samples:", len(stage2_llm_by_idx))

        # -------------------------
        # Stage1 generation
        # -------------------------
        # missing = [i for i in range(N) if i not in stage1_llm_by_idx]
        missing = [i for i in range(N) if (i not in stage1_llm_by_idx) or (stage1_llm_by_idx[i] is None)]
        print("Stage1 missing:", len(missing))

        if missing:
            isdt_in = []
            subj_in = []

            for i in missing:
                a, b = build_stage1_payload_isdt_demo(
                    isdt_json_list[i],
                    subjects_json_list[i],
                )

                stage1_input_by_idx[i] = {
                    "isdt_json": a,
                    "subject_json": b,
                }

                isdt_in.append(a)
                subj_in.append(b)

            out = process_rows(
                isdt_in,
                subj_in,
                rows_to_process=len(subj_in),
                max_workers=args.llm_workers,
            )

            for j, i in enumerate(missing):
                stage1_llm_by_idx[i] = out[j]

        print("Stage1 done")

        # -------------------------
        # Stage1 embeddings
        # -------------------------
        #miss = [i for i in range(N) if i not in stage1_emb_by_idx]
        miss = [i for i in range(N) if (i not in stage1_emb_by_idx) or (stage1_emb_by_idx[i] is None)]
        if miss:
            vec = all_text_to_vector(
                [stage1_llm_by_idx[i] for i in miss],
                max_workers=args.emb_workers,
            )

            for j, i in enumerate(miss):
                stage1_emb_by_idx[i] = vec[j]

        print("Stage1 embedding done")

        # -------------------------
        # Abnormal report generation
        # -------------------------
        missing_abn = [i for i in range(N) if i not in abnormal_by_idx]

        if missing_abn:
            for i in missing_abn:
                corr_np = _to_numpy_corr(x_list[i])

                rep = build_abnormal_report(
                    corr=corr_np,
                    ref=ref,
                    roi_names=roi_names,
                    topk_edges=args.topk_edges,
                    topk_rois=args.topk_rois,
                )

                abnormal_by_idx[i] = rep

        print("Abnormal reports done")

        # -------------------------
        # Stage2 generation
        # -------------------------
        # missing = [i for i in range(N) if i not in stage2_llm_by_idx]
        missing = [i for i in range(N) if (i not in stage2_llm_by_idx) or (stage2_llm_by_idx[i] is None)]
        print("Stage2 missing:", len(missing))

        if missing:
            isdt_in = []
            subj_in = []

            for i in missing:
                abn = abnormal_by_idx[i]
                prompt = report_to_prompt_text(abn, task_name=task_name)

                merged, subj = build_stage2_payload_isdt_abnormal_stage1(
                    isdt_json_list[i],
                    prompt,
                    abn,
                    stage1_llm_by_idx[i],
                    subjects_json_list[i],
                )

                stage2_input_by_idx[i] = {
                    "isdt_json": merged,
                    "subject_json": subj,
                }

                isdt_in.append(merged)
                subj_in.append(subj)

            out = process_rows(
                isdt_in,
                subj_in,
                rows_to_process=len(subj_in),
                max_workers=args.llm_workers,
            )

            for j, i in enumerate(missing):
                stage2_llm_by_idx[i] = out[j]

        print("Stage2 done")

        # -------------------------
        # Stage2 embeddings
        # -------------------------
        # miss = [i for i in range(N) if i not in stage2_emb_by_idx]
        
        miss = [i for i in range(N) if (i not in stage2_emb_by_idx) or (stage2_emb_by_idx[i] is None)]
        if miss:
            vec = all_text_to_vector(
                [stage2_llm_by_idx[i] for i in miss],
                max_workers=args.emb_workers,
            )

            for j, i in enumerate(miss):
                stage2_emb_by_idx[i] = vec[j]

        print("Stage2 embedding done")

        # -------------------------
        # Save all caches
        # -------------------------
        _save_pt(stage1_input_by_idx, global_dir / "stage1_input_by_idx.pt")
        _save_pt(stage1_llm_by_idx, global_dir / "stage1_llm_by_idx.pt")
        _save_pt(stage1_emb_by_idx, global_dir / "stage1_emb_by_idx.pt")

        _save_pt(abnormal_by_idx, global_dir / "abnormal_by_idx.pt")

        _save_pt(stage2_input_by_idx, global_dir / "stage2_input_by_idx.pt")
        _save_pt(stage2_llm_by_idx, global_dir / "stage2_llm_by_idx.pt")
        _save_pt(stage2_emb_by_idx, global_dir / "stage2_emb_by_idx.pt")

        print("GLOBAL CACHE DONE")
        return


if __name__ == "__main__":
    main()