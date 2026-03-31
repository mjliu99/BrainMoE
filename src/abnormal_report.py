import json
from pathlib import Path
import numpy as np


def load_aal_names(aal_path: str):
    """
    Returns:
      names: list[str] length N, where names[i] is ROI name for index i (0-based).
    Supports .xlsx or .csv. If .xls, ask user to convert.
    """
    p = Path(aal_path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
    elif suffix == ".xlsx":
        import pandas as pd
        df = pd.read_excel(p)
    elif suffix == ".xls":
        raise RuntimeError(
            f"[AAL] {aal_path} is .xls (old Excel). Please convert it to .xlsx or .csv, "
            "then update --aal_nodes_path."
        )
    else:
        raise RuntimeError(f"[AAL] Unsupported file type: {suffix}")

    # Try common column names; fallback to first 2 columns.
    cols = [c.lower() for c in df.columns]
    id_col = None
    name_col = None
    for c in df.columns:
        cl = c.lower()
        if id_col is None and ("index" in cl or "id" in cl or "roi" in cl or "label" in cl):
            id_col = c
        if name_col is None and ("name" in cl or "region" in cl or "roi" in cl):
            name_col = c

    if id_col is None or name_col is None:
        id_col, name_col = df.columns[0], df.columns[1]

    ids = df[id_col].to_numpy()
    names_raw = df[name_col].astype(str).to_numpy()

    # Convert ids to 0-based contiguous indices if needed
    # If ids are 1..N -> shift to 0..N-1
    ids_int = []
    for v in ids:
        try:
            ids_int.append(int(v))
        except Exception:
            ids_int.append(None)

    # If ids look like 1..N
    if all(v is not None for v in ids_int) and min(ids_int) == 1:
        ids_int = [v - 1 for v in ids_int]

    n = max([v for v in ids_int if v is not None] + [len(names_raw) - 1]) + 1
    names = [f"ROI_{i}" for i in range(n)]
    for idx, nm in zip(ids_int, names_raw):
        if idx is None:
            continue
        if 0 <= idx < n:
            names[idx] = nm
    return names


def fit_control_reference(corr_list, y_list, train_idx, eps=1e-6):
    """
    corr_list: list[np.ndarray] each [N,N]
    y_list: list/int labels (0/1)
    train_idx: indices for training set
    Reference is computed from TRAIN & CONTROL only (y=0).
    """
    ctrl_idx = [i for i in train_idx if int(y_list[i]) == 0]
    if len(ctrl_idx) < 5:
        # fallback: use all train if too few controls
        ctrl_idx = list(train_idx)

    stack = np.stack([corr_list[i] for i in ctrl_idx], axis=0)  # [M,N,N]
    mu = stack.mean(axis=0)
    sd = stack.std(axis=0)
    sd = np.maximum(sd, eps)
    return {"mu": mu, "sd": sd}


def zscore_corr(corr, ref):
    return (corr - ref["mu"]) / ref["sd"]


def roi_strength(corr):
    # sum of correlations (can also use sum(abs(corr)))
    return corr.sum(axis=1)


def build_abnormal_report(
    corr,
    ref,
    roi_names,
    topk_edges=20,
    topk_rois=10,
):
    """
    Build a compact report dict (later converted to prompt text).
    """
    z = zscore_corr(corr, ref)
    N = corr.shape[0]

    # Edges: use upper triangle only
    iu, ju = np.triu_indices(N, k=1)
    z_edges = z[iu, ju]
    corr_edges = corr[iu, ju]
    order = np.argsort(np.abs(z_edges))[::-1][:topk_edges]

    edges = []
    for idx in order:
        i = int(iu[idx]); j = int(ju[idx])
        zz = float(z_edges[idx])
        cc = float(corr_edges[idx])
        edges.append({
            "roi1": roi_names[i],
            "roi2": roi_names[j],
            "corr": cc,
            "z": zz,
            "direction": "increased" if zz > 0 else "decreased"
        })

    # ROIs by strength z-score
    s = roi_strength(corr)
    # reference strength from ref mu
    s_ref = roi_strength(ref["mu"])
    s_sd = np.maximum(roi_strength(ref["sd"]), 1e-6)
    z_roi = (s - s_ref) / s_sd
    order_roi = np.argsort(np.abs(z_roi))[::-1][:topk_rois]
    rois = []
    for i in order_roi:
        rois.append({
            "roi": roi_names[int(i)],
            "strength": float(s[int(i)]),
            "z": float(z_roi[int(i)]),
            "direction": "increased" if z_roi[int(i)] > 0 else "decreased"
        })

    # global stats
    mean_abs = float(np.mean(np.abs(corr)))
    mean_abs_ref = float(np.mean(np.abs(ref["mu"])))
    mean_abs_sd = float(np.mean(np.abs(ref["sd"])))
    z_mean_abs = (mean_abs - mean_abs_ref) / max(mean_abs_sd, 1e-6)

    report = {
        "global": {
            "mean_abs_corr": mean_abs,
            "mean_abs_corr_z": float(z_mean_abs),
        },
        "top_rois": rois,
        "top_edges": edges
    }
    return report


def report_to_prompt_text(report, task_name="ADHD vs Control"):
    """
    Deterministic, compact prompt text.
    """
    lines = []
    lines.append(f"Task: {task_name}.")
    lines.append("All abnormalities are z-scores computed relative to TRAIN CONTROL subjects in this CV fold.")
    g = report["global"]
    lines.append(f"Global: mean_abs_corr_z={g['mean_abs_corr_z']:.3f} (mean_abs_corr={g['mean_abs_corr']:.4f}).")

    lines.append("Top abnormal ROIs by strength z (roi, z, direction):")
    for k, r in enumerate(report["top_rois"], start=1):
        lines.append(f"{k}) {r['roi']}, z={r['z']:.3f}, {r['direction']}")

    lines.append("Top abnormal edges by |z| (roi1-roi2, corr, z, direction):")
    for k, e in enumerate(report["top_edges"], start=1):
        lines.append(f"{k}) {e['roi1']} — {e['roi2']}, corr={e['corr']:.4f}, z={e['z']:.3f}, {e['direction']}")

    lines.append(
        "Output JSON only with keys: risk_score (0-1), predicted_label, evidence_edges, evidence_rois, short_reasoning."
    )
    return "\n".join(lines)