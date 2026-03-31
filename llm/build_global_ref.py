# scripts/build_global_ref.py
import argparse
from pathlib import Path
import numpy as np
import torch

from src.abnormal_report import fit_control_reference

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_pt", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--use_fisher_z", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--sigma_floor_q", type=float, default=25.0)
    ap.add_argument("--sigma_floor_scale", type=float, default=1.0)
    args = ap.parse_args()

    cache = torch.load(args.cached_pt, weights_only=False)
    x_list = cache["x_list"]
    y_list = cache["y_list"]

    # 用“全体 subjects 的 index”，函数内部只会取 y==0 的 controls
    all_idx = list(range(len(y_list)))

    ref = fit_control_reference(
        x_list=x_list,
        y_list=y_list,
        train_indices=all_idx,
        eps=args.eps,
        use_fisher_z=args.use_fisher_z,
        sigma_floor_q=args.sigma_floor_q,
        sigma_floor_scale=args.sigma_floor_scale,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, out_path)
    print("[OK] saved global_ref:", out_path)

if __name__ == "__main__":
    main()