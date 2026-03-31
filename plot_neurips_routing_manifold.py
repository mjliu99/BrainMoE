import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOP_PAIRS = [
    ("CBL", "cheb"),
    ("VN",  "cheb"),
    ("CBL", "mlp"),
    ("VN",  "mlp"),
    ("SMN", "cheb"),
    ("DAN", "cheb"),
    ("SMN", "mlp"),
    ("DAN", "mlp"),
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def q_to_star(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def build_matrix(df, networks, experts, value_col):
    piv = df.pivot(index="network", columns="expert", values=value_col)
    piv = piv.reindex(index=networks, columns=experts)
    return piv.values


def load_merged_df(result_dir):
    overall_df = pd.read_csv(Path(result_dir) / "overall_shift.csv")
    sig_df = pd.read_csv(Path(result_dir) / "mannwhitney_fdr.csv")
    boot_df = pd.read_csv(Path(result_dir) / "bootstrap_ci.csv")

    merged = overall_df.merge(
        sig_df[["network", "expert", "q_value"]],
        on=["network", "expert"],
        how="left",
    ).merge(
        boot_df[["network", "expert", "ci_low", "ci_high"]],
        on=["network", "expert"],
        how="left",
    )

    return overall_df, sig_df, boot_df, merged


# =========================================================
# Figure 1: significance heatmap
# =========================================================
def plot_significance_heatmap(
    merged_df,
    networks,
    experts,
    save_path,
    fontsize=14,
    annotate=True,
):
    shift_mat = build_matrix(merged_df, networks, experts, "shift")
    q_mat = build_matrix(merged_df, networks, experts, "q_value")

    vmax = np.nanmax(np.abs(shift_mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=300)

    im = ax.imshow(
        shift_mat,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(experts)))
    ax.set_xticklabels(experts, fontsize=fontsize)
    ax.set_yticks(np.arange(len(networks)))
    ax.set_yticklabels(networks, fontsize=fontsize)

    ax.set_xlabel("Expert", fontsize=fontsize + 1)
    ax.set_ylabel("Network", fontsize=fontsize + 1)
    ax.set_title("Routing shift (ADHD - HC)", fontsize=fontsize + 4, pad=10)

    ax.set_xticks(np.arange(-0.5, len(experts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(networks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(len(networks)):
            for j in range(len(experts)):
                v = shift_mat[i, j]
                q = q_mat[i, j]
                star = q_to_star(q)
                txt = f"{v:+.3f}{star}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=fontsize - 3,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize - 1)
    cbar.set_label("ADHD - HC", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Figure 2: ranking across all pairs (FROM overall_shift.csv)
# =========================================================
def plot_routing_shift_ranking_from_df(
    overall_df,
    save_path,
    fontsize=12,
    ranking_metric="abs_diff",
    label_extremes_only=True,
):
    df = overall_df.copy()
    df["label"] = df["network"].astype(str) + "-" + df["expert"].astype(str)
    df["abs_shift"] = df["shift"].abs()

    if ranking_metric == "signed_diff":
        df = df.sort_values("shift", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("abs_shift", ascending=False).reset_index(drop=True)

    labels = df["label"].tolist()[::-1]
    vals = df["shift"].to_numpy(dtype=float)[::-1]

    vmax = np.max(np.abs(vals)) + 1e-12
    colors = plt.cm.coolwarm((vals + vmax) / (2 * vmax))

    fig_h = max(7.5, 0.26 * len(labels))
    fig, ax = plt.subplots(figsize=(9.6, fig_h), dpi=300)

    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize - 1)
    ax.tick_params(axis="x", labelsize=fontsize - 1)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_xlabel("Mean usage difference (ADHD - HC)", fontsize=fontsize)
    ax.set_title("Routing shift ranking across all network-expert pairs", fontsize=fontsize + 3)

    # -------- key fix 1: enlarge x-axis margins --------
    xmin_data = float(np.min(vals))
    xmax_data = float(np.max(vals))
    xpad = max(0.006, 0.10 * max(abs(xmin_data), abs(xmax_data)))
    ax.set_xlim(xmin_data - xpad, xmax_data + xpad)

    # -------- key fix 2: label offset --------
    pos_offset = 0.004
    neg_offset = 0.004

    # -------- key fix 3: avoid left labels colliding with ytick labels --------
    xmin, xmax = ax.get_xlim()
    left_safe = xmin + 0.010
    right_safe = xmax - 0.010

    if label_extremes_only:
        n = len(vals)
        extreme_idx = set(list(range(min(6, n))) + list(range(max(0, n - 6), n)))
        label_indices = extreme_idx
    else:
        label_indices = set(range(len(vals)))

    for i, v in enumerate(vals):
        if i not in label_indices:
            continue

        if v >= 0:
            tx = min(v + pos_offset, right_safe)
            ha = "left"
            txt = f"{v:.3f}"
        else:
            tx = max(v - neg_offset, left_safe)
            ha = "right"
            txt = f"{v:.3f}"

        ax.text(
            tx,
            i,
            txt,
            va="center",
            ha=ha,
            fontsize=fontsize - 2,
            clip_on=True,
        )

    ax.grid(True, axis="x", alpha=0.22)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# Figure 3: top pairs with bootstrap CI
# =========================================================
def plot_top_ci(
    merged_df,
    save_path,
    fontsize=14,
):
    rows = []
    for net, exp in TOP_PAIRS:
        sub = merged_df[(merged_df["network"] == net) & (merged_df["expert"] == exp)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append({
            "pair": f"{net}-{exp}",
            "shift": float(r["shift"]),
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "q_value": float(r["q_value"]) if not pd.isna(r["q_value"]) else np.nan,
        })

    df = pd.DataFrame(rows)
    df["abs_shift"] = df["shift"].abs()
    df = df.sort_values("abs_shift", ascending=True).reset_index(drop=True)

    y = np.arange(len(df)) * 1.25
    x = df["shift"].values
    xerr_low = x - df["ci_low"].values
    xerr_high = df["ci_high"].values - x

    xmin = float(np.min(df["ci_low"]) - 0.025)
    xmax = float(np.max(df["ci_high"]) + 0.025)

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=300)

    ax.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        capsize=5,
        linewidth=1.8,
        markersize=7,
    )
    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["pair"].tolist(), fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 1)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Mean usage difference (ADHD - HC)", fontsize=fontsize + 1)
    ax.set_title("Top routing shifts with 95% bootstrap CI", fontsize=fontsize + 4, pad=10)

    ax.grid(True, axis="x", alpha=0.22)

    for yi, xi, lo, hi in zip(y, x, df["ci_low"].values, df["ci_high"].values):
        if xi >= 0:
            tx = xi + 0.004
            ha = "left"
        else:
            tx = xi - 0.004
            ha = "right"

        label = f"{xi:+.3f}\n[{lo:+.3f}, {hi:+.3f}]"
        ax.text(
            tx,
            yi + 0.08,
            label,
            fontsize=fontsize - 3,
            ha=ha,
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Table 1: top pairs summary
# =========================================================
def build_top_table(merged_df, save_csv):
    rows = []
    for net, exp in TOP_PAIRS:
        sub = merged_df[(merged_df["network"] == net) & (merged_df["expert"] == exp)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append({
            "pair": f"{net}-{exp}",
            "shift": float(r["shift"]),
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "q_value": float(r["q_value"]) if not pd.isna(r["q_value"]) else np.nan,
        })

    out = pd.DataFrame(rows)
    out["abs_shift"] = out["shift"].abs()
    out = out.sort_values("abs_shift", ascending=False).drop(columns=["abs_shift"])
    out.to_csv(save_csv, index=False)
    print(f"[Saved table] {save_csv}")


# =========================================================
# Table 2: full table
# =========================================================
def build_full_table(merged_df, save_csv):
    out = merged_df[["network", "expert", "shift", "ci_low", "ci_high", "q_value"]].copy()
    out["pair"] = out["network"].astype(str) + "-" + out["expert"].astype(str)
    out["abs_shift"] = out["shift"].abs()
    out = out.sort_values(["abs_shift", "pair"], ascending=[False, True]).drop(columns=["abs_shift"])
    out = out[["pair", "shift", "ci_low", "ci_high", "q_value"]]
    out.to_csv(save_csv, index=False)
    print(f"[Saved table] {save_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--networks", type=str, default="SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL")
    parser.add_argument("--experts", type=str, default="mlp,cheb,gt,gcn")
    parser.add_argument("--fontsize", type=int, default=14)
    parser.add_argument("--ranking_metric", type=str, default="abs_diff", choices=["abs_diff", "signed_diff"])
    parser.add_argument("--annotate_heatmap", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.save_dir)

    networks = [x.strip() for x in args.networks.split(",") if x.strip()]
    experts = [x.strip() for x in args.experts.split(",") if x.strip()]

    overall_df, sig_df, boot_df, merged_df = load_merged_df(args.result_dir)

    plot_significance_heatmap(
        merged_df=merged_df,
        networks=networks,
        experts=experts,
        save_path=Path(args.save_dir) / "routing_shift_significance_heatmap.png",
        fontsize=args.fontsize,
        annotate=args.annotate_heatmap,
    )

    plot_routing_shift_ranking_from_df(
        overall_df=overall_df,
        save_path=Path(args.save_dir) / "routing_shift_ranking_all_pairs.png",
        fontsize=max(args.fontsize - 1, 10),
        ranking_metric=args.ranking_metric,
        label_extremes_only=True,
    )

    plot_top_ci(
        merged_df=merged_df,
        save_path=Path(args.save_dir) / "top_routing_shifts_bootstrap_ci.png",
        fontsize=args.fontsize,
    )

    build_top_table(
        merged_df=merged_df,
        save_csv=Path(args.save_dir) / "top_pairs_summary.csv",
    )

    build_full_table(
        merged_df=merged_df,
        save_csv=Path(args.save_dir) / "full_pairs_summary.csv",
    )

    print("[Done]")
    print(Path(args.save_dir) / "routing_shift_significance_heatmap.png")
    print(Path(args.save_dir) / "routing_shift_ranking_all_pairs.png")
    print(Path(args.save_dir) / "top_routing_shifts_bootstrap_ci.png")
    print(Path(args.save_dir) / "top_pairs_summary.csv")
    print(Path(args.save_dir) / "full_pairs_summary.csv")


if __name__ == "__main__":
    main()