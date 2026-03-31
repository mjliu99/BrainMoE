import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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
# =========================
# Utils
# =========================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pivot_shift_matrix(df, networks, experts, value_col):
    """
    df must contain columns: network, expert, value_col
    return matrix [len(networks), len(experts)]
    """
    piv = df.pivot(index="network", columns="expert", values=value_col)
    piv = piv.reindex(index=networks, columns=experts)
    return piv.values


def flatten_pairs(df, networks, experts):
    rows = []
    for n in networks:
        for e in experts:
            sub = df[(df["network"] == n) & (df["expert"] == e)]
            if len(sub) > 0:
                rows.append(sub.iloc[0].to_dict())
    return pd.DataFrame(rows)


# =========================
# Core computations
# =========================
def compute_fold_shift(df, networks, experts):
    """
    For each fold, compute:
      shift = mean(ADHD) - mean(HC)
    using correctly classified subjects only.
    """
    out_rows = []
    folds = sorted(df["fold"].dropna().unique().tolist())

    for f in folds:
        dff = df[(df["fold"] == f) & (df["correct"] == 1)]

        for n in networks:
            for e in experts:
                sub = dff[(dff["network"] == n) & (dff["expert"] == e)]
                hc = sub[sub["group"] == "HC"]["mean_prob"].values
                adhd = sub[sub["group"] == "ADHD"]["mean_prob"].values

                hc_mean = np.mean(hc) if len(hc) > 0 else np.nan
                adhd_mean = np.mean(adhd) if len(adhd) > 0 else np.nan
                shift = adhd_mean - hc_mean if len(hc) > 0 and len(adhd) > 0 else np.nan

                out_rows.append({
                    "fold": f,
                    "network": n,
                    "expert": e,
                    "hc_mean": hc_mean,
                    "adhd_mean": adhd_mean,
                    "shift": shift,
                    "n_hc": len(hc),
                    "n_adhd": len(adhd),
                })

    return pd.DataFrame(out_rows)


def compute_overall_shift(df, networks, experts):
    """
    Pool all correctly classified subjects across folds.
    """
    dff = df[df["correct"] == 1].copy()
    out_rows = []

    for n in networks:
        for e in experts:
            sub = dff[(dff["network"] == n) & (dff["expert"] == e)]
            hc = sub[sub["group"] == "HC"]["mean_prob"].values
            adhd = sub[sub["group"] == "ADHD"]["mean_prob"].values

            hc_mean = np.mean(hc) if len(hc) > 0 else np.nan
            adhd_mean = np.mean(adhd) if len(adhd) > 0 else np.nan
            shift = adhd_mean - hc_mean if len(hc) > 0 and len(adhd) > 0 else np.nan

            out_rows.append({
                "network": n,
                "expert": e,
                "hc_mean": hc_mean,
                "adhd_mean": adhd_mean,
                "shift": shift,
                "n_hc": len(hc),
                "n_adhd": len(adhd),
            })

    return pd.DataFrame(out_rows)


def compute_fold_consistency(fold_shift_df, networks, experts):
    """
    For each pair, compute:
      - mean shift across folds
      - std shift across folds
      - positive_count
      - negative_count
      - sign_consistency
    """
    out_rows = []

    for n in networks:
        for e in experts:
            sub = fold_shift_df[(fold_shift_df["network"] == n) & (fold_shift_df["expert"] == e)]
            vals = sub["shift"].dropna().values

            if len(vals) == 0:
                out_rows.append({
                    "network": n,
                    "expert": e,
                    "mean_shift": np.nan,
                    "std_shift": np.nan,
                    "positive_count": np.nan,
                    "negative_count": np.nan,
                    "sign_consistency": np.nan,
                    "n_folds": 0,
                })
                continue

            pos = np.sum(vals > 0)
            neg = np.sum(vals < 0)
            sign_consistency = max(pos, neg) / len(vals)

            out_rows.append({
                "network": n,
                "expert": e,
                "mean_shift": float(np.mean(vals)),
                "std_shift": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "positive_count": int(pos),
                "negative_count": int(neg),
                "sign_consistency": float(sign_consistency),
                "n_folds": int(len(vals)),
            })

    return pd.DataFrame(out_rows)


def mannwhitney_fdr(df, networks, experts):
    """
    Subject-level group comparison:
    ADHD vs HC for each network-expert pair.
    """
    dff = df[df["correct"] == 1].copy()
    rows = []

    for n in networks:
        for e in experts:
            sub = dff[(dff["network"] == n) & (dff["expert"] == e)]
            hc = sub[sub["group"] == "HC"]["mean_prob"].values
            adhd = sub[sub["group"] == "ADHD"]["mean_prob"].values

            if len(hc) < 2 or len(adhd) < 2:
                rows.append({
                    "network": n,
                    "expert": e,
                    "p_value": np.nan,
                    "effect_shift": np.nan,
                    "n_hc": len(hc),
                    "n_adhd": len(adhd),
                })
                continue

            stat, p = mannwhitneyu(adhd, hc, alternative="two-sided")
            effect_shift = float(np.mean(adhd) - np.mean(hc))

            rows.append({
                "network": n,
                "expert": e,
                "p_value": float(p),
                "effect_shift": effect_shift,
                "n_hc": len(hc),
                "n_adhd": len(adhd),
            })

    res = pd.DataFrame(rows)

    valid = res["p_value"].notna().values
    qvals = np.full(len(res), np.nan)
    reject = np.full(len(res), False)

    if valid.sum() > 0:
        rej, q, _, _ = multipletests(res.loc[valid, "p_value"].values, alpha=0.05, method="fdr_bh")
        qvals[valid] = q
        reject[valid] = rej

    res["q_value"] = qvals
    res["significant"] = reject
    return res


def bootstrap_ci(df, networks, experts, n_boot=2000, seed=42):
    """
    Bootstrap ADHD-HC shift for each network-expert pair.
    Resample subjects WITH replacement within each group.
    """
    rng = np.random.default_rng(seed)
    dff = df[df["correct"] == 1].copy()

    rows = []

    for n in networks:
        for e in experts:
            sub = dff[(dff["network"] == n) & (dff["expert"] == e)]
            hc = sub[sub["group"] == "HC"]["mean_prob"].values
            adhd = sub[sub["group"] == "ADHD"]["mean_prob"].values

            if len(hc) < 2 or len(adhd) < 2:
                rows.append({
                    "network": n,
                    "expert": e,
                    "shift_mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_cross_zero": np.nan,
                })
                continue

            boot_stats = []
            for _ in range(n_boot):
                hc_s = rng.choice(hc, size=len(hc), replace=True)
                adhd_s = rng.choice(adhd, size=len(adhd), replace=True)
                boot_stats.append(np.mean(adhd_s) - np.mean(hc_s))

            boot_stats = np.array(boot_stats)
            shift_mean = float(np.mean(adhd) - np.mean(hc))
            ci_low = float(np.percentile(boot_stats, 2.5))
            ci_high = float(np.percentile(boot_stats, 97.5))
            cross_zero = (ci_low <= 0.0 <= ci_high)

            rows.append({
                "network": n,
                "expert": e,
                "shift_mean": shift_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_cross_zero": cross_zero,
            })

    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================
def plot_shift_heatmap_with_significance(
    shift_df,
    sig_df,
    networks,
    experts,
    save_path,
    title="Routing shift (ADHD - HC)"
):
    mat = pivot_shift_matrix(shift_df, networks, experts, "shift")
    qmat = pivot_shift_matrix(sig_df, networks, experts, "q_value")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(experts)))
    ax.set_xticklabels(experts, fontsize=13)
    ax.set_yticks(np.arange(len(networks)))
    ax.set_yticklabels(networks, fontsize=13)
    ax.set_xlabel("Expert", fontsize=15)
    ax.set_ylabel("Network", fontsize=15)
    ax.set_title(title, fontsize=18, pad=12)

    for i in range(len(networks)):
        for j in range(len(experts)):
            val = mat[i, j]
            q = qmat[i, j]

            if np.isnan(val):
                text = "NA"
            else:
                star = ""
                if not np.isnan(q):
                    if q < 0.001:
                        star = "***"
                    elif q < 0.01:
                        star = "**"
                    elif q < 0.05:
                        star = "*"
                text = f"{val:.3f}{star}"

            ax.text(j, i, text, ha="center", va="center", fontsize=11, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_fold_consistency_heatmap(
    cons_df,
    networks,
    experts,
    save_path,
    fontsize=14,
):
    mat = pivot_shift_matrix(cons_df, networks, experts, "mean_shift")
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=300)
    im = ax.imshow(
        mat,
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
    ax.set_title("Fold consistency of routing shift", fontsize=fontsize + 4, pad=10)

    ax.set_xticks(np.arange(-0.5, len(experts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(networks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, n in enumerate(networks):
        for j, e in enumerate(experts):
            row = cons_df[(cons_df["network"] == n) & (cons_df["expert"] == e)].iloc[0]
            mean_shift = row["mean_shift"]
            pos = row["positive_count"]
            neg = row["negative_count"]
            nf = row["n_folds"]

            if np.isnan(mean_shift) or nf == 0:
                txt = "NA"
            else:
                dominant = max(pos, neg)
                txt = f"{mean_shift:+.3f}\n{int(dominant)}/{int(nf)}"

            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=fontsize - 4,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize - 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def build_top_table(overall_shift_df, sig_df, boot_df, save_csv):
    merged = overall_shift_df.merge(
        sig_df[["network", "expert", "q_value"]],
        on=["network", "expert"],
        how="left",
    ).merge(
        boot_df[["network", "expert", "ci_low", "ci_high"]],
        on=["network", "expert"],
        how="left",
    )

    rows = []
    for net, exp in TOP_PAIRS:
        sub = merged[(merged["network"] == net) & (merged["expert"] == exp)]
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

def build_full_table(overall_shift_df, sig_df, boot_df, fold_cons_df, save_csv):
    merged = overall_shift_df.merge(
        sig_df[["network", "expert", "q_value"]],
        on=["network", "expert"],
        how="left",
    ).merge(
        boot_df[["network", "expert", "ci_low", "ci_high"]],
        on=["network", "expert"],
        how="left",
    ).merge(
        fold_cons_df[["network", "expert", "sign_consistency"]],
        on=["network", "expert"],
        how="left",
    )

    merged["pair"] = merged["network"].astype(str) + "-" + merged["expert"].astype(str)
    merged["abs_shift"] = merged["shift"].abs()
    merged = merged.sort_values(
        ["abs_shift", "pair"], ascending=[False, True]
    ).drop(columns=["abs_shift"])

    out = merged[["pair", "shift", "ci_low", "ci_high", "q_value", "sign_consistency"]].copy()
    out.to_csv(save_csv, index=False)
    print(f"[Saved table] {save_csv}")

def plot_top_routing_shifts_bootstrap_ci(
    overall_shift_df,
    sig_df,
    boot_df,
    save_path,
    fontsize=14,
):
    merged = overall_shift_df.merge(
        sig_df[["network", "expert", "q_value"]],
        on=["network", "expert"],
        how="left",
    ).merge(
        boot_df[["network", "expert", "ci_low", "ci_high"]],
        on=["network", "expert"],
        how="left",
    )

    rows = []
    for net, exp in TOP_PAIRS:
        sub = merged[(merged["network"] == net) & (merged["expert"] == exp)]
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

    y = np.arange(len(df)) * 1.65
    x = df["shift"].values
    xerr_low = x - df["ci_low"].values
    xerr_high = df["ci_high"].values - x

    xmin = float(np.min(df["ci_low"]) - 0.030)
    xmax = float(np.max(df["ci_high"]) + 0.030)

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
    ax.set_ylim(-0.45, y[-1] + 1.05)

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
    plt.close()

def plot_foldwise_small_multiples(fold_shift_df, networks, experts, save_path):
    folds = sorted(fold_shift_df["fold"].unique().tolist())
    n = len(folds)

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), dpi=180)
    axes = np.array(axes).reshape(-1)

    all_vals = fold_shift_df["shift"].dropna().values
    vmax = np.max(np.abs(all_vals)) if len(all_vals) else 1.0

    for ax, f in zip(axes, folds):
        sub = fold_shift_df[fold_shift_df["fold"] == f]
        mat = pivot_shift_matrix(sub, networks, experts, "shift")

        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Fold {f}", fontsize=13)
        ax.set_xticks(np.arange(len(experts)))
        ax.set_xticklabels(experts, fontsize=10)
        ax.set_yticks(np.arange(len(networks)))
        ax.set_yticklabels(networks, fontsize=10)

        for i in range(len(networks)):
            for j in range(len(experts)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    for ax in axes[len(folds):]:
        ax.axis("off")

    fig.suptitle("Fold-wise routing shift (ADHD - HC)", fontsize=18, y=0.98)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input subject-level routing CSV")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--networks", type=str, default="SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL")
    parser.add_argument("--experts", type=str, default="mlp,cheb,gt,gcn")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    networks = [x.strip() for x in args.networks.split(",")]
    experts = [x.strip() for x in args.experts.split(",")]

    df = pd.read_csv(args.csv)

    # Basic clean
    df["group"] = df["group"].astype(str)
    df["network"] = df["network"].astype(str)
    df["expert"] = df["expert"].astype(str)
    df["correct"] = df["correct"].astype(int)

    # 1) fold-level shifts
    fold_shift_df = compute_fold_shift(df, networks, experts)
    fold_shift_df.to_csv(Path(args.outdir) / "fold_shift.csv", index=False)

    # 2) overall pooled shift
    overall_shift_df = compute_overall_shift(df, networks, experts)
    overall_shift_df.to_csv(Path(args.outdir) / "overall_shift.csv", index=False)

    # 3) fold consistency
    fold_cons_df = compute_fold_consistency(fold_shift_df, networks, experts)
    fold_cons_df.to_csv(Path(args.outdir) / "fold_consistency.csv", index=False)

    # 4) significance
    sig_df = mannwhitney_fdr(df, networks, experts)
    sig_df.to_csv(Path(args.outdir) / "mannwhitney_fdr.csv", index=False)

    # 5) bootstrap CI
    boot_df = bootstrap_ci(df, networks, experts, n_boot=args.n_boot, seed=42)
    boot_df.to_csv(Path(args.outdir) / "bootstrap_ci.csv", index=False)

    # plots
    plot_shift_heatmap_with_significance(
        overall_shift_df,
        sig_df,
        networks,
        experts,
        save_path=Path(args.outdir) / "routing_shift_significance_heatmap.png",
        title="Routing shift (ADHD - HC)"
    )

    plot_fold_consistency_heatmap(
        fold_cons_df,
        networks,
        experts,
        save_path=Path(args.outdir) / "fold_consistency_heatmap.png",
        fontsize=14,
    )

    plot_top_routing_shifts_bootstrap_ci(
        overall_shift_df,
        sig_df,
        boot_df,
        save_path=Path(args.outdir) / "top_routing_shifts_bootstrap_ci.png",
        fontsize=14,
    )

    plot_foldwise_small_multiples(
        fold_shift_df,
        networks,
        experts,
        save_path=Path(args.outdir) / "foldwise_shift_small_multiples.png"
    )

    build_top_table(
        overall_shift_df,
        sig_df,
        boot_df,
        save_csv=Path(args.outdir) / "top_pairs_summary.csv",
    )

    build_full_table(
        overall_shift_df,
        sig_df,
        boot_df,
        fold_cons_df,
        save_csv=Path(args.outdir) / "full_pairs_summary.csv",
    )

    print(f"[Done] Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()