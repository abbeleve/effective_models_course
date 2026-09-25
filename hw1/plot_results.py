"""Plot analytical predictions against the measured CNN forward passes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from equations import KERNEL_COUNT, data_moved, energy, flops, latency, memory
from models import CNN


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
S_GRID = np.arange(32, 513, 16)
# torch.cuda.get_device_properties(0).total_memory of the RTX 5060 Ti used for measurements
GPU_MEMORY_BYTES = 16_616_521_728

BLUE = "#215d91"
GRAY = "#28323c"
ORANGE = "#d87519"
RED = "#b3261e"

METRICS = (
    ("memory", "memory_bytes", "Memory [MiB]", 1 / 2**20, False),
    ("latency", "latency_s", "Latency [ms]", 1e3, True),
    ("energy", "energy_j", "Energy [J]", 1.0, True),
)
FLOPS_METRIC = ("flops", "flops_counted", "Work [GFLOP]", 1e-9, True)


def counted_flops(S, B):
    """Independent FLOP count from PyTorch's per-operator counter (shapes only, meta device)."""
    model = CNN().to("meta").eval()
    counts = []
    with torch.inference_mode():
        for s, b in zip(S, B):
            with FlopCounterMode(display=False) as counter:
                model(torch.empty(int(b), 3, int(s), int(s), device="meta"))
            counts.append(counter.get_total_flops())
    return np.array(counts, dtype=float)


def load_results():
    df = pd.read_csv(RESULTS / "measurements.csv")
    required = {"S", "B", "oom", "is_validation", "latency_s", "memory_bytes", "energy_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")

    df["is_validation"] = df["is_validation"].astype(str).str.lower().eq("true")
    df["oom"] = df["oom"].astype(str).str.lower().eq("true")
    for column in ("S", "B", "latency_s", "memory_bytes", "energy_j"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    measured = df.loc[~df["oom"]].dropna(subset=["S", "B", "latency_s", "memory_bytes", "energy_j"]).copy()
    if measured.empty:
        raise ValueError("No complete non-OOM measurements available")

    theta = json.loads((RESULTS / "theta.json").read_text(encoding="utf-8"))
    required_theta = {
        "tau_s_per_launch", "flops_per_s", "bytes_per_s",
        "base_power_w", "joules_per_gflop", "joules_per_gb",
    }
    missing_theta = required_theta - set(theta)
    if missing_theta:
        raise ValueError(f"Missing theta values {sorted(missing_theta)}; run calibrate.py")

    S = measured["S"].to_numpy(dtype=int)
    B = measured["B"].to_numpy(dtype=int)
    measured["pred_memory_bytes"] = memory(S, B)
    measured["pred_latency_s"] = latency(S, B, theta)
    measured["pred_energy_j"] = energy(S, B, theta)
    measured["pred_flops_counted"] = flops(S, B)
    measured["flops_counted"] = counted_flops(S, B)
    return df, measured, theta


def prediction(name, S, B, theta):
    if name == "memory":
        return memory(S, B)
    if name == "latency":
        return latency(S, B, theta)
    if name == "flops":
        return flops(S, B)
    return energy(S, B, theta)


def legend_handles(point_label="Measured"):
    return [
        Line2D([0], [0], color=BLUE, lw=2, label="Analytical prediction"),
        Line2D([0], [0], marker="o", linestyle="", color=GRAY, label=f"{point_label}: calibration"),
        Line2D([0], [0], marker="s", linestyle="", color=ORANGE, label=f"{point_label}: validation"),
    ]


def scatter_split(ax, points, x_column, y_column, scale=1.0, size=35):
    for validation, marker, color in ((False, "o", GRAY), (True, "s", ORANGE)):
        selected = points.loc[points["is_validation"] == validation]
        ax.scatter(
            selected[x_column], selected[y_column] * scale,
            marker=marker, color=color, s=size, zorder=3,
        )


def plot_metric(df, theta, name, measured_column, ylabel, scale, log_y, point_label="Measured"):
    batches = sorted(df["B"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True)
    for ax, batch in zip(axes.flat, batches):
        subset = df.loc[df["B"] == batch].sort_values("S")
        ax.plot(S_GRID, prediction(name, S_GRID, batch, theta) * scale, color=BLUE, lw=2)
        scatter_split(ax, subset, "S", measured_column, scale)
        ax.set_title(f"B = {batch}")
        ax.set_xlim(24, 520)
        ax.set_xticks((32, 128, 224, 320, 416, 512))
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.22)

    fig.suptitle(f"{name.capitalize()}: prediction and measurements", fontsize=15)
    fig.supxlabel("Image size S [pixels]", y=0.05)
    fig.supylabel(ylabel)
    fig.legend(
        handles=legend_handles(point_label), loc="lower center",
        bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.96))
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_memory_jump(df, theta):
    fig, ax = plt.subplots(figsize=(8, 5))
    S = np.arange(368, 417, 16)
    ax.plot(S, memory(S, 2) / 2**20, color=BLUE, lw=2)
    points = df.loc[(df["B"] == 2) & df["S"].isin((384, 400))]
    scatter_split(ax, points, "S", "memory_bytes", 1 / 2**20, size=65)
    for row in points.itertuples():
        ax.annotate(
            f"{row.memory_bytes / 2**20:.1f} MiB",
            (row.S, row.memory_bytes / 2**20), xytext=(7, 4),
            textcoords="offset points", fontsize=10,
        )
    ax.set(xlim=(372, 412), xlabel="Image size S [pixels]", ylabel="Peak memory [MiB]")
    ax.set_title("Memory jump at B = 2: conv2 switches to Winograd (see kernels.csv)")
    ax.grid(alpha=0.22)
    ax.legend(handles=legend_handles(), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "memory_b2_jump.png", dpi=180)
    plt.close(fig)


def plot_parity(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, measured_column, ylabel, scale, _) in zip(axes, METRICS):
        predicted_column = f"pred_{measured_column}"
        x = df[predicted_column].to_numpy(dtype=float) * scale
        y = df[measured_column].to_numpy(dtype=float) * scale
        lo = min(x.min(), y.min()) * 0.8
        hi = max(x.max(), y.max()) * 1.25
        ax.plot((lo, hi), (lo, hi), "--", color=BLUE, lw=1.5, label="Perfect prediction")
        for validation, marker, color in ((False, "o", GRAY), (True, "s", ORANGE)):
            mask = df["is_validation"].to_numpy() == validation
            ax.scatter(x[mask], y[mask], marker=marker, color=color, s=22, alpha=0.8)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_title(name.capitalize())
        ax.set_xlabel(f"Predicted {ylabel}")
        ax.set_ylabel(f"Measured {ylabel}")
        ax.grid(alpha=0.2)

    fig.suptitle("All non-OOM configurations: measured vs predicted", fontsize=15)
    handles = legend_handles()[1:]
    handles.insert(0, Line2D([0], [0], linestyle="--", color=BLUE, label="Perfect prediction"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIGURES / "prediction_vs_measurement.png", dpi=180)
    plt.close(fig)


def plot_error_heatmap(df):
    """Relative error on the whole (S, B) plane; validation cells are outlined."""
    S_values = sorted(df["S"].unique())
    B_values = sorted(df["B"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    for ax, (name, measured_column, _, _, _) in zip(axes, METRICS):
        error = 100 * (df[f"pred_{measured_column}"] - df[measured_column]) / df[measured_column]
        table = (
            df.assign(error=error)
            .pivot(index="S", columns="B", values="error")
            .reindex(index=S_values, columns=B_values)
        )
        image = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-60, vmax=60, origin="lower", aspect="auto")
        for i, S in enumerate(S_values):
            for j, B in enumerate(B_values):
                value = table.loc[S, B]
                if np.isnan(value):
                    ax.text(j, i, "OOM", ha="center", va="center", fontsize=7)
                    continue
                ax.text(j, i, f"{int(round(value))}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 40 else "black")
        validation = df.loc[df["is_validation"], ["S", "B"]]
        for row in validation.itertuples():
            ax.add_patch(Rectangle(
                (B_values.index(row.B) - 0.5, S_values.index(row.S) - 0.5), 1, 1,
                fill=False, edgecolor=ORANGE, lw=1.4,
            ))
        ax.set_xticks(range(len(B_values)), B_values)
        ax.set_yticks(range(len(S_values)), S_values)
        ax.set(xlabel="Batch size B", ylabel="Image size S [pixels]", title=name.capitalize())
        fig.colorbar(image, ax=ax, label="(predicted − measured) / measured [%]", shrink=0.9)

    fig.suptitle("Relative prediction error on the (S, B) plane (orange frame = validation pair)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "error_heatmap.png", dpi=150)
    plt.close(fig)


def regime_boundaries(N, regime):
    changes = np.flatnonzero(np.diff(regime)) + 1
    return [(N[i], regime[i - 1], regime[i]) for i in changes]


def plot_regimes(df, theta):
    """Latency against pixels per pass N = B * S^2 with the three terms of the model."""
    N = np.geomspace(4e2, 2e8, 600)
    S_path = 128  # B = N / S^2; terms that depend on B alone change the curve by < 3 %
    B_path = N / S_path**2
    launch = np.full_like(N, KERNEL_COUNT * theta["tau_s_per_launch"])
    compute = flops(S_path, B_path) / theta["flops_per_s"]
    transfer = data_moved(S_path, B_path) / theta["bytes_per_s"]
    regime = np.where(launch >= np.maximum(compute, transfer), 0, np.where(transfer > compute, 1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    names = ("launch-bound", "memory-bound", "compute-bound")
    shades = ("#ececec", "#dce8f5", "#f8e6d4")
    edges = [N[0], *[n for n, _, _ in regime_boundaries(N, regime)], N[-1]]
    for left, right in zip(edges[:-1], edges[1:]):
        kind = regime[np.searchsorted(N, np.sqrt(left * right))]
        ax.axvspan(left, right, color=shades[kind], zorder=0)
        ax.text(np.sqrt(left * right), 0.97, names[kind], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9)

    ax.plot(N, launch * 1e3, ":", color=GRAY, lw=1.5, label=f"launch: {KERNEL_COUNT}·τ")
    ax.plot(N, transfer * 1e3, "--", color="#5b7fb3", lw=1.5, label="memory: bytes / BW")
    ax.plot(N, compute * 1e3, "--", color="#c0703a", lw=1.5, label="compute: FLOPs / P")
    ax.plot(N, latency(S_path, B_path, theta) * 1e3, color=BLUE, lw=2.2, label="Latency model")
    points = df.assign(N=df["B"] * df["S"] ** 2)
    scatter_split(ax, points, "N", "latency_s", 1e3, size=22)
    ax.set(
        xscale="log", yscale="log", xlim=(N[0], N[-1]),
        xlabel="Pixels per forward pass N = B·S²", ylabel="Latency [ms]",
        title="Latency regimes: launch overhead → memory bandwidth → compute",
    )
    handles, _ = ax.get_legend_handles_labels()
    handles += legend_handles()[1:]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "regimes.png", dpi=180)
    plt.close(fig)
    return regime_boundaries(N, regime), names


def plot_oom_boundary(df_all):
    """Largest batch that memory() allows on this GPU vs the configurations actually run."""
    base = memory(S_GRID, 0)
    per_image = memory(S_GRID, 1) - base
    b_max = (GPU_MEMORY_BYTES - base) / per_image

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(S_GRID, b_max, color=BLUE, lw=2, label=f"Predicted OOM boundary: memory(S, B) = {GPU_MEMORY_BYTES / 2**30:.1f} GiB")
    ax.fill_between(S_GRID, b_max, 1e5, color=BLUE, alpha=0.08, label="Predicted OOM region")
    scatter_split(ax, df_all.loc[~df_all["oom"]], "S", "B", size=28)
    oom = df_all.loc[df_all["oom"]]
    ax.scatter(oom["S"], oom["B"], marker="x", color=RED, s=60, zorder=4)
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="o", linestyle="", color=GRAY, label="Measured, fits: calibration"),
        Line2D([0], [0], marker="s", linestyle="", color=ORANGE, label="Measured, fits: validation"),
        Line2D([0], [0], marker="x", linestyle="", color=RED, label=f"Measured OOM ({len(oom)})"),
    ]
    ax.set(
        yscale="log", xlim=(24, 520), ylim=(0.8, 3e4),
        xlabel="Image size S [pixels]", ylabel="Batch size B",
        title="OOM: memory() prediction vs measured configurations",
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "oom_boundary.png", dpi=180)
    plt.close(fig)
    return dict(zip(S_GRID.tolist(), b_max.tolist()))


def main():
    df_all, df, theta = load_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(df, theta, *metric)
    plot_metric(df, theta, *FLOPS_METRIC, point_label="Counted by FlopCounterMode")
    plot_memory_jump(df, theta)
    plot_parity(df)
    plot_error_heatmap(df)
    boundaries, names = plot_regimes(df, theta)
    b_max = plot_oom_boundary(df_all)

    for name, measured_column, *_ in (*METRICS, FLOPS_METRIC):
        predicted_column = f"pred_{measured_column}"
        error = 100 * np.abs(df[predicted_column] - df[measured_column]) / df[measured_column]
        print(
            f"{name} MAPE: calibration {error[~df['is_validation']].mean():.1f}%, "
            f"validation {error[df['is_validation']].mean():.1f}%"
        )
    for n, before, after in boundaries:
        print(f"regime change {names[before]} -> {names[after]} at N = B*S^2 = {n:,.0f}")
    print(f"predicted max batch without OOM: S=224 -> {b_max[224]:.0f}, S=512 -> {b_max[512]:.0f}")
    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
