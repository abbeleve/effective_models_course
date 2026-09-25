import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import KERNEL_COUNT, data_moved, flops, latency


HERE = Path(__file__).resolve().parent
MEASUREMENTS = HERE / "results" / "measurements.csv"
THETA = HERE / "results" / "theta.json"
SEED = 676767

def _theta(log_params):
    total_launch_time, performance, bandwidth = np.exp(log_params)
    return {
        "tau_s_per_launch": float(total_launch_time / KERNEL_COUNT),
        "flops_per_s": float(performance),
        "bytes_per_s": float(bandwidth),
    }


def calibrate(measurements_path=MEASUREMENTS, theta_path=THETA):
    """Fit on 80% of usable rows and report error on the held-out 20%."""
    try:
        df = pd.read_csv(measurements_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Measurements CSV is empty; run measure.py on a GPU first") from exc

    required = {"S", "B", "oom", "latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measurements CSV is missing columns: {sorted(missing)}")

    oom = df["oom"].astype(str).str.strip().str.lower().eq("true")
    measured = pd.to_numeric(df["latency_s"], errors="coerce")
    usable = (~oom) & measured.notna() & np.isfinite(measured) & measured.gt(0)
    if usable.sum() < 5:
        raise ValueError("Need at least five positive, non-OOM latency measurements")

    if "is_validation" not in df.columns:
        df["is_validation"] = False
        indices = df.index[usable].to_numpy()
        rng = np.random.default_rng(SEED)
        count = max(1, round(0.2 * len(indices)))
        df.loc[rng.choice(indices, size=count, replace=False), "is_validation"] = True
        df.to_csv(measurements_path, index=False)

    S = df.loc[usable, "S"].to_numpy(dtype=float)
    B = df.loc[usable, "B"].to_numpy(dtype=float)
    times = measured.loc[usable].to_numpy(dtype=float)
    validation = df.loc[usable, "is_validation"].fillna(False).to_numpy(dtype=bool)
    training = ~validation
    if not training.any() or not validation.any():
        raise ValueError("Both training and validation rows are required")
    if not (np.isfinite(S).all() and np.isfinite(B).all()):
        raise ValueError("Image sizes and batch sizes must be finite")

    def residuals(log_params):
        prediction = latency(S[training], B[training], _theta(log_params))
        return np.log(prediction) - np.log(times[training])

    launch_start = max(float(times[training].min()) / 2, 1e-8)
    lower = np.log([1e-9, 1e8, 1e7])
    upper = np.log([1.0, 1e15, 1e13])
    fits = []
    for performance, bandwidth in [(1e11, 1e10), (1e12, 1e11), (1e13, 1e12)]:
        start = np.log([launch_start, performance, bandwidth])
        fits.append(
            least_squares(
                residuals,
                x0=start,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.15,
                max_nfev=2000,
            )
        )
    successful = [fit for fit in fits if fit.success]
    if not successful:
        raise RuntimeError("Latency calibration did not converge")
    result = min(successful, key=lambda fit: fit.cost)
    theta = _theta(result.x)

    predicted = latency(S[validation], B[validation], theta)
    mape = 100 * np.mean(np.abs(predicted - times[validation]) / times[validation])

    theta_path = Path(theta_path)
    theta_path.parent.mkdir(parents=True, exist_ok=True)
    theta_path.write_text(json.dumps(theta, indent=2) + "\n", encoding="utf-8")

    compute_time = flops(S[training], B[training]) / theta["flops_per_s"]
    transfer_time = data_moved(S[training], B[training]) / theta["bytes_per_s"]
    memory_bound = int(np.count_nonzero(transfer_time > compute_time))
    compute_bound = int(training.sum()) - memory_bound
    print(f"Fitted theta: {theta}")
    print(f"Validation MAPE: {mape:.1f}% ({validation.sum()} rows)")
    print(f"Training branches: memory {memory_bound}, compute {compute_bound}")
    if memory_bound == 0 or compute_bound == 0:
        print("Warning: one throughput parameter is weakly identified by this dataset")
    return theta, float(mape)


if __name__ == "__main__":
    calibrate()