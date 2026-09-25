import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import KERNEL_COUNT, data_moved, energy, flops, latency


HERE = Path(__file__).resolve().parent
MEASUREMENTS = HERE / "results" / "measurements.csv"
THETA = HERE / "results" / "theta.json"

def _theta(log_params):
    total_launch_time, performance, bandwidth = np.exp(log_params)
    return {
        "tau_s_per_launch": float(total_launch_time / KERNEL_COUNT),
        "flops_per_s": float(performance),
        "bytes_per_s": float(bandwidth),
    }


def calibrate(measurements_path=MEASUREMENTS, theta_path=THETA):
    """Fit on the base grid (is_validation=False) and report error on the random S/B pairs."""
    try:
        df = pd.read_csv(measurements_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Measurements CSV is empty; run measure.py on a GPU first") from exc

    required = {"S", "B", "oom", "latency_s", "is_validation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measurements CSV is missing columns: {sorted(missing)}")

    oom = df["oom"].astype(str).str.strip().str.lower().eq("true")
    measured = pd.to_numeric(df["latency_s"], errors="coerce")
    usable = (~oom) & measured.notna() & np.isfinite(measured) & measured.gt(0)
    if usable.sum() < 5:
        raise ValueError("Need at least five positive, non-OOM latency measurements")

    S = df.loc[usable, "S"].to_numpy(dtype=float)
    B = df.loc[usable, "B"].to_numpy(dtype=float)
    times = measured.loc[usable].to_numpy(dtype=float)
    validation = df.loc[usable, "is_validation"].astype(str).str.lower().eq("true").to_numpy()
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

    if "energy_j" not in df.columns:
        raise ValueError("Measurements CSV is missing the energy_j column")
    measured_energy = pd.to_numeric(df.loc[usable, "energy_j"], errors="coerce").to_numpy(dtype=float)
    valid_energy = np.isfinite(measured_energy) & (measured_energy > 0)
    energy_training = training & valid_energy
    energy_validation = validation & valid_energy
    if energy_training.sum() < 3 or not energy_validation.any():
        raise ValueError("Need positive energy measurements in training and validation rows")

    # Fit relative energy error so large configurations do not dominate the fit.
    features = np.column_stack((
        latency(S, B, theta),
        flops(S, B) / 1e9,
        data_moved(S, B) / 1e9,
    ))
    energy_values = measured_energy[energy_training]
    energy_params, _ = nnls(
        features[energy_training] / energy_values[:, None],
        np.ones(len(energy_values)),
    )
    theta.update({
        "base_power_w": float(energy_params[0]),
        "joules_per_gflop": float(energy_params[1]),
        "joules_per_gb": float(energy_params[2]),
    })
    energy_predicted = energy(S[energy_validation], B[energy_validation], theta)
    energy_mape = 100 * np.mean(
        np.abs(energy_predicted - measured_energy[energy_validation])
        / measured_energy[energy_validation]
    )

    theta_path = Path(theta_path)
    theta_path.parent.mkdir(parents=True, exist_ok=True)
    theta_path.write_text(json.dumps(theta, indent=2) + "\n", encoding="utf-8")

    compute_time = flops(S[training], B[training]) / theta["flops_per_s"]
    transfer_time = data_moved(S[training], B[training]) / theta["bytes_per_s"]
    memory_bound = int(np.count_nonzero(transfer_time > compute_time))
    compute_bound = int(training.sum()) - memory_bound
    print(f"Fitted theta: {theta}")
    print(f"Latency validation MAPE: {mape:.1f}% ({validation.sum()} rows)")
    print(f"Energy validation MAPE: {energy_mape:.1f}% ({energy_validation.sum()} rows)")
    print(f"Training branches: memory {memory_bound}, compute {compute_bound}")
    if memory_bound == 0 or compute_bound == 0:
        print("Warning: one throughput parameter is weakly identified by this dataset")
    return theta, float(mape)


if __name__ == "__main__":
    calibrate()
