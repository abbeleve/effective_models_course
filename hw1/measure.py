import argparse
import subprocess
import sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset
from models import CNN
import random
import pandas as pd
from statistics import median
import math
from cuda.bindings import nvml

RESULTS = Path(__file__).resolve().parent / "results"
device = torch.device("cuda")
seed = 676767
rng = random.Random(seed)


BASE_IMAGE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
random_image_sizes = [i for i in range(32, 512, 16) if i not in BASE_IMAGE_SIZES]
random_batch_sizes = [i for i in range(1, 257) if i not in [2**j for j in range(1, 10)] and i not in BASE_BATCH_SIZES]

image_sizes = BASE_IMAGE_SIZES + rng.sample(random_image_sizes, 4)
batch_sizes = BASE_BATCH_SIZES + rng.sample(random_batch_sizes, 3)


def is_validation(S, B):
    """Calibrate on the base grid; hold out every pair with a random S or B."""
    return S not in BASE_IMAGE_SIZES or B not in BASE_BATCH_SIZES

class MyOwnDataset(Dataset):
    def __init__(self, image_sizes):
        super().__init__()
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    def __getitem__(self, index):
        return torch.rand(3, self.image_sizes[index], self.image_sizes[index])

dataset = MyOwnDataset(image_sizes)

def measure(model, dataset, image_index, B, gpu, warmup=5, repeats=20):
    S = dataset.image_sizes[image_index]
    x = None
    try:
        with torch.inference_mode():
            x = torch.stack(
                [dataset[image_index] for _ in range(B)],
                dim=0,
            ).to(device)
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            start = torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(repeats):
                start.record()
                model(x)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000)
            latency = median(samples)
            energy_j = measure_energy_j(model, x, latency, gpu)

        return {
            "S": S, "B": B, "oom": False,
            "latency_s": latency,
            "memory_bytes": peak_bytes,
            "energy_j": energy_j,
        }

    except torch.cuda.OutOfMemoryError:
        del x
        torch.cuda.empty_cache()
        return {
            "S": S, "B": B, "oom": True,
            "latency_s": None, "energy_j": None, "memory_bytes": None,
        }

def measure_energy_j(model, x, latency_s, gpu, repeats=3):
    n = max(10, min(10_000, math.ceil(2.0 / latency_s)))
    values = []

    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            before_mj = nvml.device_get_total_energy_consumption(gpu)

            for _ in range(n):
                model(x)

            torch.cuda.synchronize(device)
            after_mj = nvml.device_get_total_energy_consumption(gpu)
            values.append((after_mj - before_mj) / (1000 * n))

    return median(values)

def layer_steps(model):
    # Same order as CNN.forward; ReLU module act1 is reused after every conv.
    return [
        ("conv1", model.conv1), ("relu1", model.act1), ("maxpool1", model.maxpool1),
        ("conv2", model.conv2), ("relu2", model.act1),
        ("conv3", model.conv3), ("relu3", model.act1),
        ("conv4", model.conv4), ("relu4", model.act1),
        ("conv5", model.conv5), ("relu5", model.act1),
        ("conv6", model.conv6), ("relu6", model.act1),
        ("gap", model.gap), ("flatten", lambda t: torch.flatten(t, 1)),
        ("fc1", model.fc[0]), ("relu_fc", model.fc[1]), ("fc2", model.fc[2]),
    ]

def collect_kernels(model, S, B):
    """Run the forward pass layer by layer and record every GPU kernel of each layer."""
    rows = []
    try:
        x = torch.rand(B, 3, S, S, device=device)
        with torch.inference_mode():
            model(x)  # warmup: cuDNN picks its algorithm, cuBLAS allocates workspace
            torch.cuda.synchronize(device)
            for layer, step in layer_steps(model):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    y = step(x)
                    torch.cuda.synchronize(device)
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA:
                        rows.append({"S": S, "B": B, "layer": layer, "kernel": event.name})
                x = y
    except torch.cuda.OutOfMemoryError:
        rows = []
    x = y = None
    torch.cuda.empty_cache()
    return rows

def test_memory_peak(model, S, B=2, warmup=5):
    x = torch.rand(B, 3, S, S, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(device)

        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        model(x)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)

    extra = peak - baseline
    mib = 1024 ** 2
    print(
        f"S={S}, B={B}: baseline={baseline} ({baseline / mib:.2f} MiB), "
        f"peak={peak} ({peak / mib:.2f} MiB), "
        f"extra={extra} ({extra / mib:.2f} MiB)",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-test_hyp", "--test-hyp", nargs="?", const="True",
        choices=("True", "False"), default="False",
        help="проверить пик памяти для S=384 и S=400 при B=2",
    )
    parser.add_argument(
        "--memory-size", type=int, choices=(384, 400),
        help="проверить один размер в отдельном процессе",
    )
    parser.add_argument(
        "--kernels-only", action="store_true",
        help="собрать только results/kernels.csv, не перезаписывая measurements.csv",
    )
    args = parser.parse_args()

    if args.test_hyp == "True" and args.memory_size is None:
        for S in (384, 400):
            subprocess.run(
                [sys.executable, __file__, "--memory-size", str(S)],
                check=True,
            )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Для измерений требуется GPU с CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CNN().to(device).eval()

    if args.memory_size is not None:
        test_memory_peak(model, args.memory_size)
        return

    kernel_rows = []
    if args.kernels_only:
        for batch_size in batch_sizes:
            for S in image_sizes:
                kernel_rows += collect_kernels(model, S, batch_size)
                print("kernels", batch_size, S, flush=True)
        pd.DataFrame(kernel_rows).to_csv(RESULTS / "kernels.csv", index=False)
        return

    nvml.init_v2()
    gpu = nvml.device_get_handle_by_index_v2(0)

    rows = []
    for batch_size in batch_sizes:
        for image_index in range(len(dataset)):
            measure_results = measure(model, dataset, image_index, batch_size, gpu)
            measure_results["is_validation"] = is_validation(measure_results["S"], batch_size)
            rows.append(measure_results)
            if not measure_results["oom"]:
                kernel_rows += collect_kernels(model, measure_results["S"], batch_size)
            print(batch_size, image_sizes[image_index])

    pd.DataFrame(rows).to_csv(RESULTS / "measurements.csv", index=False)
    pd.DataFrame(kernel_rows).to_csv(RESULTS / "kernels.csv", index=False)

if __name__ == "__main__":
    main()
