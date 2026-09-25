import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models import CNN
import random
import time
import pandas as pd
from statistics import median

if not torch.cuda.is_available():
    raise RuntimeError("Для измерений требуется GPU с CUDA")
device = torch.device("cuda")
seed = 676767
rng = random.Random(676767)

model = CNN()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model = model.to(device).eval()


image_sizes = [32, 64, 128, 224, 256, 384, 512]
random_image_sizes = [i for i in range(32, 512, 16) if i not in image_sizes]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
random_batch_sizes = [i for i in range(1, 257) if i not in [2**j for j in range(1, 10)] and i not in batch_sizes]

image_sizes += rng.sample(random_image_sizes, 4)
batch_sizes += rng.sample(random_batch_sizes, 3)

class MyOwnDataset(Dataset):
    def __init__(self, image_sizes):
        super().__init__()
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    def __getitem__(self, index):
        return torch.rand(3, self.image_sizes[index], self.image_sizes[index])

dataset = MyOwnDataset(image_sizes)

def measure(model, dataset, image_index, B, warmup=5, repeats=20):
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

        return {
            "S": S, "B": B, "oom": False,
            "latency_s": median(samples),
            "memory_bytes": peak_bytes,
        }

    except torch.cuda.OutOfMemoryError:
        del x
        torch.cuda.empty_cache()
        return {
            "S": S, "B": B, "oom": True,
            "latency_s": None, "memory_bytes": None,
        }

rows = []
for batch_size in batch_sizes:
    for image_index in range(len(dataset)):
        measure_results = measure(model, dataset, image_index, batch_size)
        rows.append(measure_results)
        print(batch_size, image_sizes[image_index])

pd.DataFrame(rows).to_csv("results/measurements.csv", index=False)
