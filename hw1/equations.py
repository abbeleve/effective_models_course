import numpy as np

KERNEL_COUNT = 17
# PyTorch allocates the cuBLAS workspace through its caching allocator on the first
# Linear call, so it is counted by max_memory_allocated(). Measured on RTX 5060 Ti
# (sm_120, torch 2.14): exactly 32 MiB = 2**25 bytes; set by CUBLAS_WORKSPACE_CONFIG.
CUBLAS_WORKSPACE_BYTES = 32 * 2**20

def flops(image_size, batch): # -> float (число FLOPs)
    res = 7 * 7 * 3 * 2 * image_size**2 // 4 * 32 # first layer x = self.maxpool1(self.act1(self.conv1(x)))
    image_size = image_size // 4
    res = res + 5 * 5 * 32 * 64 * image_size**2 * 2 # second layer x = self.act1(self.conv2(x))
    res = res + 3 * 3 * 64 * 128 * image_size**2 // 4  * 2 # third layer x = self.act1(self.conv3(x))
    image_size = image_size // 2
    res = res + 1 * 1 * 128 * 256 * image_size**2 * 2# forth layer x = self.act1(self.conv4(x))
    res = res + 3 * 3 * 256 * 256 * image_size**2 // 4 * 2# fifth layer x = self.act1(self.conv5(x))
    image_size = image_size // 2
    res = res + 1 * 1 * 256 * 512 * image_size**2 * 2# sixth layer x = self.act1(self.conv6(x))
    res = res + 512 * image_size**2 # seventh layer x = self.gap(x)
    res = res + 512 * 256 * 2 + 256 * 100 * 2# fc layer x = self.fc(x)
    return res * batch


def memory(image_size, batch): # -> float (байты)
    parameters_amount = 7 * 7 * 32 * 3 + 5 * 5 * 64 * 32 + 3 * 3 * 128 * 64 + 1 * 1 * 256 * 128 + 3 * 3 * 256 * 256 + 1 * 1 * 512 * 256 + (512 + 1) * 256 + (256 + 1) * 100
    parameters_in_bytes = parameters_amount * 4
    # peak is inside maxpool1: input (3 S^2) + conv1 output (8 S^2) + maxpool output (2 S^2) floats
    memory_for_image = (3 + 8 + 2) * batch * image_size ** 2 * 4
    # CUDA max_pool2d always builds int64 argmax indices, one per maxpool output element
    maxpool_indices = 2 * batch * image_size ** 2 * 8
    return parameters_in_bytes + CUBLAS_WORKSPACE_BYTES + memory_for_image + maxpool_indices

def data_moved(image_size, batch):
    return 4 * (91 * batch * image_size**2 + 2148 * batch + 1040324)

def latency(image_size, batch, theta): # -> float (секунды)
    return KERNEL_COUNT * theta["tau_s_per_launch"] + np.maximum(flops(image_size, batch) / theta["flops_per_s"],
    data_moved(image_size, batch) / theta["bytes_per_s"],)


def energy(image_size, batch, theta_energy): # -> float (джоули)
    return (theta_energy["base_power_w"] * latency(image_size, batch, theta_energy) + theta_energy["joules_per_gflop"]
        * flops(image_size, batch) / 1e9
        + theta_energy["joules_per_gb"]
        * data_moved(image_size, batch) / 1e9
    )