import numpy as np

KERNEL_COUNT = 17

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
    memory_for_image = (3 + 8 + 2) * batch * image_size ** 2 * 4
    return parameters_in_bytes + memory_for_image

def data_moved(image_size, batch):
    return 4 * (91 * batch * image_size**2 + 2148 * batch + 1040324)

def latency(image_size, batch, theta): # -> float (секунды)
    return KERNEL_COUNT * theta["tau_s_per_launch"] + np.maximum(flops(image_size, batch) / theta["flops_per_s"],data_moved(image_size, batch) / theta["bytes_per_s"],)


def energy(image_size, batch, theta_energy): ...   # -> float (джоули)