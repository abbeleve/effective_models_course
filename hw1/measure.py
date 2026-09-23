import torch
import torch.nn as nn
from models import CNN

model = CNN()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model = model.cuda().eval()