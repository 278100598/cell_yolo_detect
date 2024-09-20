import os
import pickle

import torch

path = "/home/raymond0920/yolov4/PyTorch_YOLOv4/runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize2/weights/best.pt"
x = torch.load(path)
print(x['epoch'], x['time'])
