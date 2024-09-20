import torch
import os

path = '/home/raymond0920/yolov7_xiang/yolov7-main/runs/train/'
for exp in os.listdir(path):
  best = os.path.join(path,exp,'weights','best.pt')
  x = torch.load(best)
  print(exp, x['epoch'], x['time'])