import os

import torch
from ultralytics import YOLO

print(torch.cuda.is_available())

idx = 3

exps = [
    "2-1 Our Aug+v8 Aug",
    "2-2 Our Aug+v8 Aug + Gussian",
    "2-3 Our Aug+v8 Aug+ Normalize",
    "2-4 Our Aug+v8 Aug + Gussian + Normalize",
]
model = YOLO(f'/home/raymond0920/cell_yolo_detect/exp_results/new_datasets_result/v10/2-1 Our Aug+v8 Aug/weights/best.pt')

for image in os.listdir("/home/raymond0920/cell_yolo_detect/datasets/3d_cell/"):
    print("image:", image)
    ret = model.predict(f"/home/raymond0920/cell_yolo_detect/datasets/3d_cell/{image}", 
                    imgsz=1024, epochs=6000, patience=3000, workers=0, batch=2, 
                    pretrained=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=180, 
                    translate=0.5, scale=0, shear=0, perspective=0, flipud=0.5, 
                    fliplr=0.5, mosaic=1.0, mixup=0.5, copy_paste=0, conf=0.3, 
                    max_det=3000, single_cls=False, seed=0, half=True, 
                    save_txt=True, save_conf=True)

#print(ret)

