import subprocess
import os

exps = [
    "1-1 Our Aug",
    "1-2 Our Aug + Gussian",
    "1-3 Our Aug + Normalize",
    "1-4 Our Aug + Gussian + Normalize",
    "2-1 Our Aug+v8 Aug",
    "2-2 Our Aug+v8 Aug + Gussian",
    "2-3 Our Aug+v8 Aug+ Normalize",
    "2-4 Our Aug+v8 Aug + Gussian + Normalize",
    "3-1 v8 Aug",
    "3-2 v8 Aug + Gussian",
    "3-3 v8 Aug+ Normalize",
    "3-4 v8 Aug + Gussian + Normalize",
]

tests = [
    "TEST-1 Original",
    "TEST-2 Gussian",
    "TEST-3 Normalize",
    "TEST-4 Gussian + Normalize",
]

for epoch in [1000, 2000, 4000, 8000, 16000]:
    for idx, exp in enumerate(exps[4:5]):
        command = [
            "python", "train.py",
            "--weights", "yolov7.pt",
            "--device", "0",
            "--batch-size", "2",
            "--name", f"yolov7_epoch{epoch} {exp}",
            "--img", "1024", "1024",
            "--cfg", "./yolov7.yaml",
            "--data", f"./config/new {exp}.yaml",
            "--hyp", "./no_aug_hyp.yaml" if exp in [exps[:4]] else "./yolov8_aug_hyp.yaml",
            "--epochs", str(epoch),
        ]
        result = subprocess.run(command)
    
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "new 2-4" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/new 2-4.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 500
