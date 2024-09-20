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

for idx, exp in enumerate(exps[4:8]):
    test = tests[idx]
    command = [
        "python", "test.py",
        "--weights", f"/mnt/backups/xiang1078/yolov7/result epoch 1500/{exp}/weights/best.pt",
        "--data", f"./config/{exp}.yaml",
        "--batch-size", "2",
        "--img-size", "1024",
        "--device", "0",
        "--name", exp,
        "--conf-thres", "0.001",
        "--iou-thres", "0.6"
    ]
    result = subprocess.run(command, env={"CONF_VALUE":"233", **os.environ})
    
    command = [
        "python", "test.py",
        "--weights", f"/mnt/backups/xiang1078/yolov7/result epoch 1500/{exp}/weights/best.pt",
        "--data", f"./config/{test}.yaml",
        "--batch-size", "2",
        "--img-size", "1024",
        "--device", "0",
        "--name", test,
        "--conf-thres", "0.001",
        "--iou-thres", "0.6"
    ]
    result = subprocess.run(command, env={"CONF_VALUE":"233", **os.environ})
    
