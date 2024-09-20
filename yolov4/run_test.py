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

os.environ['CONF_VALUE'] = "319"

for idx, exp in enumerate(exps[4:8]):
    test = tests[idx % 4]
    
    command = [
        "python", "test.py",
        "--weights", f"runs/train/{exp}/weights/best.pt",
        "--data", f"./data/{test}.yaml",
        "--batch-size", "2",
        "--img-size", "1024",
        "--device", "0",
        "--name", test,
        "--cfg", "cfg/yolo-cell-mish.cfg",
        "--conf-thres", "0.3",
    ]
    result = subprocess.run(command, env=os.environ)
    
    command = [
        "python", "test.py",
        "--weights", f"runs/train/{exp}/weights/best.pt",
        "--data", f"./data/{exp}.yaml",
        "--batch-size", "2",
        "--img-size", "1024",
        "--device", "0",
        "--name", exp,
        "--cfg", "cfg/yolo-cell-mish.cfg",
        "--conf-thres", "0.3",
    ]
    result = subprocess.run(command, env=os.environ)