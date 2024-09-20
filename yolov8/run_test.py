import os
from ultralytics import YOLO

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


#os.environ['ALGORITHM'] = 'valid'
for idx, exp in enumerate(exps[4:5]):
    test = tests[idx%4]
    model = YOLO(f'./result epoch 6000/{exp}/weights/best.pt')
    model.val(name=exp, data=f'config/{exp}.yaml', imgsz=1024, epochs=6000, patience=3000, workers=0, batch=2, pretrained=True, 
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=180, translate=0.5, scale=0, shear=0, perspective=0, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.5, copy_paste=0,
            conf=0.3, max_det=3000, single_cls=False, seed=0,half=True)
    
#os.environ['ALGORITHM'] = 'test'
for idx, exp in enumerate(exps[4:5]):
    test = tests[idx%4]
    model = YOLO(f'./result epoch 6000/{exp}/weights/best.pt')
    model.val(name=test, data=f'config/{test}.yaml', imgsz=1024, epochs=6000, patience=3000, workers=0, batch=2, pretrained=True, 
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=180, translate=0.5, scale=0, shear=0, perspective=0, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.5, copy_paste=0,
            conf=0.3, max_det=3000, single_cls=False, seed=0,half=True)