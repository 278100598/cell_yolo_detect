python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "1-1 Our Aug" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/1-1 Our Aug.yaml" --hyp ./no_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "1-2 Our Aug + Gussian" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/1-2 Our Aug + Gussian.yaml" --hyp ./no_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "1-3 Our Aug + Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/1-3 Our Aug + Normalize.yaml" --hyp ./no_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "1-4 Our Aug + Gussian + Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/1-4 Our Aug + Gussian + Normalize.yaml" --hyp ./no_aug_hyp.yaml --epochs 1500 



python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "2-1 Our Aug+v8 Aug" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/2-1 Our Aug+v8 Aug.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "2-2 Our Aug+v8 Aug + Gussian" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/2-2 Our Aug+v8 Aug + Gussian.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "2-3 Our Aug+v8 Aug+ Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/2-3 Our Aug+v8 Aug+ Normalize.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "2-4 Our Aug+v8 Aug + Gussian + Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/2-4 Our Aug+v8 Aug + Gussian + Normalize.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 



python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "3-1 v8 Aug" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/3-1 v8 Aug.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "3-2 v8 Aug + Gussian" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/3-2 v8 Aug + Gussian.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yyolov4.weights --workers 8 --device 0 --batch-size 2 --name "3-3 v8 Aug+ Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/3-3 v8 Aug+ Normalize.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 

python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "3-4 v8 Aug + Gussian + Normalize" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/3-4 v8 Aug + Gussian + Normalize.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500 


python train.py --weights yolov4.weights --workers 8 --device 0 --batch-size 2 --name "" --img 1024 1024 --cfg cfg/yolo-cell-mish.cfg --data "./data/new 2-4.yaml" --hyp ./yolov8_aug_hyp.yaml --epochs 1500