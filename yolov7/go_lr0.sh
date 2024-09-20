
python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "cls 10" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./10.yaml --epochs 6000
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "conf 3" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./3.yaml --epochs 6000
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "conf 1" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./1.yaml --epochs 6000
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "conf 0.3" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./0.3.yaml --epochs 6000
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "conf 0.03" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./0.03.yaml --epochs 6000
#python train.py --weights yolov7.pt --workers 8 --device 0 --batch-size 2 --name "conf 0.003" --img 1024 1024 --cfg ./yolov7.yaml --data "./config/1-1 Our Aug.yaml" --hyp ./0.003.yaml --epochs 6000
