#!/bin/bash
# "runs/conf 0.001 max_det 300 train/$name/weights/best.pt" "runs/conf 0.001/$name.pt"

for filename in ./config/*; do
    name=${filename##*/}
    name=${name%.*}
    python test.py --weights "runs/train/$name/weights/best.pt" --data "./config/$name.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_$name" --conf-thres 0.001 --iou-thres 0.6
done

#python test.py --weights "/home/raymond0920/yolov7_xiang/yolov7-main/runs/train/2-1/weights/best.pt"  --data "./config/TEST-1 Original.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_TEST_1" --conf-thres 0.001 --iou-thres 0.6
#python test.py --weights "/mnt/backups/xiang1078/yolov7/result epoch 1500/2-2 Our Aug+v8 Aug + Gussian/weights/best.pt"  --data "./config/TEST-2 Gussian.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_TEST_2" --conf-thres 0.001 --iou-thres 0.6
#python test.py --weights "/mnt/backups/xiang1078/yolov7/result epoch 1500/2-3 Our Aug+v8 Aug+ Normalize/weights/best.pt"  --data "./config/TEST-3 Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_TEST_3" --conf-thres 0.001 --iou-thres 0.6
#python test.py --weights "/mnt/backups/xiang1078/yolov7/result epoch 1500/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt"  --data "./config/TEST-4 Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_TEST_4" --conf-thres 0.001 --iou-thres 0.6
#python test.py --weights ./best.pt  --data "./config/3d_test.yaml" --batch-size 2 --img-size 1024 --device 0 --name "3d_test" --conf-thres 0.001 --iou-thres 0.6

#python test.py --weights "./best.pt"  --data "./config/TEST-4 Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "exp_TEST_4" --conf-thres 0.001 --iou-thres 0.6
