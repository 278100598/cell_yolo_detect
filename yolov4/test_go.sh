#python test.py --weights "runs/train/1-1 Our Aug/weights/best.pt" --data "./data/1-1 Our Aug.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_1-1" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/1-2 Our Aug + Gussian/weights/best.pt" --data "./data/1-2 Our Aug + Gussian.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_1-2" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/1-3 Our Aug + Normalize/weights/best.pt" --data "./data/1-3 Our Aug + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_1-3" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/1-4 Our Aug + Gussian + Normalize/weights/best.pt" --data "./data/1-4 Our Aug + Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_1-4" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-1 Our Aug+v8 Aug/weights/best.pt" --data "./data/2-1 Our Aug+v8 Aug.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_2-1" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-2 Our Aug+v8 Aug + Gussian/weights/best.pt" --data "./data/2-2 Our Aug+v8 Aug + Gussian.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_2-2" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-3 Our Aug+v8 Aug+ Normalize/weights/best.pt" --data "./data/2-3 Our Aug+v8 Aug+ Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_2-3" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/2-4 Our Aug+v8 Aug + Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_2-4" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/3-1 v8 Aug/weights/best.pt" --data "./data/3-1 v8 Aug.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_3-1" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/3-2 v8 Aug + Gussian/weights/best.pt" --data "./data/3-2 v8 Aug + Gussian.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_3-2" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/3-3 v8 Aug+ Normalize/weights/best.pt" --data "./data/3-3 v8 Aug+ Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_3-3" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/3-4 v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/3-4 v8 Aug + Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "nexp_3-4" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-1 Our Aug+v8 Aug/weights/best.pt" --data "./data/TEST-1 Original.yaml" --batch-size 2 --img-size 1024 --device 0 --name "TEST-1 Original" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3
#python test.py --weights "runs/train/2-2 Our Aug+v8 Aug + Gussian/weights/best.pt" --data "./data/TEST-2 Gussian.yaml" --batch-size 2 --img-size 1024 --device 0 --name "TEST-2 Gussian" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3
#python test.py --weights "runs/train/2-3 Our Aug+v8 Aug+ Normalize/weights/best.pt" --data "./data/TEST-3 Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "TEST-3 Normalize" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3
python test.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/TEST-4 Gussian + Normalize.yaml" --batch-size 2 --img-size 1024 --device 0 --name "TEST-4 Gussian + Normalize" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/3d_cell.yaml" --batch-size 2 --img-size 1024 --device 0 --name "3d_cell_preprocessed" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/livecell.yaml" --batch-size 2 --img-size 1024 --device 0 --name "livecell_preprocessed" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3

#python test.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --data "./data/bloodcell.yaml" --batch-size 2 --img-size 1024 --device 0 --name "bloodcell_preprocessed" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3