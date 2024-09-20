#python detect.py --weights "runs/train/2-4 Our Aug+v8 Aug + Gussian + Normalize/weights/best.pt" --source "../datasets/new_3d_preprocess/" --output "./new_3d_predict_label/" --img-size 1024 --device 0 --names "./data/cell.names" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3 --save-txt


#python detect.py --weights "runs/train/2-4/weights/best.pt" --source "../datasets/new_3d_preprocess/" --img-size 1024 --device 0 --name "new_3d_cell_label" --conf-thres 0.3 --save-txt

#python detect.py --weights "runs/train/weights/best.pt" --source "../datasets/new_3d_preprocess/" --output "./new_3d_predict_label_v3/" --img-size 1024 --device 0 --names "./data/cell.names" --cfg "cfg/yolo-cell-mish.cfg" --conf-thres 0.3 --save-txt

experiments=(
    "2-1 Our Aug+v8 Aug"
    "2-2 Our Aug+v8 Aug + Gussian"
    "2-3 Our Aug+v8 Aug+ Normalize"
    "2-4 Our Aug+v8 Aug + Gussian + Normalize"
)

weights_base_path="/home/raymond0920/cell_yolo_detect/new_datasets_result/v4"
img_size=1024
device=0
names="./data/cell.names"
cfg="cfg/yolo-cell-mish.cfg"
conf_thres=0.3

for exp in "${experiments[@]}"; do
    echo "exp: $exp"

    weights_path="$weights_base_path/$exp/weights/best.pt"
    output_path="./new_3d_predict_label_v4/$exp/"
    
    if [[ "$exp" == "2-1 Our Aug+v8 Aug" ]]; then
        source_path="../datasets/3d_cell/"
    else
        source_path="../datasets/3d_cell_preprocess_v3/"
    fi
    
    mkdir -p "$output_path"

    python detect.py --weights "$weights_path" \
                     --source "$source_path" \
                     --output "$output_path" \
                     --img-size $img_size \
                     --device $device \
                     --names "$names" \
                     --cfg "$cfg" \
                     --conf-thres $conf_thres \
                     --save-txt

done

echo "all done"

