import os

def print_result():
    train_path = "/home/raymond0920/yolov4/PyTorch_YOLOv4/runs/train/"
    for file in os.listdir(train_path):
        result_path = os.path.join(train_path, file, "results.txt")
        with open(result_path, "r") as f:
            best = ["?/?", 0]
            for line in f.readlines():
                epoch, gpu_cost, bbox_loss, object_loss, cls_loss, tot_loos, targets_num, input_size, p, r, mp05, mp05095, val_bbox_loss, val_object_loss, val_cls_loss = line.split()
                if best[1] < float(mp05):
                    best = [epoch, float(mp05)]

        print(file)
        print(best)
        print()

if __name__=='__main__':
    print_result()