import os
from typing import List

import cv2
import numpy as np
import albumentations as A
from albumentations import Rotate, HorizontalFlip, VerticalFlip
import random

class OurCellAug:
    def __init__(self, imgsz:int):
        self.aug = A.Compose([Rotate(limit=180, p=1, border_mode=cv2.BORDER_TRANSPARENT),
                 HorizontalFlip(p=0.5),
                 VerticalFlip(p=0.5)],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls']))
        self.yellow = []
        self.blue = []
        self.red = []
        self.save = {}
        self.ori_shape = None

        PATH = "./augment_base_img"
        GAP = 10
        for file in os.listdir(PATH):
            if not file.endswith(".txt"):
                continue

            idx = file.split('.')[0]
            img = cv2.imread(os.path.join(PATH, f"{idx}.jpg"))
            self.ori_shape = img.shape[:2]
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            self.save[idx] = img

            hh, ww, dd = img.shape
            labels = []
            with open(os.path.join(PATH,file)) as f:
                for l in f.readlines():
                    cls, x, y, w, h = map(float, l.split(' '))
                    x1, y1, x2, y2 = self.xywh_to_xyxy(x, y, w, h, hh, ww)
                    labels.append([cls, x1, y1, x2, y2])

            labels = self.remove_overlap(labels, gap=GAP)
            self.blue.extend([[idx, *x] for x in labels if x[0] == 2])
            self.red.extend([[idx, *x] for x in labels if x[0] == 1])
            self.yellow.extend([[idx, *x] for x in labels if x[0] == 0])

    def xywh_to_xyxy(self, x: float, y: float, w: float, h: float, hh: int, ww: int) -> (int, int, int, int):
        return round((x - w / 2) * ww), round((y - h / 2) * hh), round((x + w / 2) * ww), round((y + h / 2) * hh)

    def remove_overlap(self, df: List[List[int]], gap: float) -> List[List[int]]:
        ret = []
        for i in range(len(df)):
            flag = True
            for j in range(len(df)):
                if i != j:
                    if df[i][1] < df[j][3] + gap and df[j][1] < df[i][3] + gap and df[i][2] < df[j][4] + gap and df[j][2] < \
                            df[i][4] + gap:
                        flag = False
                        break
            if flag:
                ret.append(df[i])
        return ret

    def paste(self, a: np.ndarray, b: np.ndarray, bbox: List[int], dx: int, dy: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        a[dy + y1:dy + y2, dx + x1:dx + x2] = b[y1:y2, x1:x2]
        return a

    def check(self, label: List[int]) -> bool:
        for f_label in self.final:
            if f_label[1] < label[3] and label[1] < f_label[3] and f_label[2] < label[4] and label[2] < f_label[4]:
                return True
        return False

    def rand_paste(self, a: np.ndarray, b: np.ndarray, label: List[int]) -> [np.ndarray, [int, int, int, int, int]]:
        x1, y1, x2, y2 = label[1:]
        dx = round(b.shape[1] / 2 - (x1 + x2) / 2)
        dy = round(b.shape[0] / 2 - (y1 + y2) / 2)
        image = self.paste(np.zeros(b.shape, dtype="uint8"), b, label[1:], dx=dx, dy=dy)

        ret = self.aug(image=image, bboxes=[[x1 + dx, y1 + dy, x2 + dx, y2 + dy]], cls=[label[0]])
        bbox = [*map(round, ret['bboxes'][0])]
        cls = int(ret['cls'][0])
        xx, yy = ret['image'].shape[:2]

        while 1:
            dx = np.random.randint(low=-bbox[0], high=xx - bbox[2])
            dy = np.random.randint(low=-bbox[1], high=yy - bbox[3])
            if self.check([cls, bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]):
                continue
            img = self.paste(a, ret["image"], bbox, dx, dy)
            break
        # show(img,[[cls,bbox[0]+x,bbox[1]+y,bbox[2]+x,bbox[3]+y]])
        return img, [cls, bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

    def generate_one_image_label(self, cell_per_image:int):
        self.final = []
        image = np.zeros(list(self.save.items())[0][1].shape, dtype=np.uint8)
        rand_cell_num = random.sample([*(["yellow"] * cell_per_image), *(["red"] * cell_per_image), *(["blue"] * cell_per_image)],k=cell_per_image)

        for l in random.choices(self.blue, k=rand_cell_num.count("blue")):
            image, x = self.rand_paste(image, self.save[l[0]], l[1:])
            self.final.append(x)
        for l in random.choices(self.red, k=rand_cell_num.count("red")):
            image, x = self.rand_paste(image, self.save[l[0]], l[1:])
            self.final.append(x)
        for l in random.choices(self.yellow, k=rand_cell_num.count("yellow")):
            image, x = self.rand_paste(image, self.save[l[0]], l[1:])
            self.final.append(x)

        for i, l in enumerate(self.final):
            h, w = float(image.shape[0]), float(image.shape[1])
            x = (l[1] + l[3]) / (2 * w)
            y = (l[2] + l[4]) / (2 * h)
            hh = (l[3] - l[1]) / (h)
            ww = (l[4] - l[2]) / (w)
            self.final[i] = [l[0], x, y, ww, hh]

        return {
            "img": image,
            "ori_shape": self.ori_shape,
            "resized_shape": image.shape[:2],
            "cls": np.array([[l[0]] for l in self.final]),
            "ratio_pad": (self.ori_shape[0] / image.shape[0], self.ori_shape[1] / image.shape[1]),
            "bboxes": np.array([l[1:] for l in self.final]),
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "segments": None,
            "im_file": "Our_aug",
        }




