import random
from typing import List

import cv2
import os
import albumentations as A
import numpy as np
from albumentations.augmentations.geometric.rotate import SafeRotate, Rotate
from albumentations.augmentations.geometric.transforms import HorizontalFlip , VerticalFlip
import random


# 把0~1 float表示的xywh座標變成0~hh,0~ww int表示的xyxy座標，hh是圖高，ww是圖寬
def xywh_to_xyxy(x: float, y: float, w: float, h: float, hh: int, ww: int) -> (int, int, int, int):
    return round((x - w / 2) * ww), round((y - h / 2) * hh), round((x + w / 2) * ww), round((y + h / 2) * hh)


# 把img和其對應的label畫出來label的格式是cls,x1,y1,x2,y2
def show(img: np.ndarray, labels: List[List[int]]):
    cp = img.copy()
    for l in labels:
        cls, x1, y1, x2, y2 = l
        start = [x1, y1]
        end = [x2, y2]

        if cls == 0:
            color = (0, 255, 255)
        elif cls == 1:
            color = (0, 0, 255)
        elif cls == 2:
            color = (255, 0, 0)
        cv2.rectangle(cp, start, end, color)
    cv2.imshow('www', cv2.resize(cp, (512, 512)))
    cv2.waitKey()


# 用來確定目前這個label選取的位置會不會撞到其他已經選好了的label
def check(label: List[int]) -> bool:
    global final
    for i in range(len(final)):
        if final[i][1] < label[3] and label[1] < final[i][3] and final[i][2] < label[4] and label[2] < final[i][4]:
            return True
    return False


# 用來把有重疊到的細胞的bbox去除後回傳沒有與其他細胞重疊的那些
def remove_overlap(df: List[List[int]], gap: float) -> List[List[int]]:
    ret = []
    for i in range(len(df)):
        flag = True
        for j in range(len(df)):
            if i == j:
                continue
            if df[i][1] < df[j][3]+gap and df[j][1] < df[i][3]+gap and df[i][2] < df[j][4]+gap and df[j][2] < df[i][4]+gap:
                flag = False
                break
        if flag:
            ret.append(df[i])
    return ret

# 把b圖片的的bbox內的像素貼到a圖片上
def paste(a: np.ndarray, b: np.ndarray, bbox: List[int], dx: int, dy: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    a[dy + y1:dy + y2, dx + x1:dx + x2] = b[y1:y2, x1:x2]
    return a

#找出細胞的mask
def find_mask(img:np.ndarray):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    binaryIMG = cv2.Canny(blurred, 20, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binaryIMG = cv2.dilate(binaryIMG, kernel)  # 擴張
    binaryIMG = cv2.erode(binaryIMG, kernel)  # 侵蝕
    cnts, _ = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = cv2.drawContours(binaryIMG, cnts, -1, 255, -1)
    return mask

#去除細胞的背景
def remove_background(img:np.ndarray, bbox:list):
    object_img = paste(np.zeros(img.shape, dtype="uint8"), img, bbox, dx=0, dy=0)
    mask = find_mask(object_img)
    #cv2.drawContours(object_img, cnts, -1,(0,255,0),2)
    return object_img
    #return cv2.bitwise_and(object_img, object_img, mask=mask)

# 這個是會圖片去做旋轉，會連bbox一起變化
aug = A.Compose([Rotate(limit=180, p=1, border_mode=cv2.BORDER_TRANSPARENT),
                 HorizontalFlip(p=0.5),
                 VerticalFlip(p=0.5)],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls']))





# 把b圖片上的label找一個好位置貼到a圖片上去，如果在random位置的時候有重疊就會print "re"
def rand_paste(a: np.ndarray, b: np.ndarray, label: List[int]) -> [np.ndarray, [int, int, int, int, int]]:
    b = remove_background(b, label[1:])

    x1, y1, x2, y2 = label[1:]
    dx = round(b.shape[1]/2-(x1+x2)/2)
    dy = round(b.shape[0]/2-(y1+y2)/2)
    image = paste(np.zeros(b.shape, dtype="uint8"), b, label[1:], dx=dx, dy=dy)

    ret = aug(image=image, bboxes=[[x1+dx, y1+dy, x2+dx, y2+dy]], cls=[label[0]])
    bbox = [*map(round, ret['bboxes'][0])]
    cls = int(ret['cls'][0])
    xx, yy = ret['image'].shape[:2]

    #可蓋住模式
    """
    dx = np.random.randint(low=-bbox[0], high=xx - bbox[2])
    dy = np.random.randint(low=-bbox[1], high=yy - bbox[3])
    b = paste(np.zeros(b.shape, dtype="uint8"), ret['image'], bbox, dx, dy)
    mask = find_mask(b)
    a = cv2.subtract(a,cv2.bitwise_and(a,a,mask=mask))
    img = cv2.add(a, b)
    """

    while 1:
        dx = np.random.randint(low=-bbox[0], high=xx - bbox[2])
        dy = np.random.randint(low=-bbox[1], high=yy - bbox[3])
        if check([cls, bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]):
            continue
        img = paste(a, ret["image"], bbox, dx, dy)
        break
    # show(img,[[cls,bbox[0]+x,bbox[1]+y,bbox[2]+x,bbox[3]+y]])
    return img, [cls, bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]


final = []


def run(path: str, target: List[str], gap:int, show_img:bool, cell_num: list, cell_per_image:int):
    global final
    # 存yellow細胞的label
    yellow = []
    # 存blue細胞的label
    blue = []
    # 存red細胞的label
    red = []
    # 暫存圖片，雖然只有兩張圖片XD
    save = {}


    # 把0.jpg還有2.jpg讀近來並去除掉重疊的細胞，把blue跟red分別存好
    for i in target:
        img = cv2.imread(f'{path}/{i}.jpg')
        save[i] = img
        hh, ww, dd = img.shape
        labels = []
        with open(f'{path}/{i}.txt') as f:
            for l in f.readlines():
                cls, x, y, w, h = map(float, l.split(' '))
                x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h, hh, ww)
                labels.append([cls, x1, y1, x2, y2])
                cell_num[int(cls)] -= 1

        labels = remove_overlap(labels, gap=gap)
        blue.extend([[i, *x] for x in labels if x[0] == 2])
        red.extend([[i, *x] for x in labels if x[0] == 1])
        yellow.extend([[i, *x] for x in labels if x[0] == 0])

        if show_img:
            show(img, labels=labels)

    # 取名的數字
    cnt = 6
    # 生成8張，在黑色背景上貼上所有blue細胞跟與blue細胞相同數量的red細胞
    while sum(cell_num) != 0:
        print(f"remain cell_num: {cell_num}, cnt: {cnt}")

        cnt += 1
        bb = np.zeros(img.shape, dtype=np.uint8)
        final = []

        rand_cell_num = random.sample([*(["yellow"] * cell_num[0]), *(["red"] * cell_num[1]), *(["blue"] * cell_num[2])], k=min(cell_per_image,sum(cell_num)))
        for l in random.choices(blue, k=rand_cell_num.count("blue")):
            bb, x = rand_paste(bb, save[l[0]], l[1:])
            final.append(x)
            cell_num[2] -= 1
        for l in random.choices(red, k=rand_cell_num.count("red")):
            bb, x = rand_paste(bb, save[l[0]], l[1:])
            final.append(x)
            cell_num[1] -= 1
        for l in random.choices(yellow, k=rand_cell_num.count("yellow")):
            bb, x = rand_paste(bb, save[l[0]], l[1:])
            final.append(x)
            cell_num[0] -= 1
        #print(cell_num)
        if show_img:
            show(bb, final)

        cv2.imwrite(f'./datasets/augment/{cnt}.jpg', bb)
        with open(f'./datasets/augment/{cnt}.txt', 'w') as f:
            for l in final:
                h, w = float(bb.shape[0]), float(bb.shape[1])
                x = (l[1] + l[3]) / (2 * w)
                y = (l[2] + l[4]) / (2 * h)
                hh = (l[3] - l[1]) / (h)
                ww = (l[4] - l[2]) / (w)
                f.write(f'{l[0]} {x} {y} {ww} {hh}\n')




if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    path = 'original'
    for file in os.listdir('./datasets/augment'):
        os.remove(os.path.join('./datasets/augment',file))

    run(path, ['0', '2', '4', '5'], gap=10, show_img=False, cell_num=[1200,1200,1200],cell_per_image=100)






