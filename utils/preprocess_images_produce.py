import os
import shutil

import cv2

original_path = "../datasets/augment"

name = "gauss"
os.mkdir(f"../datasets/{name}")
for i in set(x.split('.')[0] for x in os.listdir(original_path)):
    img = cv2.imread(f'{original_path}/{i}.jpg', 0)
    img = cv2.GaussianBlur(img,(3,3),sigmaX=0.0,sigmaY=0.0)
    img = cv2.Canny(img, threshold1=139, threshold2=256,apertureSize=3)
    if not os.path.isdir(f"../datasets/{name}"):
        os.mkdir(f"../datasets/{name}")

    if i == "1":
        if not os.path.isdir(f"../datasets/{name}/valid"):
            os.mkdir(f"../datasets/{name}/valid")
        cv2.imwrite(f'../datasets/{name}/valid/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/valid/{i}.txt")
    else:
        if not os.path.isdir(f"../datasets/{name}/train"):
            os.mkdir(f"../datasets/{name}/train")
        cv2.imwrite(f'../datasets/{name}/train/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/train/{i}.txt")


name = "normalize"
os.mkdir(f"../datasets/{name}")
for i in set(x.split('.')[0] for x in os.listdir(original_path)):
    img = cv2.imread(f'{original_path}/{i}.jpg', 0)
    img = cv2.normalize(img, None, alpha=256, beta=15, norm_type=cv2.NORM_MINMAX)
    img = cv2.Canny(img, threshold1=139, threshold2=256,apertureSize=3)
    if not os.path.isdir(f"../datasets/{name}"):
        os.mkdir(f"../datasets/{name}")

    if i == "1":
        if not os.path.isdir(f"../datasets/{name}/valid"):
            os.mkdir(f"../datasets/{name}/valid")
        cv2.imwrite(f'../datasets/{name}/valid/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/valid/{i}.txt")
    else:
        if not os.path.isdir(f"../datasets/{name}/train"):
            os.mkdir(f"../datasets/{name}/train")
        cv2.imwrite(f'../datasets/{name}/train/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/train/{i}.txt")


name = "normalize_gauss"
os.mkdir(f"../datasets/{name}")
for i in set(x.split('.')[0] for x in os.listdir(original_path)):
    img = cv2.imread(f'{original_path}/{i}.jpg', 0)
    img = cv2.normalize(img, None, alpha=256, beta=15, norm_type=cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img,(3,3),sigmaX=0.0,sigmaY=0.0)
    img = cv2.Canny(img, threshold1=105, threshold2=256,apertureSize=3)
    if not os.path.isdir(f"../datasets/{name}"):
        os.mkdir(f"../datasets/{name}")

    if i == "1":
        if not os.path.isdir(f"../datasets/{name}/valid"):
            os.mkdir(f"../datasets/{name}/valid")
        cv2.imwrite(f'../datasets/{name}/valid/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/valid/{i}.txt")
    else:
        if not os.path.isdir(f"../datasets/{name}/train"):
            os.mkdir(f"../datasets/{name}/train")
        cv2.imwrite(f'../datasets/{name}/train/{i}.jpg', img)
        shutil.copyfile(f"{original_path}/{i}.txt", f"../datasets/{name}/train/{i}.txt")

#livecell
#cv2.normalize(src,dst,alpha=252.2368,beta=71.5264,norm_type=cv2.NORM_MINMAX)
#cv2.GaussianBlur(img,(9,9),sigmaX=0.0,sigmaY=0.0)
#cv2.Canny(img, threshold1=7.5264, threshold2=18.816,apertureSize=3)

#blood
#cv2.normalize(src,dst,alpha=240.9472,beta=71.5264,norm_type=cv2.NORM_MINMAX)
#cv2.GaussianBlur(img,(13,13),sigmaX=0.0,sigmaY=0.0)
#cv2.Canny(img, threshold1=26.3424, threshold2=22.5792,apertureSize=3)