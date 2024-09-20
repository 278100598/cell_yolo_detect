import os
import shutil
from tqdm import tqdm
import cv2

original_path = "./original"
"""
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
"""
'''
txt = []

for i in tqdm(os.listdir('./blood')):
    if i.endswith('txt'):
        continue
    img = cv2.imread(f'./blood/{i}')
    
    img = cv2.normalize(img,None,alpha=240.9472,beta=71.5264,norm_type=cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img,(13,13),sigmaX=0.0,sigmaY=0.0)
    img = cv2.Canny(img, threshold1=26.3424, threshold2=22.5792,apertureSize=3)


    cv2.imwrite(f'./datasets/blood/valid/{i}', img)
    shutil.copyfile(f'./blood/{i.split(".")[0]}.txt', f'./datasets/blood/valid/{i.split(".")[0]}.txt')
    
    txt.append(f'./valid/{i}\n')

with open('./datasets/blood/valid.txt', 'w') as f:
    f.writelines(txt)
txt = []

for i in tqdm(os.listdir('./livecell/test')):
    if i.endswith('txt'):
        continue
    img = cv2.imread(f'./livecell/test/{i}')
    
    img = cv2.normalize(img,None,alpha=240.9472,beta=71.5264,norm_type=cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img,(13,13),sigmaX=0.0,sigmaY=0.0)
    img = cv2.Canny(img, threshold1=26.3424, threshold2=22.5792,apertureSize=3)


    cv2.imwrite(f'./datasets/livecell/valid/{i}', img)
    shutil.copyfile(f'./livecell/test/{i.split(".")[0]}.txt', f'./datasets/livecell/valid/{i.split(".")[0]}.txt')
    
    txt.append(f'./valid/{i}\n')

with open('./datasets/livecell/valid.txt', 'w') as f:
    f.writelines(txt)
'''


for file in os.listdir('./3d_data'):
    img = cv2.imread(f'./3d_data/{file}')
    #img[:,:,1:] = 0
    img = img[:,:,0]
    #ret, img  = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    cv2.imwrite(f'./3d_data_gray/{os.path.splitext(file)[0]}.jpg', img)