import numpy as np
import cv2
from numpy import mean
import os
from tqdm import tqdm
from chicken_filter import chikenFilter 


def img_ok(img):
    isOk = np.mean(img) > 10 and np.mean(img) < 180
    return isOk

def imgRead(path):
    count = 0
    if not os.path.exists('data/yes/'):
        os.makedirs('data/yes/')
    if not os.path.exists('data/not/'):
        os.makedirs('data/not/')

    for i in tqdm(os.listdir(path)):
        folder = path+'/'+i+'/'
        for j in os.listdir(folder):
            imgPth = folder+j
            img = cv2.imread(imgPth)
            isOk = img_ok(img)
            if isOk:
                #cv2.imshow('image', img)
                cv2.imwrite('data/yes/img_'+str(count)+'.jpg', img)
            else:
                #cv2.imshow('notOkay', img)
                cv2.imwrite('data/not/img_'+str(count)+'.jpg', img)
            #cv2.waitKey(500)
            count += 1


img = cv2.imread('./data/All/image-0.jpg')

inter = chikenFilter.loadInterpreter()
res = chikenFilter.predictAI(img=img,interpreter=inter)
print(res)

