'''
create sample for training
'''

import os
import cv2
import numpy as np
import random
from findFace import transFace
from loadBasicEnv import IMGSHAPE, OUTPUT_NUM, OUTPUT_DIR, SAMPLE_DIR


filenames = [x for x in os.listdir(SAMPLE_DIR) if x.endswith('jpg')]
LENGTH = len(filenames)


class TransImg:
    ITERMAXDEPTH = 4
    OUTPUTSIZE = (IMGSHAPE, IMGSHAPE)

    def __init__(self, img):
        img = transFace(img)
        self.img = img
        self.img_backup = img.copy()
        self.size = img.shape

    def Contrast_and_Brightness(self, img, alpha, beta):
        blank = np.zeros(img.shape, img.dtype)
        dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
        return dst

    def flip(self, img, dirction):
        return cv2.flip(img, dirction)

    def trans_getRandomRangeXY(self, img):
        H, W = img.shape
        random_rangeH = 0.1 * H
        random_rangeW = 0.1 * W
        randomH0 = random.randint(0, random_rangeH)
        randomH1 = random.randint(0, random_rangeH)
        randomW0 = random.randint(0, random_rangeW)
        randomW1 = random.randint(0, random_rangeW)
        return img[randomH0:H-randomH1, randomW0:W-randomW1]

    def trans_flip(self, img):
        return self.flip(img, 0)

    def trans_flip1(self, img):
        return self.flip(img, 1)

    def trans_flip2(self, img):
        return self.flip(img, -1)

    def trans_increaseValue(self, img):
        return self.Contrast_and_Brightness(img, alpha=1, beta=25)

    def trans_increaseValue2(self, img):
        return self.Contrast_and_Brightness(img, alpha=1, beta=50)

    def trans_increaseValue3(self, img):
        return self.Contrast_and_Brightness(img, alpha=1.1, beta=0)

    def trans_decreaseValue(self, img):
        return self.Contrast_and_Brightness(img, alpha=1, beta=-25)

    def trans_decreaseValue2(self, img):
        return self.Contrast_and_Brightness(img, alpha=1, beta=-50)

    def trans_decreaseValue2(self, img):
        return self.Contrast_and_Brightness(img, alpha=0.9, beta=0)

    def trans_PepperandSalt(self, src, percetage=0.005):
        NoiseImg=src
        NoiseNum=int(percetage*src.shape[0]*src.shape[1])
        for i in range(NoiseNum):
            randX=random.randint(0,src.shape[0]-1)
            randY=random.randint(0,src.shape[1]-1)
            if random.randint(0,1)<=0.5:
                NoiseImg[randX,randY]=0
            else:
                NoiseImg[randX,randY]=255
        return NoiseImg

    def trans_PepperandSalt2(self, src, percetage=0.01):
        NoiseImg=src
        NoiseNum=int(percetage*src.shape[0]*src.shape[1])
        for i in range(NoiseNum):
            randX=random.randint(0,src.shape[0]-1)
            randY=random.randint(0,src.shape[1]-1)
            if random.randint(0,1)<=0.5:
                NoiseImg[randX,randY]=0
            else:
                NoiseImg[randX,randY]=255
        return NoiseImg

    def trans_GaussNoise(self, image, mean=0, var=0.0002):
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

    def trans_transNoise(self, img):
        h, w = img.shape
        img = cv2.resize(img, (100, 100))
        return cv2.resize(img, (h, w))

    def trans_transNoise2(self, img):
        h, w = img.shape
        img = cv2.resize(img, (200, 200))
        return cv2.resize(img, (h, w))

    def selfFunctions(self):
        tmp = []
        for item in self.__dir__():
            if item.startswith('trans'):
                tmp.append(item)
        return tmp

    def genRandomTreeOfFunctions(self, lst):
        MAXDEPTH = self.ITERMAXDEPTH
        TREELENGTH = len(self.selfFunctions())
        TreeIndex = random.sample(range(TREELENGTH), random.randint(1, MAXDEPTH))
        tmp = []
        for index in TreeIndex:
            tmp.append(self.selfFunctions()[index])
        return tmp

    def compose(self, lst):
        img = self.img
        for func in lst:
            s = f"self.{func}(img)"
            #  print(s)
            img = eval(s)
        self.clean()
        #  return cv2.resize(img, self.size[::-1])
        return cv2.resize(img, self.OUTPUTSIZE)

    def getProcessedImg(self):
        return self.compose(self.genRandomTreeOfFunctions(self.selfFunctions()))

    def imgSeries(self, N=1):
        for i in range(N):
            yield self.getProcessedImg()

    def clean(self):
        self.img = self.img_backup.copy()


def getReady(imgName):
    print("processing imagine", imgName)
    img = cv2.imread(f"{SAMPLE_DIR}/{imgName}.jpg", 0)
    test = TransImg(img)
    for i, item in enumerate(test.imgSeries(OUTPUT_NUM)):
        cv2.imwrite(f"{OUTPUT_DIR}/{imgName}_{i+1:04d}.jpg", item)
        if i % (OUTPUT_NUM//10) == 0:
            print(f"process {i/OUTPUT_NUM*100:.02f} %")
    else:
        print("Process Done!")

for i, filename in enumerate(filenames):
    name, _ = filename.split('.')
    getReady(name)
    print(f"Total: {100*i/LENGTH:.02f} %")
