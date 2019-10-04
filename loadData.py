import cv2
from loadBasicEnv import WORKDIR, filenames, DICT

LENGTH = len(filenames)
data = []
tags = []

for imgName in filenames:
    name, _ = imgName.split(".")
    tag, _ = name.split("_")
    tags.append(DICT[tag])
    img = cv2.imread(f"./{WORKDIR}/processed/{imgName}", 0)
    data.append(img)

def load_data():
    sp = int(LENGTH*0.8)

    xTrain = data[:sp]
    yTrain = tags[:sp]
    xTest  = data[sp:]
    yTest  = tags[sp:]
    print(len(xTrain))
    print(len(yTrain))

    return (xTrain, yTrain), (xTest, yTest)
