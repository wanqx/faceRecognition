from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from loadBasicEnv import NAMEDICT, NET_SAVE_NAME, IMGSHAPE
#  from loadData import load_data
from vgg import ConvNet, accuracy
from findFace import transFace


#  (x_train, y_train), (x_test, y_test) = load_data()
#  x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
#  x_train, x_test = x_train / 255., x_test / 255.

conv_net = ConvNet()
conv_net.load_weights(f"./modelSave/{NET_SAVE_NAME}")

#  # Test model on validation set.
#  pred = conv_net(x_test[:100])
#  print("Test Accuracy: %f" % accuracy(pred, y_test[:100]))


# Visualize predictions.
import matplotlib.pyplot as plt
import os


for name in [x for x in os.listdir() if x.endswith('.jpg')]:
    img = cv2.imread(name, 0)
    img0 = img.copy()
    img = transFace(img)
    img = cv2.resize(img, (IMGSHAPE, IMGSHAPE))
    img = np.array(img, np.float32)
    img = img / 255.
    pred = conv_net(img)
    #  pred = NAMEDICT[np.argmax(pred.numpy())]
    #  print(pred)
    topK, index = map(lambda x: x.numpy(), tf.nn.top_k(pred, k=3))
    res = [(pred[0][x].numpy(), NAMEDICT[x]) for x in index[0]]
    for r in res:
        probability, name = r
        print(f"{name}: {probability:.06f}")

    plt.imshow(img, cmap='gray')
    plt.show()
# Predict 5 images from validation set.
#  n_images = 10
#  test_images = x_test[:n_images]
#  predictions = conv_net(test_images)
#
#  # Display image and model prediction.
#  for i in range(n_images):
    #  plt.imshow(np.reshape(test_images[i], [50, 50]), cmap='gray')
    #  plt.show()
    #  rawPredict = np.argmax(predictions.numpy()[i])
    #  rawReal = y_test[i]
    #  print(f"Model prediction: {NAMEDICT[rawPredict]}; real: {NAMEDICT[rawReal]}")
