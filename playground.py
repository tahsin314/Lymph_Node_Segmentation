import numpy as np
import cv2
import copy

img = cv2.imread('output.jpg')
target = np.ones(img.shape)

target[img==255] = 0
target[img==0] = 255

cv2.imwrite('target.jpg',target)