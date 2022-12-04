import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hp
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog


def preprocess(train_dir,test_dir,BATCH_SIZE):
    SEED = 42
    IMG_HEIGHT = 48
    IMG_WIDTH = 48

    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.05, 
                                    validation_split=0.2,
                                    rescale=1./255)
    test_datagen = ImageDataGenerator(validation_split=0.2,
                                    rescale=1./255)

    train_gen = train_datagen.flow_from_directory(directory=train_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='training', 
                                                seed = SEED,
                                                )

    val_gen = train_datagen.flow_from_directory(directory=train_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='validation', 
                                                seed = SEED)
    test_gen = test_datagen.flow_from_directory(directory=test_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=1,
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='training', 
                                                seed = SEED)
    return train_gen,val_gen,test_gen







# img1 = cv2.imread('./data/train/angry/im0.png')
# img1 = cv2.resize(img1,(128,64))
# print(img1.shape)
# fd, hog_image = hog(img1 , orientations=9, pixels_per_cell=(8, 8), 
#                     cells_per_block=(2, 2), visualize=True, multichannel=True)
# cv2.imshow('img',hog_image)
# cv2.waitKey(0)
# cv2.imshow('img',img1)
# cv2.waitKey(0)

# img2 = plt.imread('./data/train/fearful/im0.png')
# img3 = plt.imread('./data/train/happy/im0.png')
# img4 = plt.imread('./data/train/neutral/im0.png')
# img5 = plt.imread('./data/train/sad/im0.png')
# img6 = plt.imread('./data/train/surprised/im0.png')
# img7 = plt.imread('./data/train/disgusted/im0.png')
# plt.imshow(img2, cmap="gray")
# plt.show()
# plt.imshow(img3, cmap="gray")
# plt.show()
# plt.imshow(img4, cmap="gray")
# plt.show()
# plt.imshow(img5, cmap="gray")
# plt.show()
# plt.imshow(img6, cmap="gray")
# plt.show()
# plt.imshow(img7, cmap="gray")
# plt.show()







