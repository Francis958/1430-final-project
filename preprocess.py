import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hp
import matplotlib.pyplot as plt



def preprocess(train_dir,test_dir,BATCH_SIZE):
    SEED = 12
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

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
                                                seed = SEED)

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







# img1 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/angry/im0.png')
# img2 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/disgusted/im0.png')
# img3 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/fearful/im0.png')
# img4 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/happy/im0.png')
# img5 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/neutral/im0.png')
# img6 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/sad/im0.png')
# img7 = plt.imread('/Users/jiataoyuan/Desktop/final project/archive/train/surprised/im0.png')

# plt.imshow(img1)
# plt.imshow(img2)
# plt.imshow(img3)
# plt.imshow(img4)
# plt.imshow(img5)
# plt.imshow(img6)
# plt.imshow(img7)
# plt.show()







