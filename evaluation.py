import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hp
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
import model
from sklearn.metrics import classification_report,accuracy_score

def evaluate(model,test_dir,BATCH_SIZE):
    SEED = 48
    IMG_HEIGHT = 48
    IMG_WIDTH = 48

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_gen = test_datagen.flow_from_directory(directory=test_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='training',
                                                classes={'anger':0,'disgusted':1,'fear':2,'happy':3,'neutral':4,'sadness':5,'surprise':6},
                                                seed = SEED)
    y_pred= np.argmax(model.predict(test_gen),axis = 1)
    y_test = test_gen.labels
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    return y_pred,y_test
