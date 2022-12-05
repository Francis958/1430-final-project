import tensorflow as tf
import model
import hp
import cv2
import numpy as np
from keras.models import load_model

def prediction(img_path,model_weights_path):
    model_ = model.simple_CNN(hp.shape,hp.num_class)
    model_.load_weights(model_weights_path)
    mapping = {0:'angry',1:'disgusted',2:'fearful',3:'happy',4:'neutral',5:'sad',6:'surprised'}
    img1 = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img1= cv2.resize(img1,(48,48))
    img1 = np.expand_dims(img1,axis =2)
    img1 = img1/255
    img1= np.reshape(img1,(1,48,48,1))
    prob = (model_.predict(img1))
    pred = (np.argmax(prob,axis = 1))
    return mapping[pred[0]]

# print(prediction(r'D:\Desktop\surprise.jpg',r'D:\Desktop\1430-final-project\base_model\2022-12-05_01-18-26_AM\checkpoint\val_acc-0.554-val_loss-1.1659epoch-035.h5'))