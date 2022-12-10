import dlib
import model 
import hp
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)
