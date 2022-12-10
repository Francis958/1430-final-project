import dlib
import model 
import hp
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2

emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "./shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

model_ = model.simple_CNN(hp.shape,hp.num_class)
model_.load_weights(r'D:\Desktop\1430-final-project\base_model\2022-12-05_01-18-26_AM\checkpoint\val_acc-0.554-val_loss-1.1659epoch-035.h5')
emotionTargetSize = model_.input_shape[1:3]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (720, 480))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = model_.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        if (emotion_probability > 0.36):
            emotion_label_arg = np.argmax(emotion_prediction)
            color = emotions[emotion_label_arg]['color']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                     color,
                     thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                          color, -1)
            cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
