
# Created by: PyQt5 UI code generator 5.11.3


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import time
import numpy as np


import numpy as np
import cv2
import pyshine as ps

import dlib
import model 
import hp
import tensorflow as tf
from tensorflow.keras.models import load_model


try:
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    print('Warning...',e)

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

model_ = model.mini_Inception(hp.shape,hp.num_class)
model_.load_weights(r'mini_Inception/2022-12-05_01-51-05_AM/checkpoint/val_acc-0.590-val_loss-1.1005epoch-022.h5')
emotionTargetSize = model_.input_shape[1:3]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("images/H.png"))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.gridLayout.addWidget(self.verticalSlider, 0, 0, 1, 1)
        # self.verticalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        # self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        # self.verticalSlider_2.setObjectName("verticalSlider_2")
        # self.gridLayout.addWidget(self.verticalSlider_2, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(313, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.verticalSlider.valueChanged['int'].connect(self.brightness_value)
        # self.verticalSlider_2.valueChanged['int'].connect(self.blur_value)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton.clicked.connect(self.savePhoto)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # Added code here
        self.filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png' # Will hold the image address location
        self.tmp = None # Will hold the temporary image for display
        self.brightness_value_now = 0 # Updated brightness value
        self.blur_value_now = 0 # Updated blur value
        self.fps=0
        self.started = False

    def loadImage(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        if self.started:
            self.started=False
            self.pushButton_2.setText('Start')	
        else:
            self.started=True
            self.pushButton_2.setText('Stop')
        
        cam = True # True for webcam
        vid = cv2.VideoCapture(1)
        # if cam:
        #     vid = cv2.VideoCapture(1)
        # else:
        #     vid = cv2.VideoCapture('video.mp4')
        
        cnt=0
        frames_to_count=20
        st = 0
        fps=0

        
        while(vid.isOpened()):
            QtWidgets.QApplication.processEvents()	
            # img, self.image = vid.read()

            ret, self.image = vid.read()
            if not ret:
                break
            self.image = cv2.resize(self.image, (1080, 720))

            grayFrame = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            rects = detector(grayFrame, 0)
            for rect in rects:
                shape = predictor(grayFrame, rect)
                points = shapePoints(shape)
                (x, y, w, h) = rectPoints(rect)
                grayFace = grayFrame[y:y + h+10, x:x + w]
                try:
                    grayFace = cv2.resize(grayFace, (emotionTargetSize))
                except:
                    continue

                grayFace = grayFace.astype('float32')
                grayFace = grayFace / 255.0
                grayFace = np.expand_dims(grayFace, 0)
                grayFace = np.expand_dims(grayFace, -1)
                emotion_prediction = model_.predict(grayFace)
                emotion_probability = np.max(emotion_prediction)
                if (emotion_probability > 0.36):
                    emotion_label_arg = np.argmax(emotion_prediction)
                    color = emotions[emotion_label_arg]['color']
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                    cv2.line(self.image, (x, y + h), (x + 20, y + h + 20),
                             color,
                             thickness=2)
                    cv2.rectangle(self.image, (x + 20, y + h + 20), (x + 110, y + h + 40),
                                  color, -1)
                    cv2.putText(self.image, emotions[emotion_label_arg]['emotion'],
                                (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    color = (255, 255, 255)
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)

            
            if cnt == frames_to_count:
                try: # To avoid divide by 0 we put it in try except
                    print(frames_to_count/(time.time()-st),'FPS') 
                    self.fps = round(frames_to_count/(time.time()-st)) 
                    
                    
                    st = time.time()
                    cnt=0
                except:
                    pass
            
            cnt+=1
            
            self.update()
            key = cv2.waitKey(1) & 0xFF
            if self.started==False:
                break
                print('Loop break')

    def setPhoto(self,image):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image,width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def brightness_value(self,value):
        """ This function will take value from the slider
            for the brightness from 0 to 99
        """
        self.brightness_value_now = value
        print('Brightness: ',value)
        self.update()
        
        
    def blur_value(self,value):
        """ This function will take value from the slider 
            for the blur from 0 to 99 """
        self.blur_value_now = value
        print('Blur: ',value)
        self.update()


    def changeBrightness(self,img,value):
        """ This function will take an image (img) and the brightness
            value. It will perform the brightness change using OpenCv
            and after split, will merge the img and return it.
        """
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        lim = 255 - value
        v[v>lim] = 255
        v[v<=lim] += value
        final_hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        return img
        
    def changeBlur(self,img,value):
        """ This function will take the img image and blur values as inputs.
            After perform blur operation using opencv function, it returns 
            the image img.
        """
        kernel_size = (value+1,value+1) # +1 is to avoid 0
        img = cv2.blur(img,kernel_size)
        return img

    def update(self):
        """ This function will update the photo according to the 
            current values of blur and brightness and set it to photo label.
        """
        img = self.changeBrightness(self.image,self.brightness_value_now)
        img = self.changeBlur(img,self.blur_value_now)

        # Here we add display text to the image
        text  =  'FPS: '+str(self.fps)
        img = ps.putBText(img,text,text_offset_x=20,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
        text = str(time.strftime("%H:%M %p"))
        img = ps.putBText(img,text,text_offset_x=self.image.shape[1]-180,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(228,20,222),text_RGB=(255,255,255))
        text  =  f"Brightness: {self.brightness_value_now}"
        img = ps.putBText(img,text,text_offset_x=80,text_offset_y=425,vspace=20,hspace=10, font_scale=1.0,background_RGB=(20,210,4),text_RGB=(255,255,255))
        # text  =  f'Blur: {self.blur_value_now}: '
        # img = ps.putBText(img,text,text_offset_x=self.image.shape[1]-200,text_offset_y=425,vspace=20,hspace=10, font_scale=1.0,background_RGB=(210,20,4),text_RGB=(255,255,255))


        self.setPhoto(img)

    def savePhoto(self):
        """ This function will save the image"""
        self.filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png'
        cv2.imwrite(self.filename,self.tmp)
        print('Image saved as:',self.filename)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Emotion Recognition"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Brightness"))
        # self.label_3.setText(_translate("MainWindow", "Blur"))
        self.pushButton.setText(_translate("MainWindow", "Take picture"))

# Subscribe to PyShine Youtube channel for more detail! 

# WEBSITE: www.pyshine.com


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())

