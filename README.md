# Live Emotion Classifier
* For this repository, this is for the live emotion classifer using the FER_2013 dataset. The detailed procedure is firstly, determine if there is any face in the camera. After that, we use the pre-trained model to detect the face's landmark for the detected face. Finally, we feed that gray-scale face into the live-camera to make the final_predictions.
# Dataset: FER_2013
* You could use the command python live_cam.py to run the live_camera emotion detection. By default, it uses the miniInception model's weights. You could change the model by selecting the model weights in the repository.
* You can git clone this repo and run this model through the command line with the command:<br />
python main.py --model [Resnet,mini_Inception,Big_Inception,base_model] <br>
* To make your prediction using the trained model, use python main.py --evaluate --data [YOUR_DATA_PATH] --model_weights [YOUR_MODEL_WEIGHTS_PATH]
You could try to connect the folked Github repo to the Colab to make fully use of the Colab GPU with your personal Token specifcied.
#Limitations:
* Since we are pramarily use the FER-2013 dataset, this emotion classification has more bias towards the some races while Asian people's race's face emotion is slightly degraded. In the furture, we need to add more training emotion data for different races.
# Model Performance
* Base_line: Accuracy:0.5 <br>
* mini_Inception: 0.59 <br>
* Big_Inception: 0.55, more epochs needed <br>

# Graphical User Interface

You could use the command python live_cam_UI.py to run the program with GUI. Before running the program, you need to install three library 'PyQt5', 'pyshine' and 'imutils' by following commands:

```text
pip3 install PyQt5
pip3 install pyshine==0.0.6
pip3 install imutils
```

To play with the live_cam_UI, press 'start' button to activate the live camera. 

You can make a screenshot by pressing 'Take picture' button.

To exit the program, please press 'stop' button and then press 'x' button at the top left.

# Status:   

* Need to make further improvement about the emotion classifier's accuracy
* Need to find and add more training data of different people.
* Need to replace the emotion with cute emojis :)

    

