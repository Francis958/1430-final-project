import tensorflow as tf
from tensorflow import keras
import math
import hp
import model
from preprocess import preprocess
import pandas as pd

def train(train_dir,test_dir,my_model,model_name):
   train_gen,val_gen,test_gen = preprocess(train_dir,test_dir,hp.BATCH_SIZE)
   initial_learning_rate = hp.initial_learning_rate
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=hp.decay_rate,
      staircase=True)
   early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=hp.patience,verbose = 2)
   tb_filepath = './{}/Graph'.format(model_name)
   tensor_board = tf.keras.callbacks.TensorBoard(log_dir=tb_filepath, histogram_freq=0, write_graph=True, write_images=True)
   checkpoint_filepath = './{}/checkpoint'.format(model_name)
   model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
   callbacks_list = [early_stop,model_checkpoint_callback,tensor_board]
   my_model.compile(optimizer = tf.keras.optimizers.Adam(lr_schedule),loss='categorical_crossentropy',metrics = 'accuracy')
   history = my_model.fit(x = train_gen, epochs = hp.epochs,validation_data = val_gen,callbacks = callbacks_list)
   print(pd.DataFrame(history.history))




   



