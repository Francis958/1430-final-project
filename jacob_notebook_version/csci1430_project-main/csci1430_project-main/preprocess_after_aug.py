import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def preprocessing_fcn(train_dir, test_dir, more_aug = False, color_mode = 'grayscale'):
    ## Initiative preprocessing
    preprocess_fn = tf.keras.applications.densenet.preprocess_input
    if more_aug:
        train_datagen = ImageDataGenerator(horizontal_flip=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.05, 
                                           validation_split=0.2,
                                           rescale=1./255)

        test_datagen = ImageDataGenerator(horizontal_flip=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.05, 
                                           rescale=1./255)
    else:
        train_datagen = ImageDataGenerator(validation_split=0.2)
        test_datagen = ImageDataGenerator()


    ## Getting training, validation, and testing data
    train_gen = train_datagen.flow_from_directory(directory=train_dir,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  color_mode=color_mode,
                                                  class_mode='categorical',
                                                  subset='training', 
                                                  seed = SEED)

    val_gen = train_datagen.flow_from_directory(directory=train_dir,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  color_mode=color_mode,
                                                  class_mode='categorical',
                                                  subset='validation', 
                                                  seed = SEED)

    test_gen = test_datagen.flow_from_directory(directory=test_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                color_mode=color_mode,
                                                class_mode='categorical',
                                                seed = SEED)
    return train_datagen, test_datagen, train_gen, val_gen, test_gen


def preprocess_after_aug_main(train_dir, test_dir, more_aug = False, color_mode = 'grayscale'):
    ## Grayscale
    if color_mode == 'grayscale':
        train_datagen, test_datagen, train_gen, val_gen, test_gen = preprocessing_fcn(train_dir, test_dir, True, 'grayscale')
    ## rgb scale
    else:
        train_datagen, test_datagen, train_gen, val_gen, test_gen = preprocessing_fcn(train_dir, test_dir, True, 'rgb')

    return train_datagen, test_datagen, train_gen, val_gen, test_gen
