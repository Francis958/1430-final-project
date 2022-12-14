# csci1430_project

cd to your project repo

Our TA suggests us to use the dataset that takes jpg as inputs. The source of the dataset comes from https://www.kaggle.com/datasets/msambare/fer2013. I have uploaded this dataset to my Github repo.

In order to do data augmentation, you can directly use aug_fer2013 after unzipping aug_fer2013.zip. It has already splitted into training set and testing set. Run the data_augmentation.ipynb only if you don't have the aug_fer2013 folder. Each emotion in the trianing set now has 7215 images.

For the preprocessing part, import the preprocessing_after_aug.py or you could copy the code in the preprocessing_after_aug.ipynb into your notebook. It will split the training set into training and vailidation.
