from train import train
import hp
import model
import os
import sys
import argparse
import re
from prediction import prediction
def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Emotion Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        choices=["Resnet","base_model","mini_Inception","big_Inception"],
        help="""model selection""",
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    parser.add_argument(
        '--data',
        default=None,
        help='''data''')
    
    parser.add_argument(
        '--model_weights',
        default=None,
        help='''model_weights''')

    return parser.parse_args()


if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'  
    ARGS = parse_args()

    if ARGS.data is not None:
        test_data = ARGS.data
    if ARGS.model_weights is not None:
        model_weights = ARGS.model_weights
    
    if ARGS.evaluate:
        print(prediction(test_data,model_weights))
    if ARGS.model == 'Resnet':
        my_model = model.ResNet34()
        train(train_dir,test_dir,my_model,ARGS.model)
    elif ARGS.model == 'base_model':
        my_model = model.simple_CNN(hp.shape,hp.num_class)
        train(train_dir,test_dir,my_model,ARGS.model)
    elif ARGS.model == 'mini_Inception':
        my_model = model.mini_Inception(hp.shape,hp.num_class)
        train(train_dir,test_dir,my_model,ARGS.model)
    elif ARGS.model == 'big_Inception':
        my_model = model.big_Inception(hp.shape,hp.num_class)
        train(train_dir,test_dir,my_model,ARGS.model)
    else:
        pass
    



    

