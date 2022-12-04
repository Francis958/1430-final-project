from train import train
import hp
import model
import os
import sys
import argparse
import re

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["Resnet", "base_model","simple_CNN"],
        help="""model selection""",
    )
    return parser.parse_args()


if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'  
    ARGS = parse_args()
    if ARGS.model == 'Resnet':
        my_model = model.ResNet34()
        train(train_dir,test_dir,my_model,ARGS.model)
    elif ARGS.model == 'base_model':
        my_model = model.basic_model()
        train(train_dir,test_dir,my_model,ARGS.model)
    elif ARGS.model == 'simple_CNN':
        my_model = model.simple_CNN(hp.shape,hp.num_class)
        train(train_dir,test_dir,my_model,ARGS.model)


    

