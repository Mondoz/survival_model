# -*- coding: utf-8 -*-
import mxnet as mx
import argparse
import os
from new_train_net import new_train_net_resp


def parse_args():
    parser = argparse.ArgumentParser(description='Train a chexnet network')
    parser.add_argument('--train-data-csv', dest='train_data', help='.csv file to use',
                        default='./train_data.csv', type=str)

    parser.add_argument('--test-csv', dest='test_csv', help='.csv file to use',
                        default='./test_data.csv', type=str)
    parser.add_argument('--image-path', dest='image_path', help='image path to load images',
                        default='/ps2/cv4/hucheng/data/Mhd_data/Bounding_Box_Nor_Abnor_lung/', type=str)
    parser.add_argument('--model-path', dest='model_path', type=str,
                        default=os.path.join(os.getcwd(), 'model'),
                        help='trained model path')    
    parser.add_argument('--network', dest='network', type=str, default='resnet18_v1',   #densenet121
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=30,
                        help='training batch size')
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--epochs', dest='epochs', help='number of epochs of training',
                        default=500, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=[55, 55, 55],
                        help='set image shape')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.95,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0009,
                        help='weight decay')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=list, default=[10,30,60,90],
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=float, default=0.02,
                        help='ratio to refactor learning rate')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    print (ctx)
    ctx = [mx.cpu()] if not ctx else ctx

    # start training

    new_train_net_resp(args.network, args.train_data, args.test_csv,args.image_path,args.model_path, args.batch_size,
                   args.data_shape, ctx, args.epochs,
                   args.learning_rate, args.momentum, args.weight_decay,
                   args.lr_refactor_step, args.lr_refactor_ratio)