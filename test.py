# -*- coding: utf-8 -*-
import sys
sys.path.append("/ps2/cv4/hucheng/code/Lung_noraml_abnormal/Bounding_box/Nor_Abnor/multi_batch_one_update/test_demo/mxnet-medical-20180302-dev/python")
import mxnet as mx
import importlib
import re
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import time
import argparse
import math
from dataiter import TrainLoader, TestLoader
from lifelines.utils import concordance_index
from resnet import resnet18_v1

def parse_args():
    parser = argparse.ArgumentParser(description='test for own dataset')
    parser.add_argument('--network', dest='network', type=str, default='resnet34_v1',
                        help='which network to use')
    parser.add_argument('--images-path', dest='image_path', type=str, default='/ps2/cv4/hucheng/data/Mhd_data/Nor_Abnor/',
                        help='image_path')
    parser.add_argument('--model-path', dest='model_path', type=str,
                        default=os.path.join(os.getcwd(), 'model_new'),
                        help='trained model path')
    parser.add_argument('--test-csv', dest = 'test_csv', type = str, 
                        default = './test_data.csv'  )
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=[55, 55, 55],
                        help='set image shape')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.5')
    parser.add_argument('--class-name', dest='class_name', type=str, default='abnormal',
                        help='class name for classification')
    args = parser.parse_args()
    return args


def resize_3D(img_ori,data_shape):
    Img_temp = sitk.GetArrayFromImage(img_ori)
    
    Z_dim,Y_dim,X_dim = Img_temp.shape
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(img_ori.GetDirection())
    resample.SetOutputOrigin(img_ori.GetOrigin())
    Origin_spacing = img_ori.GetSpacing()
    newspacing = [Origin_spacing[0]*X_dim/data_shape[2],
                  Origin_spacing[1]*Y_dim/data_shape[1],
                  Origin_spacing[2]*Z_dim/data_shape[0]]
    
    resample.SetSize([data_shape[2],data_shape[1],data_shape[0]])
    resample.SetOutputSpacing(newspacing)
    newimage = resample.Execute(img_ori)
    Image_array = sitk.GetArrayFromImage(newimage)
    return Image_array

def metrics_ci(label_true, y_pred):
    """Compute the concordance-index value.

    Parameters
    ----------
    label_true : dict
        Status and Time of patients in survival analyze,
        example like as {'e': event, 't': time}.
    y_pred : np.array
        Proportional risk.

    Returns
    -------
    float
        Concordance index.
    """
    hr_pred = - y_pred
    ci = concordance_index(label_true[:, 1], hr_pred, label_true[:, 0])
    return ci


def evaluate_resp(net, test_csv, ctx, data_shape, image_path):
    loss, acc= 0., 0.
    Pre, Rec,count_pre,count_rec=0., 0., 0., 0.
    AUC = np.zeros((1,))
    result = open('record.csv','w')
    num = 0
    test_id=[]
    label_list=[]
    output_list=[]
    output_score=[]
    test_csv  = [line for line in open(test_csv).readlines()]
    Len=len(test_csv)
    output_hr = np.zeros((Len, 1), dtype = np.float32)
    Y_test = np.zeros((Len, 2), dtype = np.float32)
    for i in range(Len):
        num += 1
        X_test = np.zeros((1,1,data_shape[0],data_shape[1],data_shape[2]),dtype=np.float32)
        
        value=test_csv[i].strip()
        im_id=value.split(',')[0]
        test_id.append(im_id)
        
        # read mhd
        img = sitk.ReadImage((im_id+'.mhd'),sitk.sitkFloat32)   #X  Y  Z
        Image_array = resize_3D(img, data_shape)
        Image_array = (Image_array + 1000)/1400
        X_test = Image_array.reshape(1,1,data_shape[0],data_shape[1],data_shape[2])      # 5D-->NCDHW
                
        # 1-->event, 3-->time
        Y_test[i,0] = value.split(',')[1]
        Y_test[i,1] = value.split(',')[2]
        
       
        data=mx.nd.array(X_test)
        label = mx.nd.array([Y_test])

        data_context, lable_context = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data_context)
        output_hr[i, 0] = output.asnumpy()
        result.write(im_id + ',' + str(output.asnumpy()) + ',' + value.split(',')[1] +',' +value.split(',')[2] +'\n')
    #test_loss = survial_loss(nd.array(output_hr), nd.array(Y_test[:,0]))
    CI = metrics_ci(Y_test, output_hr)
    print(CI)


if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)
    net = resnet18_v1()
    # init
    with net.name_scope(): 
        net.output = nn.Dense(1)
    model_path = '/ps2/cv4/hucheng/Lung_hospital/train_use_mhd/new_train_crop/Ori_loss_resize/model'
    params_features = os.path.join(model_path,'Survial_Epoch109.params')
    net.load_params(params_features, ctx=ctx)
    evaluate_resp(net, args.test_csv, ctx,  args.data_shape, 
    			args.image_path)