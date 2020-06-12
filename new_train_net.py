# -*- coding: utf-8 -*-
import sys
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
from loss import Survial_loss
import numpy as np
import pandas as pd
import os
from dataiter import TrainLoader, TestLoader
import SimpleITK as sitk
import time
from lifelines.utils import concordance_index
from resnet import resnet18_v1

survial_loss = Survial_loss()

def get_optimizer_params(optimizer=None, learning_rate=None, momentum=None,weight_decay=None, ctx=None):
    if optimizer.lower() == 'rmsprop':
        opt = 'rmsprop'
        print('you chose RMSProp, decreasing lr by a factor of 10')
        optimizer_params = {'learning_rate': learning_rate / 10.0,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'sgd':
        opt = 'sgd'
        optimizer_params = {'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'adadelta':
        opt = 'adadelta'
        optimizer_params = {}
    elif optimizer.lower() == 'adam':
        opt = 'adam'
        optimizer_params = {'learning_rate': learning_rate,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    return opt, optimizer_params

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

def prepare_data(x, label):
 
    e, t = label[:, 0], label[:, 1]

    # Sort Training Data for Accurate Likelihood
    # sort array using pandas.DataFrame(According to DESC 't' and ASC 'e')  
    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    label[:, 0] = e[sort_idx]
    label[:, 1] = t[sort_idx]

    return x, label


def prepare_data(x, label):
 
    e, t = label[:, 0], label[:, 1]

    # Sort Training Data for Accurate Likelihood
    # sort array using pandas.DataFrame(According to DESC 't' and ASC 'e')  
    df1 = pd.DataFrame({'t': t, 'e': e})
    df1.sort_values(['t', 'e'], ascending=[False, True], inplace=True)
    sort_idx = list(df1.index)
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties


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
    hr_pred = -y_pred
    ci = concordance_index(label_true[:, 1], hr_pred, label_true[:, 0])
    return ci

def get_Loss(net, flag):
    count =0.
    layers = net.features
    for layer in layers:
        params = layer.collect_params().values()
        for i in params:
            if 'weight' in str(i):
                Weight = i.data()
                if flag==1:
                    if  count == 0.:
                        Regularization_loss = nd.sum(nd.abs(Weight))
                    else:
                        Regularization_loss = nd.add(Regularization_loss, nd.sum(nd.abs(Weight)))
                elif flag==2:
                    if  count == 0.:
                        Regularization_loss = nd.sum(Weight ** 2)
                    else:
                        Regularization_loss = nd.add(Regularization_loss, nd.sum(Weight ** 2))
                count += 1
    return Regularization_loss
'''
def cumsum(data):
    for i in len(data):
'''

def read_all_data(data, data_shape):
    Len = len(data)
    X = np.zeros((Len,1,data_shape[0],data_shape[1],data_shape[2]),dtype=np.float32)
    Y = np.zeros((Len,2), dtype=np.float32)
    for i in range(Len):
        value = data[i].strip()
        im_id = value.split(',')[0]
        img = sitk.ReadImage(im_id+ '.mhd', sitk.sitkFloat32)

        # resize
        Image_array = resize_3D(img, data_shape)
        Image_array = (Image_array + 1000)/1400
        X[i] = Image_array.reshape(1,1,data_shape[0],data_shape[1],data_shape[2])

        # 1-->event, 3-->time
        Y[i,0] = value.split(',')[1]
        Y[i,1] = value.split(',')[2]
    return X, Y

# 测试评估
def evaluate_resp(net, test_csv, ctx, data_shape, image_path):
    loss, acc= 0., 0.
    Pre, Rec,count_pre,count_rec=0., 0., 0., 0.
    AUC = np.zeros((1,))
    num = 0
    test_id=[]
    label_list=[]
    output_list=[]
    output_score=[]
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
    #test_loss = survial_loss(nd.array(output_hr), nd.array(Y_test[:,0]))
    CI = metrics_ci(Y_test, output_hr)
    return  CI



def new_train_net_resp(network, train_data, test_csv,image_path, model_path, batch_size,
                   data_shape, ctx, epochs, learning_rate,
                   momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,optimizer='sgd'):

    train = [x for x in open(train_data).readlines()]
    test  = [line for line in open(test_csv).readlines()]
    
    #
    print('total num of training set:', len(train))
    print('total num of valid set:',len(test))

    train_loader = TrainLoader(train, batch_size = batch_size)
    log_txt = open('log_new.txt', 'w')
    X, Y = read_all_data(train, data_shape)
    all_train = dict()
    all_train['X'], all_train['E'], all_train['T'], all_train['failures'], \
    all_train['atrisk'], all_train['ties'] = parse_data(X, Y)

    #net =  getattr(models, network)(classes=1)      #获取对象属性值，即获取network值
    net = resnet18_v1()
    # init
    with net.name_scope(): 
        net.output = nn.Dense(1)
    net.initialize(init.Xavier(rnd_type='uniform', factor_type="in", magnitude=0.8))

    net.collect_params().reset_ctx(ctx)
    net.hybridize() 

    loss = Survial_loss()
    best_auc, best_acc = 0., 0.

    # optimizer
    opt, opt_params = get_optimizer_params(optimizer=optimizer, learning_rate=learning_rate, momentum=momentum,
                                           weight_decay=weight_decay, ctx=ctx)
    print('Running on', ctx)

    trainer = gluon.Trainer(net.collect_params(), opt,opt_params)
 
    for epoch in range(epochs):
        train_loss = 0.
        train_total_ci = 0.   #add the each batch of acc_list
        steps = 0
        #reset lr
        trainer.set_learning_rate(learning_rate/(1 + epoch * lr_refactor_ratio))
        '''
        if len(lr_refactor_step) > 0 :
            if epoch == lr_refactor_step[0]:
                trainer.set_learning_rate(trainer.learning_rate*lr_refactor_ratio)
                del lr_refactor_step[0]
        '''
        #load train-data
        for t, batch in enumerate(train_loader):
            steps +=1
            n = len(batch)
            X = np.zeros((n,1,data_shape[0],data_shape[1],data_shape[2]),dtype=np.float32)
            Y = np.zeros((n,2), dtype=np.float32)
            for i,value in enumerate(batch):
                value = value.strip()
                im_id = value.split(',')[0]
                img = sitk.ReadImage(im_id+ '.mhd', sitk.sitkFloat32)   #X  Y  Z
                # resize
                Image_array = resize_3D(img, data_shape)
                Image_array = (Image_array + 1000)/1400
                X[i] = Image_array.reshape(1,1,data_shape[0],data_shape[1],data_shape[2])      # 5D-->NCDHW
                
                # 1-->event, 3-->time
                Y[i,0] = value.split(',')[1]
                Y[i,1] = value.split(',')[2]


            train_data = dict()
            train_data['X'], train_data['E'], train_data['T'], train_data['failures'], \
            train_data['atrisk'], train_data['ties'] = parse_data(X, Y)
            data = mx.nd.array(train_data['X'])
            label = mx.nd.array(train_data['E'])
            #train_data['ties'] = all_train['ties']
            train_data['ties'] = 'noties'
            data_context = data.as_in_context(ctx[0])      ##  ???
            lable_context = label.as_in_context(ctx[0])

            # autograd

            with autograd.record():
                output = net(data_context)
                L1_loss = get_Loss(net, 1)
                L2_loss = get_Loss(net, 2)

                losses = loss(lable_context, output) + 2e-5 * L1_loss + 3e-3 * L2_loss
                #losses = [loss(net(x), y) for x,y in zip(data_list,label_list)]
            #backward

            losses.backward()
            #output = net(data_context) 
            

            # compute ci            
            ci = metrics_ci(Y, output.asnumpy())
            
            #loss
            lmean = losses.mean().asscalar()
            
            if steps%10 == 0:
                print(output)
                print ("Batch %d. train_loss: %.4f, ci: %.2f" % (steps, lmean, ci))
            train_loss += lmean
            #train_total_ci += ci
            trainer.step(1)
        print('total num of batches:', steps)
        
        #ci = metrics_ci(final_label, final_output)
        #print ('epoch: %d, train_loss: %.4f, epoch_ci: %.2f', (epoch, train_loss / steps, train_total_ci/steps))

        train_ci = evaluate_resp(net, train, ctx[0], data_shape, image_path)

        # evaluate
        val_ci = evaluate_resp(net, test, ctx[0], data_shape, image_path)
        

        print("Epoch %d, epoch_train_loss: %.4f; epoch_train_CI: %.4f, val_CI %.2f" % (
                        epoch, train_loss / steps, train_ci, val_ci))
        log_txt.write('Epoch:' + ',' +str(epoch) + ',' + 'train_loss:' + str(train_loss / steps) \
                               + ',' + 'train_ci:' + str(train_ci) + ',' + 'val_ci:' + str(val_ci) +'\n')
        net.save_params('./model/%s_Epoch%d.params'%('Survial', epoch))
        train_loader.reset()
