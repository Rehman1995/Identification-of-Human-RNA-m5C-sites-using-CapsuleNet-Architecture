#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 16:101:59 2021

@author: mobeen
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D,LayerNormalization, Dropout, Flatten, Dense, concatenate 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import binary_accuracy
from keras import regularizers
import matplotlib.pyplot as plt
#from group_norm import GroupNormalization
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, roc_auc_score, auc
import math
import tensorflow as tf
from tensorflow.keras import losses
import random
# import scikitplot as skplt
import seaborn as sn
import itertools
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask,CapsuleLayer_nogradient_stop
# np.random.seed(seed=21)
# ************ Data Processing ****************

def encode_seq(s):
    Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'U':[0,0,0,1],'T':[0,0,0,1],'N':[0,0,0,0]}
    return np.array([Encode[x] for x in s])

def EIIP(s):
    Encode = {'A':[0.1260],'C':[0.1340],'G':[0.0806],'T':[0.1335],'U':[0.1335],'N':[0]}
    return np.array([Encode[x] for x in s])
def cal(c,cb,i):
    bases ={'A':[1,1,1], 'C':[0,1,0], 'G':[1,0,0,], 'T':[0,0,1],'U':[0,0,1]}
    p=[]
    p=bases[c]
    p.append(np.round(cb/float(i+1),2))
    return(p)
def calculate(s):
    p=f=list()
    cba=cbc=cbt=cbg=0
    for i,c in enumerate(s):
        if c=='A':
            cba+=1
            p=cal(c,cba,i)
        elif c=='T':
            cbt+=1
            p=cal(c,cbt,i)
        elif c=='C':
            cbc+=1
            p=cal(c,cbc,i)
        elif c=='G':
            cbg+=1
            p=cal(c,cbg,i)
        else:
            p=[0,0,0,0]
        f.append(p)
    return(f)



def listToString(s):  

    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1   


def dataProcessing(path):
    file = open(path,"r")

    alphabet = np.array(['A', 'G', 'T', 'C','N'])
    X = [];
    l1 = len(open(path).readlines(  ))

    count=0
    Training=[0]*l1
    for line in file:
      
      #Let's split the line into an array called "fields" using the ";" as a separator:
      Data = line.split(':')
      Training[count] = Data
      count=count+1
    
    elements1 = {}
    
    accumulator=0
    for row in Training:
      #print(row)
      row=listToString(row)
      row=row.strip('\n')
      my_hottie = encode_seq((row))
      out_final=my_hottie
      out_final = np.array(out_final)
      elements1[accumulator]=out_final
      accumulator += 1 
   
    X = list(elements1.items())
    an_array = np.array(X)
    an_array=an_array[:,1]
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X=np.transpose(transpose_list)
    X=np.transpose(X)
    #y = np.array(data['label'], dtype = np.int32);
 
    return X; 


def prepareData(PositiveCSV, NegativeCSV):

    Positive_X = dataProcessing(PositiveCSV);
    Negative_X = dataProcessing(NegativeCSV);
    len_data=len(Positive_X)
    len_data2=len(Negative_X)

    Positive_y=np.ones(int(len_data));
    Negative_y=np.zeros(int(len_data2));

    return Positive_X, Positive_y, Negative_X, Negative_y

def funciton(PositiveCSV, NegativeCSV, OutputDir, folds):

    Positive_X, Positive_y, Negative_X, Negative_y = prepareData(PositiveCSV, NegativeCSV)





def calculateScore(X, y, model, folds):
    predictions = model.predict([X])
    decision = []
    predictions=predictions[0]

    for i in range(len(predictions)):
        if predictions[i,0] > predictions[i,1]:
            decision.append(predictions[i,0])
        else:
            decision.append(predictions[i,1])
    pred_y = decision 

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(decision)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    # print(tempLabel)        
    
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()
    # print(confusion)
    


    confusion_norm = confusion / confusion.astype(np.float).sum(axis=1) # Normalize confusion matrix
    sn.heatmap(confusion_norm, annot=True, cmap='Blues')
    # sn.heatmap(confusion, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        

 
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    accuracy = (TP + TN) / float(TP + TN + FP +FN)
    from sklearn.metrics import matthews_corrcoef
    MCC=matthews_corrcoef(y, tempLabel)
    
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    #pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    #print(y.shape)
    #print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    plt.show() 
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#.eval()
    print(sensitivity)
    print(specificity)
    print(accuracy)
    print(MCC)
    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}



def analyze(temp, OutputDir):

    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:


        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title +  'results\n')

        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:

            total = []

            for val in x:
                total.append(val[j])

            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();



    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig( OutputDir + '/' + title +'ROC.png')
        plt.close('all');

        index += 1;
from keras import layers, models, optimizers

def CapsNet(input_shape, n_class, routings):
    #CapsNet((101,4),2,10);
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=512, kernel_size=9, strides=1, padding='valid',kernel_initializer='random_uniform', activation='elu', name='conv1')(x)
    conv10 = Dropout(0.1)(conv1)
    conv2 = layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='valid',kernel_initializer='random_uniform', activation='elu', name='conv2')(conv10)
    conv2 = Dropout(0.1)(conv2)
    conv2 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',kernel_initializer='random_uniform', activation='elu')(conv2)
    conv2 = Dropout(0.1)(conv2)
    conv2 = tf.keras.layers.GRU(units=128,   name='Lstm',  return_sequences=True)(conv2)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=50, kernel_size=10, strides=2, padding='valid', dropout=0.2)
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,name='digitcaps', kernel_initializer='random_uniform')(primarycaps)    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps) 
    conv_flatten=layers.Flatten()(conv10)

    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(256, activation='elu', input_dim=10*n_class))
    decoder.add(layers.Dropout(0.5))
    dec_1 = decoder.add(layers.Dense(128, activation='elu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape))

    # Models for training and evaluation (prediction)
    model = models.Model([x],[out_caps, decoder(masked)]) #masked_by_y

    # manipulate model
    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked),conv_flatten])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

from keras import backend as K



def train(model, X_train, y_train, X_test, y_test):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # compile the model

#    model.compile(optimizer=optimizers.Adam(lr=0.003, epsilon=1e-08), loss=['binary_crossentropy'],
#                  loss_weights=[1., 0.392], metrics={'capsnet': 'accuracy'})

    model.compile(optimizer=optimizers.Adam(lr=0.001, epsilon=1e-08), loss=['binary_crossentropy'],
                  metrics={'capsnet': 'accuracy'})


    #model.compile(optimizers.RMSprop(lr=0.0001),loss=margin_loss,metrics=['acc'])

    
#    path = r'C:\Users\Karush\.spyder-py3\CapsNet_Weights5'
#    for j in range(1,11):
#        model.load_weights(path+r"\weights-improvement-"+str(j)+".hdf5")
#        filters = model.layers[8].get_weights()[0]
#        np.shape(filters)
#        filters = np.reshape(filters, (int(200*4),int(9324/4)))
#        plt.figure()
#        plt.imshow(filters)
#        plt.savefig(r'C:\Users\Karush\.spyder-py3\CapsNet_Activations5\c'+str(j)+'.png')

    #model.load_weights('new_capsnet20.h5')
    # Training without data augmentation:
    callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
         filepath='model.h5', 
         save_freq='epoch', verbose=1, monitor='capsnet', 
         save_weights_only=True, save_best_only=True)]
    
    hist = model.fit([X_train, y_train], [y_train, X_train], batch_size=2048, epochs=500,
               shuffle=True,callbacks=callback_list)    
    #model.save_weights('new_capsnet30.h5')
    ## Saving model to JSON
#    model_json = model.to_json()
#    with open("5capsnet_50.json", "w") as json_file:
#        json_file.write(model_json)
#    
#    ## Saving model data to JSON
#    with open('5capsnet_50.json', 'w') as f:
#        json.dump(hist.history, f)
#    
#    model.save_weights('5capsnet_50.h5')

    return model,hist


PositiveCSV = 'Positive.txt'
NegativeCSV = 'Negative.txt'
Positive_X,Positive_y, Negative_X,Negative_y = prepareData(PositiveCSV, NegativeCSV)
input_data=np.concatenate([Positive_X,Negative_X])
#input_data2=np.concatenate([NCP_D_P,NCP_D_N])
#input_data3=np.concatenate([EIIP_P,EIIP_N])

input_labels=np.concatenate([Positive_y,Negative_y])
s = np.arange(input_data.shape[0])
import random

random.Random(4).shuffle(s)
data_io = input_data[s]

labels= input_labels[s]


folds=5 
kf = KFold(n_splits=folds, shuffle=True, random_state=100)
for train_index, test_index in kf.split(data_io):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test0 = data_io[train_index], data_io[test_index]


    y_train, y_test = labels[train_index], labels[test_index]

    X_train0, X_validation0, y_train0, y_validation0 = train_test_split(X_train, y_train, test_size=0.1, random_state=10, shuffle=False)




trainning_result = []
validation_result = []
testing_result = []

for test_index in range(folds):    
    i=(41,4)
    model, eval_model, manipulate_model = CapsNet(i,2,5);
    #history = model.fit([X_train0], y_train0, batch_size = 64, epochs=20, validation_data = ([X_validation0], y_validation0));

    model,history=train(model, X_train0, y_train0, X_validation0, y_validation0)
    model.save_weights('/home/mobeen/Capsule_net/Human_Conference/model'+str(test_index+1)+'.h5')

    #**************** Plot graphs **************
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()

   

    
    trainning_result.append(calculateScore(X_train0, y_train0, eval_model, folds))
    validation_result.append(calculateScore(X_validation0, y_validation0, eval_model, folds));
    testing_result.append(calculateScore(X_test0, y_test, eval_model, folds));
        
temp_dict = (trainning_result, validation_result, testing_result)
OutputDir = '/home/mobeen/Capsule_net/Human_Conference/'
analyze(temp_dict,OutputDir)

model.load_weights('model5.h5')
#model, history = train(model, data_io, labels, data_io, labels)

PositiveCSV = 'indPositive.txt'
NegativeCSV = 'indNegative.txt'
Positive_X, Positive_y, Negative_X, Negative_y = prepareData(
    PositiveCSV, NegativeCSV)
input_data = np.concatenate([Positive_X, Negative_X])
#input_data2=np.concatenate([NCP_D_P,NCP_D_N])
#input_data3=np.concatenate([EIIP_P,EIIP_N])

input_labels = np.concatenate([Positive_y, Negative_y])
ind_result = []
ind_result.append(calculateScore(input_data, input_labels, eval_model, folds))
