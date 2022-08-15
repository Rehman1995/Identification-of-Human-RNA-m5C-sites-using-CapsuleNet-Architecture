#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:49:01 2022

@author: mobeen
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";

#import matplotlib as mpl
#mpl.use('Agg')

#from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM, RNN, Dropout, SpatialDropout1D, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,Activation,concatenate,Reshape
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import SGD, Adam
#from group_norm import GroupNormalization
import random
import pandas as pd 
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os, sys, copy, getopt, re, argparse
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import keras
# import transformers
import math

from tensorflow.keras.layers import Conv1D,BatchNormalization,MaxPooling1D,Dropout,Flatten,Dense,LSTM
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np




from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask,CapsuleLayer_nogradient_stop
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D,LayerNormalization, Dropout, Flatten, Dense, concatenate 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from keras import layers, models, optimizers
def CapsNet(input_shape, n_class, routings):

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




i=(41,4)
model, eval_model, manipulate_model = CapsNet(i,2,5);
model.compile(optimizer=optimizers.Adam(lr=0.001, epsilon=1e-08), loss=['binary_crossentropy'],
                  metrics={'capsnet': 'accuracy'})
model.load_weights('model1.h5')

def encode_seq(s):
    Encode = {'A':[1,0,0,0],'C':[0,0,1,0],'G':[0,0,0,1],'U':[0,1,0,0]}
    return np.array([Encode[x] for x in s])



def listToString(s):  
    
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1   
  
file = open("Positive.txt", "r")

count=0
Training=[0]*46529
for line in file:
  
  Data = line.split(':')
  Training[count] = Data
  count=count+1
  
elements1 = {}

accumulator=0
for row in Training:
  print(row)
  row=listToString(row)
  row=row.strip('\n')
  my_hottie = encode_seq(listToString(row))
  out_final=my_hottie
  out_final=out_final.astype(int)
  out_final = np.array(out_final)
  elements1[accumulator]=out_final
  #out_final=list(out_final)
  elements1[accumulator] = out_final
  accumulator += 1


nts= ['A','C', 'G', 'U']
x_train=Training
x_train=x_train[0:10000]
mutation = np.zeros((len(x_train),41,4)) # x_train is your original sequence (In ACGT Form)

for k, seq in enumerate(x_train):
    orig = seq
    #input_ref_seq = encode_seq(seq)  # seq is your onehot encoded form of sequence
    row=listToString(seq)
    row=row.strip('\n')
    input_ref_seq = encode_seq(listToString(row))
    input_ref_seq = np.expand_dims(input_ref_seq,axis=0)
    #print (input_ref_seq.shape)
    
    ref_pred = eval_model.predict(input_ref_seq)[0][0]
    #print(ref_pred)
    
    
    

    for i in range(41):
        for j, nt in enumerate(nts):
            tt = orig
            txt=tt[0]
            txt = list(txt)
            txt[i]=nt
            #print(alt_seq)
            #alt_seq[i]= nt
            alt_seq = ''.join(txt)
            row2=listToString(alt_seq)
            row2=row2.strip('\n')
            input_alt_seq = encode_seq(listToString(row2))
            input_alt_seq = np.expand_dims(input_alt_seq,axis=0)
            alt_pred= eval_model.predict(input_alt_seq)
            diff = abs(ref_pred - alt_pred[0][0])  # you may try the absolute differences
            print(diff)
            k=int(k)
            mutation[k,i,j] = diff[0]
            alt_seq=''



mutagenesis = np.average(mutation,axis=0)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# grid_kws = {"height_ratios": (2, .08), "hspace": .08}
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

#flights_df = flight.pivot('Sequence', 'Nucleotides') 

x_axis_labels = ['A', 'C', 'G', 'U'] # labels for x-axis
y_axis_labels = np.arange(1,42,step=1) # labels for y-axis
#mutation = np.random.randn(50, 20)
fig, ax = plt.subplots(figsize=(20,20))
sns.set(font_scale=1.8)
ax = sns.heatmap(mutagenesis, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="jet", vmax=.3, ax=ax)

plt.xlabel('Nucleotides')
plt.ylabel('Sequence')
fig.savefig("Insilico_Mutation_heatmap_Human.pdf")
