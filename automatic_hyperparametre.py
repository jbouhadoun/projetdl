from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy

import os

from contextlib import redirect_stdout
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Preparing the dataset
# Setup train and test splits
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#y_train = utils.to_categorical(y_train, 10)
#y_test = utils.to_categorical(y_test, 10)




# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


HP_STRUCTURE= hp.HParam('structure_model', hp.Discrete([2]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam',''sgd']))
HP_LEARNINGRATE=hp.HParam('leraning_rate', hp.Discrete([0.02,0.06]))
HP_MOMENTUM=hp.HParam('momentum', hp.Discrete([0.05]))
HP_L2=hp.HParam('l2', hp.Discrete([0.0]))
HP_ACTIVATION=hp.HParam('activation', hp.Discrete([ 'relu']))
HP_AUGMENTATION=hp.HParam('data_augmentation',hp.Discrete(["true"]))
batch_size=128

METRIC_ACCURACY = 'accuracy'
METRIC_LOSS='loss'
log_dir='logs/mp2/'
with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=[HP_STRUCTURE, HP_DROPOUT, HP_OPTIMIZER,HP_MOMENTUM,HP_LEARNINGRATE,HP_L2,HP_ACTIVATION,HP_AUGMENTATION],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy_test'),
             hp.Metric(METRIC_LOSS, display_name='loss_test' )
           ],
  )

def aument_model(x_train,y_train,batch_size,model):
  train_datagen = ImageDataGenerator(
          rotation_range=45,
          width_shift_range=.15,
          height_shift_range=.15,
          horizontal_flip=True,
          )
  train_datagen.fit(x_train)
  train_datagen_ = train_datagen.flow(x_train, y_train, batch_size=batch_size)
  val_datagen = ImageDataGenerator(
          rotation_range=45,
          width_shift_range=.15,
          height_shift_range=.15,
          horizontal_flip=True,
          )
  val_datagen.fit(x_test)
  val_datagen = val_datagen.flow(x_test, y_test, batch_size=batch_size)
      
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  history= model.fit_generator(train_datagen_,
                      validation_data=val_datagen,
                      epochs=100,
                      callbacks=[tensorboard_callback]
                                )
  return history
                
def model1(hparams):

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  ,kernel_regularizer=regularizers.l2(hparams[HP_L2])))
  model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
  model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
  return model
  
def model2(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(256, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  return model


def model3(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(512, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(256, activation=tf.nn.relu if hparams[HP_ACTIVATION]=='relu' else tf.nn.sigmoid  , kernel_regularizer=regularizers.l2(hparams[HP_L2])),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  return model
  


  

def train_test_model(hparams,log_dir,x_train,y_train):
  if hparams[HP_STRUCTURE]==1:
    model=model1(hparams)
  else :
    if hparams[HP_STRUCTURE]==2:
      model=model2(hparams)
    else :
      model=model3(hparams)

  if hparams[HP_OPTIMIZER]=='adam':
    optimizer=optimizers.Adam(lr= hparams[HP_LEARNINGRATE], decay=1e-6)
  else:
    optimizer= optimizers.SGD(lr= hparams[HP_LEARNINGRATE], decay=1e-6, momentum=hparams[HP_MOMENTUM])
  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  
  if  hparams[HP_AUGMENTATION]=="true":
    history=aument_model(x_train,y_train,batch_size,model)
  else:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history=model.fit(x_train, y_train, batch_size=batch_size,epochs=100,validation_split=.1,callbacks=[tensorboard_callback]) # Run with 1 epoch to speed things up for demo purposes
  loss,accuracy = model.evaluate(x_test, y_test)
  model.save_weights(log_dir+'/my_model_weights.h5')
  plot_model(model,log_dir+'/my_model_weights.png')
  with open(log_dir+'/my_model_summary.txt', 'w') as f:
     with redirect_stdout(f):
         model.summary()

  return loss,accuracy




def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    loss,accuracy = train_test_model(hparams,run_dir,x_train,y_train)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    tf.summary.scalar(METRIC_LOSS, loss, step=1)



def lancer(): 
  session_num = 0

  for aug in HP_AUGMENTATION.domain.values:
    for activate in HP_ACTIVATION.domain.values:
      for structure in HP_STRUCTURE.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
          for optimizer in HP_OPTIMIZER.domain.values:
            for learning_rate in HP_LEARNINGRATE.domain.values:
              for momentum in HP_MOMENTUM.domain.values:
                for l2 in HP_L2.domain.values:
                  print(optimizer)
                  hparams = {
                        HP_AUGMENTATION: aug,
                        HP_STRUCTURE: structure,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                        HP_L2: l2,
                        HP_MOMENTUM: momentum,
                        HP_LEARNINGRATE: learning_rate,
                        HP_ACTIVATION :activate
                        }
                  if structure==1 :
                    modelstrcture="layer_512*10"
                  else :
                    if structure==2 :
                      modelstrcture="layer_1024*512*256*10"
                    else :
                      modelstrcture="8_layers_1024*...*10"
                  run_name = "mo_"+modelstrcture+"_aug_"+str(aug)+"_act_"+str(activate)+"_do_"+str(dropout_rate)+"_l2_"+str(l2)+"_op_"+str(optimizer)+"_lr_"+str(learning_rate)+"_mome_"+str(momentum)
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparams[h] for h in hparams})
                  run(log_dir+run_name, hparams)
                  session_num += 1
                  print("-----------session_num :"+str(session_num)+"---------------")
