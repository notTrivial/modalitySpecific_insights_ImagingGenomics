import tensorflow as tf




import tensorflow as tf
import csv
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Input

from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, Nadam
from tensorflow.keras.regularizers import l1_l2, l1, l2
# from tensorflow_addons.layers import GroupNormalization, WeightNormalization
from sklearn.metrics import confusion_matrix
from datagenerator_pd_pheno import DataGenerator
from tensorflow.keras.utils import to_categorical
import os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.impute import SimpleImputer

import os
from os import listdir
from os.path import isfile, join

#from keras.engine.topology import Layer
from keras.layers import Activation, Lambda, Conv1D, SpatialDropout1D, add
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, concatenate
from tensorflow.keras.models import Model
from sklearn.impute import KNNImputer
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.impute import SimpleImputer

from tensorflow.keras.callbacks import EarlyStopping
import pickle




#############################################################################
############ SET SEEDS
############################################################################
tf.random.set_seed(1)
import random

random.seed(1)

os.environ['TF_DETERMINISTIC_OPS'] = '1'

#############################################################################
############ ARG PARSER/PARAMS
############################################################################
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_val', type=str, help='validation set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-fn_geno', type=str, help='genotype file')
parser.add_argument('-fold', type=int, help='fold #')
parser.add_argument('-results_path', type=str, help='directory for results?')
parser.add_argument('-model_name', type=str, help='model name to save')
parser.add_argument('-split', type=str, help='cross val size split eg 701515')
args = parser.parse_args()

params = {'batch_size': 2,
          'imagex': 193,
          'imagey': 229,
          'imagez': 193,
          'snps' : 61,
          'column': "Group_numeric"
          }

#############################################################################
############ FUNCTIONS
############################################################################

def sfcn(inputLayer):
    # block 1
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', name="conv1")(inputLayer[0])
    x = BatchNormalization(name="norm1")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool1")(x)
    x = ReLU()(x)

    # block 2
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name="conv2")(x)
    x = BatchNormalization(name="norm2")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool2")(x)
    x = ReLU()(x)

    # block 3
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', name="conv3")(x)
    x = BatchNormalization(name="norm3")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool3")(x)
    x = ReLU()(x)

    # block 4
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', name="conv4")(x) # 256
    x = BatchNormalization(name="norm4")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool4")(x)
    x = ReLU()(x)

    # block 5
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', name="conv5")(x)
    x = BatchNormalization(name="norm5")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool5")(x)
    x = ReLU()(x)

    # block 6
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), padding='same', name="conv6")(x)
    x = BatchNormalization(name="norm6")(x)
    x = ReLU()(x)

    # block 7
    x = AveragePooling3D()(x)
    x = Dropout(.2)(x)
    flatten = Flatten(name="flat1")(x)
    x = Dense(units=256, activation='relu', name="dense1")(flatten)
    x = Dense(units=1, activation='sigmoid', name="dense2")(x)

    return x #flatten


def geno_cnn(inputLayer):
    # Input layer
    #input_layer = Input(shape=(46314, 1))

    # Convolutional layers
    conv1 = Conv1D(filters=16, kernel_size=31, activation='relu',name="conv1G")((inputLayer[0]))
    batch_norm1 = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=6,name="pool1G")(batch_norm1)

    conv2 = Conv1D(filters=32, kernel_size=31, activation='relu',name="conv2G")(max_pooling1)
    batch_norm2 = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=6,name="pool2G")(batch_norm2)

    conv3 = Conv1D(filters=64, kernel_size=15, activation='relu',name="conv3G")(max_pooling2)
    max_pooling3 = MaxPooling1D(pool_size=6, name="pool3G")(conv3)

    # Flatten layer
    flatten = Flatten()(max_pooling3)

    # Dense layers
    #dense1 = Dense(128, activation='relu')(flatten)
    #output_layer = Dense(1, activation='sigmoid')(dense1)

    #return output_layer
    return flatten

def fusion_model(geno_flatten,pheno_flatten):
    #inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
    #inputB = Input(shape=(params['snps'], 1), name="InputB")
    #z = sfcn([inputA])
    #y = geno_cnn([inputB])

    x = Concatenate()([geno_flatten, pheno_flatten])
    x = Dense(units=128, activation='relu', name="dense1")(x)
    pred = Dense(units=1, activation='sigmoid', name="dense2")(x)

    return pred


def compile_model():
    opt = Adam(lr=0.0001)
    metr = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]
    inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
    #inputB = Input(shape=(params['snps'],1),name="InputB")
    z = sfcn([inputA])
    #y = geno_cnn([inputB])
    #pred = fusion_model(y,z)
    model = Model(inputs=[inputA], outputs=[z])
    model.summary()
    sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.7, nesterov=True)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metr) #opt
    return model


def scheduler(epoch, lr):
    #return lr * tf.math.exp(-0.5)
    return (1/(1+(1*epoch)))*0.0001

#############################################################################
############ INITIALIZE VARS
############################################################################

from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

geno_file = args.fn_geno

fold = args.fold
split = args.split

results_path = args.results_path

#############################################################################
############ INITIALIZE GENERATORS
############################################################################
fn_train = args.fn_train
train = pd.read_csv(fn_train, sep=",")
IDs_list = train['PATNO'].to_numpy()
train_IDs = IDs_list
training_generator = DataGenerator(train_IDs, params['batch_size'],
                                   (params['imagex'], params['imagey'], params['imagez']), True, fn_train,
                                   params['column'], params['snps'], geno_file)

#data_iterator = iter(training_generator)
#batch = next(data_iterator)
#print(batch)
#print(next(training_generator))
#print(training_generator)

fn_val = args.fn_val
val = pd.read_csv(fn_val, sep=",")
IDs_list = val['PATNO'].to_numpy()
val_IDs = IDs_list
val_generator = DataGenerator(val_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']),
                              False, fn_val, params['column'], params['snps'], geno_file)

batch_size = params['batch_size']
learning_rate = 0.0001

fn_test = args.fn_test
test = pd.read_csv(fn_test, sep=",")
IDs_list = test['PATNO'].to_numpy()
test_IDs = IDs_list
test_generator = DataGenerator(test_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']),
                              False, fn_test, params['column'], params['snps'], geno_file)

#############################################################################
############################################################################

#############################################################################
############ INITIALIZE/TRAIN MODEL
############################################################################
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model = compile_model()
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"{results_path}{args.model_name}_fold_{fold}_epoch{{epoch}}_loss{{val_loss:.4f}}_acc{{val_accuracy:.4f}}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{split}Split.h5"
                                                         , monitor='val_loss', verbose=2, save_best_only=True, include_optimizer=True, save_weights_only=False, mode='auto',
                                                         save_freq='epoch') #monitor='val_loss


history = model.fit(training_generator, epochs=1000, validation_data=val_generator,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13), checkpoint_callback], verbose=2) # , lr_callback

# Plot the loss curves for this fold
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Loss Curves for Fold {fold}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_curve_filename = f'{results_path}loss_curves_lr{learning_rate}_batch{batch_size}_fold_{fold}_ppmi_adni_{split}Split.svg'
plt.savefig(loss_curve_filename, format='svg', dpi=300)
plt.close()
print(f"[INFO] Loss curve saved for fold {fold}")


history_dict = history.history

#############################################################################
############ TEST SET INVESTIGATIONS
############################################################################

test_predictions, test_true_labels, patient_ids = [], [], []

all_patient_ids = []
all_y_true = []
all_y_pred = []
all_y_pred_binary = []


for i in range(len(test_generator)):
    X_test, y_test = test_generator[i]

    patient_id_batch = test_generator.get_current_IDs()
    y_pred_batch = model.predict(X_test)
    y_pred_binary_batch = (y_pred_batch >= 0.5).astype(int)

    del X_test  # Free memory after using X_test

    # Write results incrementally to CSV
    #with open(
    #        f'{results_path}predictions_test_fold_{fold}_lr_{learning_rate}_batchSize_{batch_size}_adni_ppmi_imgOnly.csv',
    #        'a') as f:
    #    if i == 0:
    #        f.write("Patient_ID,True_Label,Predicted_Probability,Predicted_Label\n")
    #    
    #    for j in range(len(patient_id_batch)):
    #        patient_id = patient_id_batch[j]
    #        true_label = y_test[j]
    #        predicted_prob = y_pred_batch[j]
    #        predicted_label = y_pred_binary_batch[j]

    #        # Directly write to the CSV
    #        f.write(f"{patient_id},{true_label},{predicted_label},{predicted_binary}\n")

    all_patient_ids.extend(patient_id_batch)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred_batch.flatten())
    all_y_pred_binary.extend(y_pred_binary_batch.flatten())



results_df = pd.DataFrame({
    'PATNO': all_patient_ids,
    'True_Label': all_y_true,
    'Actual_Value': all_y_pred,
    'Predicted_Label': all_y_pred_binary
})

results_csv_path = f'{results_path}predictions_test_fold_{fold}_lr_{learning_rate}_batchSize_{batch_size}_adni_ppmi_imgOnly_{split}Split.csv'
results_df.to_csv(results_csv_path, index=False)




print("Results of test set for fold " + str(fold) + " saved")

#############################################################################
############ CLOSING STUFF
############################################################################
del test_generator
del val_generator
del training_generator

from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
