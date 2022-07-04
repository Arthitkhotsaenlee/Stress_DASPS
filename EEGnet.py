import pickle
import mne
import pandas
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from EEGModels import EEGNet,EEGnet_MindAmpltd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from mne.decoding import Vectorizer,cross_val_multiscore,get_coef,Scaler
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
from tensorflow.

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.callback import TensorBoard
from tensorflow.
import time


def EEGNet(nb_classes, Chans=64, Samples=128,dropoutRate=0.5, kernLength=64, F1=8,D=2, F2=16, norm_rate=0.25):
    input1 = Input(shape=(Chans, Samples, 1))
    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = AveragePooling2D((1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 10),
                             use_bias=False,
                             padding='same')(block1)

    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    block2 = AveragePooling2D((1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


if __name__ == '__main__':
    # Load data
    data_name='duration_5_overlap_4.pkl'
    with open('DASPS_Database/Data_EEGnet/duration_5_overlap_4.pkl','rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    #EEGnet Algorithm part
    kernels, channels, samples = 1, X.shape[1], X.shape[2]
    class_weigths = {0:1, 1:1}
    #reshape data tobe fit with the input models
    X_test = X_test.reshape(X_test.shape[0], channels, samples, kernels)
    X_train=X_train.reshape(X_train.shape[0], channels, samples, kernels)
    #create EEGnet model
    model= EEGNet(nb_classes=2,
                             Chans=channels,
                             Samples=samples,
                             dropoutRate=0.5,
                             norm_rate=0.25,
                             kernLength=10,
                             F1=25,
                             D=2,
                             F2=25,
                             dropoutType='Dropout'
                             )
    #compile the deeplearning models
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')
    model.summary()
    numParameters = model.count_params()
    check_point = ModelCheckpoint(filepath='DASPS_Database/EEG_model/2_class.h5')
    fitted_model = model.fit(x=X_train,
                             y=y_train,
                             batch_size=32,
                             epochs=50,
                             validation_split=0.2,
                             callbacks=[check_point],
                             class_weight=class_weigths,
                             verbose=True
                             )
    model.load_weights(filepath='DASPS_Database/EEG_model/2_class.h5')
    predicts=model.predict(X_test)
    predicts=predicts.argmax(axis=-1)
    print(confusion_matrix(y_test,predicts))
    print('\n')
    print(classification_report(y_test,predicts))