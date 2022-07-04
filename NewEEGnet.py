import pickle
import itertools
import tensorflow.keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import io
# from tensorflow.plugins.hparams import api as hp
from sklearn.metrics import classification_report, confusion_matrix

import time
from datetime import datetime

def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
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
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    block2 = SeparableConv2D(F2, (1, 16),use_bias=False,padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    flatten = Flatten(name='flatten')(block2)
    #######addition as block3###########
    # block3 = Dense(F1, name='dense1', kernel_constraint=max_norm(norm_rate))(flatten)
    # block3 = Activation('relu')(block3)
    # block3 = Dropout(dropoutRate)(block3)
    block3 = Dense(int(F1/2), name='dense2', kernel_constraint=max_norm(norm_rate))(flatten)
    block3 = Activation('elu')(block3)
    # drop = Dropout(dropoutRate)(activation)
    block3 = Dense(nb_classes, name='dense3', kernel_constraint=max_norm(norm_rate))(block3)
    softmax = Activation('softmax', name='softmax')(block3)
    return Model(inputs=input1, outputs=softmax)



#plot confusion metrics
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confussion_matrix(logs):
    # load optimal weigth
    model.load_weights('DASPS_Database/EEG_model/2_class.h5')
    #Use model predict the value
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)
    actual=np.argmax(y_test, axis=1)
    #calculstion the confusion matrix
    cm = confusion_matrix(actual,test_pred)
    figure = plot_confusion_matrix(cm, class_names=class_name)
    cm_image = plot_to_image(figure)
    with file_writer_cm.as_default():
        tf.summary.image('Comfusion Matrix',cm_image, step=1)
def log_classification_report(logs):
    #load optimal weigth
    model.load_weights('DASPS_Database/EEG_model/2_class.h5')
    # Use model predict the value
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)
    actual = np.argmax(y_test, axis=1)
    # calculstion the confusion matrix
    cm = confusion_matrix(actual, test_pred)
    figure = plot_confusion_matrix(cm, class_names=class_name)
    cm_image = plot_to_image(figure)
    # classification report
    report=classification_report(actual, test_pred,target_names=class_name)
    with file_writer_report.as_default():
        tf.summary.text('classification_report', report, step=1)

if __name__ == '__main__':
    # Load data
    # data_name='duration_5_overlap_4.pkl'
    with open('DASPS_Database/Data_EEGnet/duration_5_overlap_4_V2.pkl','rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    class_name= ['Anxious', 'Normal']
    #EEGnet Algorithm part
    kernels, channels, samples = 1, X_train.shape[1], X_train.shape[2]
    class_weigths = {0:1, 1:1}
    #reshape data tobe fit with the input models
    X_test = X_test.reshape(X_test.shape[0], channels, samples, kernels)
    X_train=X_train.reshape(X_train.shape[0], channels, samples, kernels)
    ## EECnet portion
    F1s = [16,25,64]
    F2s = [16,25,32]
    kernLengths = [25,32,64]
    dropoutRates = [0.6,0.7,0.75]
    dropoutnorms = [0.1,0.2,0.4]
    for F1 in F1s:
        for F2 in F2s:
            for kl in kernLengths:
                for drop_out in dropoutRates:
                    for drop_norm in dropoutnorms:
                        Name = "F1-{}-F2-{}-KL-{}-dorp_out-{}-drop_norm-{}-time-{}".format(F1,
                                                                                               F2,kl,drop_out,
                                                                                               drop_norm,
                                                                                               int(time.time()))
                        tensorboard= TensorBoard(log_dir='logs/{}'.format(Name))
                        file_writer_cm = tf.summary.create_file_writer('logs/{}'.format(Name) + '/cm')
                        file_writer_report = tf.summary.create_file_writer('logs/{}'.format(Name) + '/report')
                        model=EEGNet(nb_classes=2,Chans=channels,
                                     Samples=samples,dropoutRate=drop_out,
                                     kernLength=kl,F1=F1,
                                     D=2, F2=F2,norm_rate=drop_norm)
                        model.compile(loss='categorical_crossentropy',
                                      optimizer='adam',
                                      metrics='accuracy')
                        #Define the per-each callback
                        cm_callback = keras.callbacks.LambdaCallback(on_train_end=log_confussion_matrix)
                        txt_callback = keras.callbacks.LambdaCallback(on_train_end=log_classification_report)
                        #check point
                        numParameters = model.count_params()
                        check_point = ModelCheckpoint(filepath='DASPS_Database/EEG_model/2_class.h5')
                        model.fit(x=X_train,
                                  y=y_train,
                                  batch_size=64,
                                  epochs=500,
                                  validation_split=0.25,
                                  # validation_data=(X_test,y_test),
                                  callbacks=[tensorboard,check_point, cm_callback,txt_callback],
                                  class_weight=class_weigths,
                                  verbose=True
                                  )


