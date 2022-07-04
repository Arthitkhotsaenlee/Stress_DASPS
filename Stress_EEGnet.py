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
import pickle

#Load data process
labels = dict(Anxious=0,Normal=1)

def load_raw_data(file_path):
    """
    this function using for load raw EEG data from .fif format
    :param file_path: The path_dir of a raw EEG file
    :return: rew data mne class
    """
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.set_eeg_reference(ref_channels=['T8'])
    raw.pick_channels(['T7'])
    return raw
def create_epochs(raw,duration=15,overlap=0,drop_bad_eooch=40e-1,flat_cri=0.5e-6):
    """

    :param raw:
    :param duration:
    :param overlap:
    :param drop_bad_eooch:
    :return:
    """
    task_id=labels[raw.annotations.description[0]]
    epochs=mne.make_fixed_length_epochs(raw=raw, duration=duration,overlap=overlap,id=task_id, verbose=False,preload=True)
    epochs.drop_bad(reject={'eeg':drop_bad_eooch},flat={'eeg':flat_cri})
    return epochs
if __name__ == '__main__':
    # Load data part
    X=[]
    y=[]
    folder_path = 'DASPS_Database/Preprocessed data 2clsaaes .fif'
    file_list = os.listdir(folder_path)
    try:
        file_list.remove('.DS_Store')
    except:
        pass
    for i in file_list:
        print(i)
        file_path = os.path.join(folder_path, i)
        raw=load_raw_data(file_path)
        raw.filter(4,45)
        raw.notch_filter(np.arange(50,64,50))
        epochs = create_epochs(raw,duration=5,overlap=4,drop_bad_eooch=100e-6,flat_cri=1e-6)
        if len(X) == 0 and len(y) == 0:
            X = epochs.get_data()
            y = epochs.events
        else:
            X=np.concatenate((X, epochs.get_data()), axis=0)
            y=np.concatenate((y, epochs.events), axis=0)
    # preprocess before training the machine learning models
    y = y[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
    scaler = Scaler(epochs.info,scalings='mean')
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    #Save data
    data_folder_path='DASPS_Database/Data_EEGnet'
    with open(data_folder_path + '/' + 'duration_5_overlap_4.pkl','wb') as db_file:
        pickle.dump(obj=dict(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test), file=db_file)

    #delete raw and epochs
    del raw, epochs
    #EEGnet Algorithm part
    kernels, channels, samples = 1, X.shape[1], X.shape[2]
    class_weigths = {0:1, 1:1}
    #reshape data tobe fit with the input models
    X_test = X_test.reshape(X_test.shape[0], channels, samples, kernels)
    X_train=X_train.reshape(X_train.shape[0], channels, samples, kernels)
    #create EEGnet model
    #
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
    # model.compile(loss='binary_crossentropy',
    #               optimizer=tf.keras.optimizers.RMSprop(
    #                                                     learning_rate=0.001,
    #                                                     rho=0.9,
    #                                                     momentum=0.0,
    #                                                     epsilon=1e-07,
    #                                                     centered=False,
    #                                                     name="RMSprop"
    #                                                 ),
    #               metrics='accuracy')
    model.summary()
    numParameters = model.count_params()
    check_point = ModelCheckpoint(filepath='DASPS_Database/EEG_model/2_class.h5')
    fitted_model = model.fit(x=X_train,
                             y=y_train,
                             batch_size=32,
                             epochs=100,
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