import mne
import pandas
import numpy as np
import os
from sklearn.model_selection import train_test_split
from mne.decoding import Scaler
from tensorflow.keras import utils as np_utils
import pickle
import random
from sklearn.utils import shuffle

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
    :param raw: The raw data from load data function
    :param duration: Defind the considering task duration of each epoch
    :param overlap:The overlap of each epoch
    :param drop_bad_eooch: Defind peak-to-peak amplitude in uv
    :return:epochs (trial x channels x  sampling point x )
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
    random.shuffle(file_list)
    try:
        file_list.remove('.DS_Store')
    except:
        print('There are no file name .DS_store')
        pass
    for i in file_list:
        print(i)
        file_path = os.path.join(folder_path, i)
        raw=load_raw_data(file_path)
        raw.filter(4,45)
        raw.notch_filter(np.arange(50,64,50))
        epochs = create_epochs(raw,duration=5,overlap=4,drop_bad_eooch=80e-6,flat_cri=1e-6)
        if len(X) == 0 and len(y) == 0:
            X = epochs.get_data()
            y = epochs.events
        else:
            X=np.concatenate((X, epochs.get_data()), axis=0)
            y=np.concatenate((y, epochs.events), axis=0)
    # preprocess before training the machine learning models
    y = y[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25,shuffle=False)
    scaler = Scaler(epochs.info, scalings='mean')
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    #shuffle train data
    X_train, y_train = shuffle(X_train, y_train)
    # #Save data
    data_folder_path='DASPS_Database/Data_resnet50'
    with open(data_folder_path + '/' + 'duration_5_overlap_4_V2.pkl','wb') as db_file:
        pickle.dump(obj=dict(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test), file=db_file)