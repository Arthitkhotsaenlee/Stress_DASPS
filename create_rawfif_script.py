import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py
import pandas as pd
import os


ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
def create_new_raw_data(data_path,df):
    save_dir='/Users/arthitkhotsaenlee/pythonProject/stress_project/DASPS_Database/Preprocessed data 2clsaaes .fif'
    subID=data_path.split('/')[-1].split('p')[0] #find subject ID from path
    # get_data_from .mat
    data = {}
    f = h5py.File(data_path)
    for k, v in f.items():
        data[k] = np.array(v)
    data=data['data']
    data = data.reshape((6,-1,14))
    # raw=[]
    # for i in range(data.shape[0]):
    #     raw.append(data[i,:,:].T)
    # raw=np.concatenate(raw,axis=1)
    # onset = np.arange(0, raw.shape[1], 1920 * 2) / 128
    for i in range(data.shape[0]):
        sit=i+1 # situation ID
        #create New raw data of each situation
        raw=(data[i,:,:].T)*1e-6
        info=mne.create_info(ch_names=ch_names,sfreq=128,ch_types=['eeg']*data.shape[2],verbose=False)
        raw=mne.io.RawArray(data=raw,info=info,verbose=False)
        #create annotation
        onset=[0]
        duration=[0]
        desc=df[(df['subjectID']==subID) & (df['situationID']==sit)]['2_class'].values[0]
        anno=mne.Annotations(onset=onset, duration=duration, description=desc)
        raw.set_annotations(anno)
        #save raw_data of ecah situation into .fif files
        # raw.save(save_dir+'/'+'{}-situation{}-{}_eeg.fif'.format(subID ,sit,desc), overwrite=True)



    #create raw data process


    return data

#get labels
exp_path='DASPS_Database/participant_rating_public_dummy.csv'
cal_names=['Id Participant','Id situation','valence','Arousal','2_class']
df = pd.read_csv(exp_path)


if __name__ == '__main__':
    # raw_all=mne.io.read_raw_edf('DASPS_Database/Raw data .edf/S01.edf',preload=True)
    # list_name=raw_all.ch_names
    # list_type=['eeg']*len(list_name)
    # raw_all.set_channel_types(dict(zip(list_name, list_type)))
    # # raw_eeg =raw.pick_type('eeg')
    # raw=raw_all.copy().pick_channels(list_name[2:15])
    # raw.notch_filter(np.arange(50,raw.info['sfreq']/2,50))
    # raw.filter(4,45)
    # raw.plot(duration=1)
    # raw.plot_psd(1,45)
    # plt.show()
    ##import math files
    # data=sio.loadmat('DASPS_Database/Preprocessed data .mat/S01preprocessed.mat')
    mat_list=os.listdir('DASPS_Database/Preprocessed data .mat')
    for i in mat_list:
        # raw_path='DASPS_Database/Preprocessed data .mat/S01preprocessed.mat'
        raw_path=os.path.join('DASPS_Database/Preprocessed data .mat',i)
        data=create_new_raw_data(raw_path,df)



