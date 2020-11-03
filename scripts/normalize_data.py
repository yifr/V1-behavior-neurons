import os
import h5py
from tqdm import tqdm
import behavenet
import numpy as np

mouse = 'MD0ST5'
expt = 'full_trial'
sessions = ['1', '2', '3', '4']

for session in sessions:
    data_path = os.path.join(behavenet.get_user_dir('data'), 'dipoppa', expt, mouse, session, 'data.hdf5')
    f = h5py.File(data_path, 'a', swmr=True, libver='latest')

    # Concatenate all trials together

    data = np.array([])

    neural = f['neural']

    print('Loading data for session: ', session)
    for trial_name in tqdm(neural):
        trial = neural[trial_name][:]

        if data.size == 0:
            data = trial
        else:
            data = np.concatenate([data, trial], axis=0)

    print('Computing mean, std...')

    # Compute mean and std
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(np.mean(mean), np.mean(std))
    '''
    for trial_name in tqdm(neural):
        neural[trial_name][:] -= mean
        neural[trial_name][:] /= std
    '''
    f.close()

