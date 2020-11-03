import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import utils
from skimage import transform
from scipy.io import loadmat
from behavenet import get_user_dir

def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Args:
        cap (cv2.VideoCapture object):
        idxs (array-like): frame indices into vido

    Returns:
        np.ndarray of shape (n_frames, y_pix, x_pix)
    """
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0:
            cap.set(1 , i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames

def ZScore(sample, axis=0):
    mean = sample.mean(axis=axis)
    std = sample.std(axis=axis)
    return (sample - mean) / std

def process(sess_id,
            lab='dipoppa',
            expt='SSSVAE',
            animal='MD0ST5',
            date='2018-04-04',
            neural_data_root='/home/yoni/behavenet/neural_dir'):

    vid_id = '{}_{}_{}'.format(date, sess_id, animal)
    neural_dir = os.path.join(neural_data_root, animal, date)

    # data will be stored in data_dir/lab/expt/animal/session/data.hdf5
    lab = lab
    expt = expt
    animal = animal
    date = date
    session = sess_id

    mp4_file = os.path.join(neural_dir, session, vid_id + '_eye.mj2')# (can be any format loadable by open cv)

    # video frames will be resized to these dimensions
    # downsampling is preferrable as it speeds up autoencoder fitting time
    # we typically use images with the largest dimension <= 256
    xpix = 256 # choose a number (e.g. 256)
    ypix = 128 # choose a number (e.g. 128)

    # processed data in behavenet format will be stored here
    data_dir = get_user_dir('data')
    proc_data_filepath = os.path.join(data_dir, lab, expt, animal, session)
    print('Writing to: ', proc_data_filepath, 'Video from: ', mp4_file)

    ###########
    # Load data
    ###########

    # set up hdf5 file
    hdf5_file = os.path.join(proc_data_filepath, 'data.hdf5')
    if False:# os.path.exists(hdf5_file):
        raise IOError('data.hdf5 file already exists; skipping')
    else:
        hdf5_dir = os.path.dirname(hdf5_file)
        if not os.path.exists(hdf5_dir):
            os.makedirs(hdf5_dir)

    # read video file and check
    cap = cv2.VideoCapture(mp4_file)
    if not cap.isOpened():
        raise IOError('error opening video file at %s' % mp4_file)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('Loading and processing data...')
    # Load trial info to sync video/neural data, and neural activity
    vvi = loadmat(os.path.join(neural_dir, sess_id, 'batch_video_id'  + vid_id + '.mat'))['batch_video_id']
    nni = loadmat(os.path.join(neural_dir, sess_id, 'batch_neural_id'  + vid_id + '.mat'))['batch_neural_id']
    nna = loadmat(os.path.join(neural_dir, sess_id, 'neural_activity'  + vid_id + '.mat'))['neural_activity']

    # Load cell labels
    cell_data_path = os.path.join(neural_dir, 'info_cells_{}_{}.mat'.format(animal, date))
    info_cells = loadmat(cell_data_path)['info_cells']

    good_cells = info_cells[0][0][1][0]
    cell_types = info_cells[0][0][0][0]

    # Remove bad cells from neural activity and labelled cell data
    bad_cells = [i for i in range(len(good_cells)) if good_cells[i] == 0]
    cell_types = np.delete(cell_types, bad_cells, axis=0)
    nna = np.delete(nna, bad_cells, axis=1)
    nna = ZScore(nna)

    # Collect indexes of inhibitory and excitatory neurons
    inh_idxs = [i for i in range(len(cell_types)) if cell_types[i] == 3]
    exc_idxs = [i for i in range(len(cell_types)) if cell_types[i] == 0]

    # Load facemap data and create labels
    facemap_data = np.load(os.path.join(neural_dir, sess_id, 'facemap', vid_id + '_eye_proc.npy'),
                          allow_pickle=True).item()

    pupil_area = facemap_data['pupil'][0]['area_smooth'].reshape(-1, 1)
    pupil_com = facemap_data['pupil'][0]['com_smooth']
    print('Neural data: ', nna.shape, '# Inhibitory neurons: ', len(inh_idxs), '# Excitatory neurons: ', len(exc_idxs))

    # Labels_sc are the un-normalized versions of the labels
    labels_sc = np.concatenate([pupil_area, pupil_com], axis=1)

    # Labels are a collection of ZScored pupil area and center of mass
    p_area_norm = ZScore(pupil_area, axis=0)
    p_com_norm = ZScore(pupil_com, axis=0)
    labels = np.concatenate([p_area_norm, p_com_norm], axis=1)
    print('Label shape (scaled+unscaled): ', labels.shape, labels_sc.shape)

    n_trials = vvi.shape[1]
    print(hdf5_file)
    print('Creating hdf5 file')
    t_beg = time.time()
    f = h5py.File(hdf5_file, 'a', libver='latest', swmr=True)
    with f as f:

        # single write multi-read
        f.swmr_mode = True

        # create image group
        group_i = f.require_group('images')

        # create neural group
        group_n = f.require_group('neural')

        # Create group for behavioral variables
        group_labels = f.require_group('labels')
        group_labels_sc = f.require_group('labels_sc')

        # create a dataset for each trial within groups
        t = 0
        for trial in range(n_trials):

            if trial % 10 == 0:
                print('processing trial %03i' % trial)

            # find video indices during this trial
            trial_beg = vvi[0, trial]
            trial_end = vvi[1, trial]
            ts_idxs = np.arange(trial_beg-1,trial_end)

            # load and process corresponding frames
            frames = get_frames_from_idxs(cap, ts_idxs)
            sh = frames.shape
            frames_proc = np.zeros((sh[0], sh[1], ypix, xpix), dtype='uint8')
            for i in range(sh[0]):
                frames_proc[i, 0, :, :] = cv2.resize(
                    frames[i, 0], (xpix, ypix))

            # save image data
            group_i.create_dataset('trial_%04i' % t, data=frames_proc, dtype='uint8')

            label_frames = labels[ts_idxs]
            label_sc_frames = labels_sc[ts_idxs]
            #print('Image: ', frames_proc.shape, 'Labels: ', label_frames.shape, label_sc_frames.shape)

            group_labels.create_dataset('trial_%04i' % t, data=label_frames, dtype='float32')
            group_labels_sc.create_dataset('trial_%04i' % t, data=label_sc_frames, dtype='float32')

            # find neural indices during this trial
            trial_beg = nni[0, trial]
            trial_end = nni[1, trial]

            # pick out corresponding neural activity
            neural = nna[trial_beg-1:trial_end,:]

            # save neural data
            group_n.create_dataset('trial_%04i' % t, data=neural, dtype='float32')
            t += 1

        # Add variable size subsamples
        samples = f.create_group('samples')
        subsamples = samples.create_group('subsamples')
        subsample_sizes = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
        n_cells = range(nna.shape[1])
        for size in subsample_sizes:
            for trial in range(10):
                # Choose random sample w/out replacement from all neural indexes
                subsample = np.random.choice(n_cells, size, replace=False)
                subsamples.create_dataset('n{}_t{}'.format(size, trial), data=subsample)

        # Add Inh/Exc subsamples
        cell_types = samples.create_group('cell_types')
        cell_types.create_dataset('inh_t0', data=inh_idxs, dtype='uint8')

        for i in range(50):
            exc_sample = np.random.choice(exc_idxs, len(inh_idxs), replace=False)
            cell_types.create_dataset('exc_t{}'.format(i), data=exc_sample, dtype='uint8')

    # print out timing info
    t_end = time.time()
    t_tot = t_end - t_beg
    print('Processed {} frames in total'.format(t))
    print('total processing time: %f sec' % t_tot)
    print('time per trial: %f sec' % (t_tot / n_trials))

def main():
    sess_ids = [4] #[1,2,3,4]
    lab = 'dipoppa'
    mouse = 'MD0ST5'
    experiment = 'full_trial'
    date = '2018-04-04'
    for sess_id in sess_ids:
        print('PROCESSING SESSION %d'%sess_id)
        process(str(sess_id), lab=lab, animal=mouse, expt=experiment, date=date)

    print('All Done')

if __name__=='__main__':
    main()
