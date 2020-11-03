import os
import h5py
import errno
import logging
import behavenet
import numpy as np
from scipy.io import loadmat

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def main():
    session_params = {
         'lab': 'dipoppa',
         'experiment_name': 'full_trial',
         'animal': 'MD0ST5',
         'date': '2018-04-04',
         'sessions': [4],
         }

    sample_sizes = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1370]
    n_samples = 50
    # sample_n_neurons(session_params, sample_sizes, n_samples)

    inh, exc, all_na = get_cell_idxs(session_params)
    logging.info('Inh size: {}, Exc size: {}, total neurons: {}'.format(len(inh), len(exc), len(all_na)))
    add_cell_type_samples(session_params, inh, exc, n_samples=100)

def get_cell_idxs(session_params, inh_key=3):
    '''
    Return indexes of all the inhibitory, excitatory, and overall good cells
    Requires info_cells file in same session level directory of behavenet's data directory
    (ie; if my data.hdf5 files are in "lab/experiment/animal/session_1/data.hdf5", the info_cells file
    should be in the same folder as "session_1")
    Params:
        session_params:dict: Dictionary specifying relevant details for session
        inh_key:int: key indicitating inhibitory cell type (2: PV, 3: SST, 5: GAD)
    Returns:
        inhibitory indexes, excitatory indexes, all good cell indexes
    '''
    lab = session_params.get('lab')
    experiment_name = session_params.get('experiment_name')
    animal = session_params.get('animal')
    date = session_params.get('date')

    # Load file and separate contents
    key = 'info_cells_' + animal + '_' + date + '.mat'
    cell_data_path = os.path.join(behavenet.get_user_dir('data'), lab, experiment_name, animal, key)
    if not os.path.exists(cell_data_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cell_data_path)

    info_cells = loadmat(cell_data_path)['info_cells']

    good_cells = info_cells[0][0][1][0]
    cell_types = info_cells[0][0][0][0]

    good_cells = info_cells[0][0][1][0]
    cell_types = info_cells[0][0][0][0]

    # Remove bad cells from neural activity and labelled cell data
    bad_cells = [i for i in range(len(good_cells)) if good_cells[i] == 0]
    cell_types = np.delete(cell_types, bad_cells, axis=0)

    # Collect indexes of inhibitory and excitatory neurons
    inh_idxs = [idx for idx,key in enumerate(cell_types) if key == inh_key]
    exc_idxs = [idx for idx,key in enumerate(cell_types) if key == 0]

    return inh_idxs, exc_idxs, list(range(len(cell_types)))

def sample_n_neurons(session_params, sample_sizes, n_samples, group_class='samples', group_name='subsamples', prefix=''):
    '''
    Create N samples of randomly selected neurons in behavenet's HDF5 files. Supports multiple sessions for a given animal,
    but not multiple animals/experiements/labs.

    Params:
        session_params:dict: Each entry in session params dictionary should
             contain another dictionary specifying the following fields which define a dataset:
                - lab (str)
                - experiment_name (str)
                - animal (str)
                - sessions (list)
        sample_sizes:list: list specifying the number of neurons in each sample
        n_samples:int: how many repeats to take of each sample size (ie; n=10 would add 10 samples for each sample size)
        group_class:str: HDF5 upper level group name
        group_name:str: name of second level group to add these samples under
        prefix:str: identifying prefix for datasets

    '''
    lab = session_params.get('lab')
    experiment_name = session_params.get('experiment_name')
    animal = session_params.get('animal')
    sessions = session_params.get('sessions')

    for session in sessions:
        data_path = os.path.join(behavenet.get_user_dir('data'), lab, experiment_name, animal, str(session), 'data.hdf5')
        if not os.path.isfile(data_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)

        logging.info('HDF5 File: {}'.format(data_path))
        data = h5py.File(data_path, 'a', libver='latest', swmr=True)
        ex_trial = list(data['neural'].keys())[0]
        n_neural = data['neural'][ex_trial][:].shape[1]

        # Delete old data
        if group_name in data[group_class]:
            logging.info('Deleting old group: {}'.format(group_name))
            del data[group_class][group_name]

        group = data[group_class].require_group(group_name)
        for size in sample_sizes:
            for i in range(n_samples):
                dset = np.random.choice(n_neural, size, replace=False)
                dset_name = prefix + 'n{}_t{}'.format(size, i)
                logging.info('Adding dataset: {}'.format(dset_name))
                group.create_dataset(dset_name, data=dset)

        data.close()

        logging.info('Added {} total datasets'.format(n_samples * len(sample_sizes)))

def add_cell_type_samples(session_params, inh_idxs, exc_idxs, n_samples, group_class='samples', group_name='cell_types', prefix=''):
    '''
    Adds n_samples of inhibitory and excitatory neurons. Uses the maximum number of neurons in the smaller
    dataset as default number of neurons
    params:
        session_params:dict: Dictionary containing information to extract data.hdf5 file
        inh_idxs:list: list of inhibitory cell indexes
        exc_idxs:list: excitatory cell indexes
        n_samples:int: how many random samples to take of each
        group_class:str: higher level group to add data to
        group_name:str: lower level group to add data to
        prefix:str: unique naming prefix (otherwise will be labelled exc_t{trial_number}, inh_t{trial_number})
    '''
    lab = session_params.get('lab')
    experiment_name = session_params.get('experiment_name')
    animal = session_params.get('animal')
    sessions = session_params.get('sessions')

    for session_number in sessions:
        data_path = os.path.join(behavenet.get_user_dir('data'), lab, experiment_name, animal, str(session_number), 'data.hdf5')
        if not os.path.isfile(data_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)

        logging.info('HDF5 File: {}'.format(data_path))
        data = h5py.File(data_path, 'a', libver='latest', swmr=True)
        data.require_group(group_class)

        # Delete old data
        if group_name in data[group_class]:
            logging.info('Deleting old group: {}'.format(group_name))
            #del data[group_class][group_name]

        group = data[group_class].require_group(group_name)

        n_neurons = min(len(inh_idxs), len(exc_idxs))

        # Assumes that there are fewer inhibitory neurons than excitatory (so we only need one inhibitory sample)
        # TODO: Probably a fair assumption, but correct this
        inh_dataset = np.random.choice(inh_idxs, n_neurons, replace=False)
        group.create_dataset(prefix + 'inh_all', data=inh_dataset)
        logging.info('Added Inhibitory dataset: {}'.format(inh_dataset.shape))
        for i in range(n_samples):
            exc_dataset = np.random.choice(exc_idxs, n_neurons, replace=False)
            group.create_dataset(prefix + 'exc_t{}'.format(i), data=exc_dataset)
            logging.info('Added excitatory dataset {}: {}'.format(i, exc_dataset.shape))



        data.close()

if __name__=='__main__':
    main()
