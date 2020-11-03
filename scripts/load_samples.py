import h5py
import numpy as np
from scipy.io import loadmat
import sys
from behavenet import get_user_dir

def main():

    # Get hdf5 file
    fs = []
    data_dir = get_user_dir('data')
    for i in [1, 3, 4, 5, 6]:
        path = data_dir + '/dipoppa/MSP_pupil/SB028/{}/data.hdf5'.format(i)
        fs.append(h5py.File(path, 'a', libver='latest', swmr=True))

    print(fs)
    # Get cell types
    info_cells = loadmat('neural_dir/SB028/2019-11-08/info_cells_SB028_2019-11-08.mat')['info_cells']
    cell_type_idxs = info_cells[0][0][0][0]    # 0 = excitatory, 3 = inhibitory
    good_cell_idxs = info_cells[0][0][1][0]

    # Label all "bad" cells as -1
    cell_type_idxs = [cell_type_idxs[i] if good_cell_idxs[i] == 1 else -1 for i in range(len(good_cell_idxs))]
    def select_idxs(arr, key):
        return [i for i in range(len(arr)) if arr[i] == key]

    inh_id = [i for i in cell_type_idxs if i != 0 and i != -1][0]
    print('Inhibitory Cell Type: ' + str(inh_id))

    inh_idxs = select_idxs(cell_type_idxs, inh_id)
    exc_idxs = select_idxs(cell_type_idxs, 0)

    num_cells = len(cell_type_idxs)
    print(num_cells, 'NUM CELLS')
    print('inh: ', len(inh_idxs))
    print('exc: ', len(exc_idxs))

    '''
    for f in fs:
        if f.get('samples'):
            del f['samples']
            del f['samples']
    '''
    print('Adding subsamples')
    add_subsamples(fs, num_cells)
    # print('Adding Histogram Trials')
    add_cell_type(fs, inh_idxs, exc_idxs)

    print('Adding powerlaw samples')
    cell_types_powerlaw(fs, inh_idxs, exc_idxs)
    print('...done')
    close(fs)

def add_subsamples(fs, num_cells):
    for f in fs:
        f.require_group('samples')
        if not f['samples'].get('subsamples'):
            f['samples'].create_group('subsamples')
        print('Processing File: {}'.format(fs.index(f)))
        subsamples = f['samples']['subsamples']
        sample_sizes = [1, 5, 10, 20, 40, 60, 80,100,200,300,400,450,500,600,700,800,900,950]
        for size in sample_sizes:
            print('Creating {} trials for {} neurons...'.format(10, size))
            for t in range(10):
                data = np.random.choice(list(range(950)), size, replace=False)
                dname = 'sample_{}_t{}'.format(size, t)
                try:
                    subsamples.create_dataset(dname, data=data)
                except:
                    print('Samples "{}" already exists; skipping...'.format(dname))
        print('Completed\n')

def add_cell_type(fs, inhibitory, excitatory):
    cell_types = []
    for f in fs:
        if not f['samples'].get('cell_types'):
            cell_type = f['samples'].create_group('cell_types')
        cell_types.append(f['samples']['cell_types'])
    print(cell_types)

    # Add inhibitory idxs for each trial
    for ct in cell_types:
        if not ct.get('inh_all'):
            print('Adding new inh dataset to session  #', cell_types.index(ct))
            ct.create_dataset('inh_all', data=inhibitory)


    for ct in cell_types:
        for i in range(50):
            dname = 'exc85_sample_{}'.format(i)
            data = np.random.choice(excitatory, len(inhibitory), replace=False)
            if not ct.get(dname):
                print('ct: ', ct, 'Data: ', data)
                ct.create_dataset(dname, data=data)
            else:
                print('{} already exists; skipping...'.format(dname))

def cell_types_powerlaw(fs, inhibitory, excitatory):
    trial_cells = []
    for f in fs:
        if not f['samples'].get('cell_types'):
            cell_type = f['samples'].create_group('cell_types')
        trial_cells.append(f['samples']['cell_types'])

    counter = 0
    sample_sizes = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 204]
    # Go through each trial dataset
    for trial in trial_cells:
        print('Generating data for trial {}'.format(trial_cells.index(trial)))

        # Add SST and non-SST data
        for ctype, cell_idxs in zip(['inh', 'exc'], [inhibitory, excitatory]):
            print('Cell Type: {}'.format(ctype))

            for size in sample_sizes:
                print('\tAdding 10 trials for sample size {}'.format(size))

                for t in range(10):
                    data = np.random.choice(cell_idxs, size, replace=False)
                    dname = ctype + '{}_sample_{}'.format(size, t)
                    if not trial.get(dname):
                        trial.create_dataset(dname, data=data)
                        counter += 1
                    else:
                        print('\t{} already exists; skipping...'.format(dname))

    print('Added additional {} datasets...'.format(counter))

def close(fs):
    fs = list(fs)
    for i, f in enumerate(fs):
        print('Closing hdf5 file for trial {}'.format(i))
        f.close()

if __name__=='__main__':
    main()

