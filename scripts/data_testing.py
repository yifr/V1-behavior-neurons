import os
import h5py
from scipy.io import loadmat
from behavenet import get_user_dir

def test_cell_type_subsamples(session='1', animal='MD0ST5', lab='dipoppa', expt='full_trial',
                              sample='sst85_sample_0', cell_id=3):
    data_dir = get_user_dir('data')
    path = os.path.join(data_dir, lab, expt, animal, session, 'data.hdf5')
    if not os.path.exists(path):
        print(path, ' does not exist.')
        return

    # Get cell types (Need to change this for different mice)
    info_cells = loadmat('neural_dir/info_cells_MD0ST5_2018-04-04.mat')['info_cells']
    cell_type_idxs = info_cells[0][0][0][0]    # 0 = excitatory, 3 = inhibitory
    good_cell_idxs = info_cells[0][0][1][0]

    # Label all "bad" cells as -1
    cell_type_idxs = [cell_type_idxs[i] if good_cell_idxs[i] == 1 else -1 for i in range(len(good_cell_idxs))]
    def select_idxs(arr, key):
        return [i for i in range(len(arr)) if arr[i] == key]

    idxs = select_idxs(cell_type_idxs, cell_id)

    data = h5py.File(path, 'r', libver='latest', swmr=True)
    cell_types = data['samples']['cell_types']
    session = cell_types[sample]

    # Correct number of indexes
    assert len(session[:]) == len(idxs)
    # No repeats
    assert len(set(session[:])) == len(idxs)

    data.close()

    print('Cell Type Subsample: Success')

def test_subsamples(session='1', animal='MD0ST5', lab='dipoppa', expt='full_trial',
                              sample='sample_100_t0', cell_id=3):
    data_dir = get_user_dir('data')
    path = os.path.join(data_dir, lab, expt, animal, session, 'data.hdf5')
    if not os.path.exists(path):
        print(path, ' does not exist.')
        return

    data = h5py.File(path, 'r', libver='latest', swmr=True)
    subsamples = data['samples']['subsamples']
    session = subsamples[sample]

    # No duplicates
    assert len(set(session[:])) == len(session[:])
    print('Subsample Test: Success')
    data.close()

test_subsamples()
test_cell_type_subsamples()
