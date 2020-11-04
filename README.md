# V1-behavior-neurons

This repo contains code for analysing behavioral and neural data. The two primary 
components are the [behavenet](https://github.com/ebatty/behavenet) package for explicit 
and latent behavioral analysis, and the notebooks folder, which contains jupyter notebooks 
to visualize the results.

## Data formatting
Behavenet requires data be formatted in hdf5 files to fit its models. For a detailed overview
of how these hdf5 files are expected to be organized, you can refer to the 
[behavenet documentation](https://behavenet.readthedocs.io/en/develop/source/data_structure.html).


For our purposes, we'll need the following data/files:
- a .mj2 video of mouse behavior (for these experiments we expect a video of a mouse's eye)
- a file containing the corresponding neural data (neurons x time) 
- two files that contain indexes to line up video and neural data (2 x num_trials, where batch_idxs[0][i] indicates
  start index of a trial and batch_idxs[1][i] indicates the end index)
- behavioral data - a .npy file extracted from facemap (for the time being - this may switch to 
  deeplabcut extracted labels soon)
- a file that contains information about individual cell types, and good/bad cells 

The most relevant piece of code is scripts/data_to_hdf5.py script, which formats the data into the 
required format. This code expects data to be organized in the following directory structure:
    - root_folder
    -   lab
    -       date
    -           cell_info_for_mouse
