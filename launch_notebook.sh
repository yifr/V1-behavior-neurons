#!/bin/bash

#SBATCH --job-name=behavenet_notebook
#SBATCH --time=3:00:00
#SBATCH -J notebook
#SBATCH --account=zi

export XDG_RUNTIME_DIR=""

jupyter notebook --no-browser --ip "*" \
      --notebook-dir $HOME
