#!/bin/bash
#SBATCH --job-name=label_decoder   # Job name
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jf3338@columbia.edu   # Where to send mail
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --account=zi
#SBATCH --nodes 1
#SBATCH  --cpus 1
#SBATCH --mem-per-cpu 2G
