#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=zscoring   # Job name
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jf3338@columbia.edu   # Where to send mail
#SBATCH --time=00:45:00               # Time limit hrs:min:sec
#SBATCH --nodes 1
#SBATCH  --cpus 1
#SBATCH --mem-per-cpu 3G

python normalize_data.py
