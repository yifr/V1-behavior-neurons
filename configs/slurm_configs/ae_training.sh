#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=ae_latent_search   # Job name
#SBATCH --mail-type=END,FAIL,BEGIN          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jf3338@columbia.edu   # Where to send mail
#SBATCH --time=18:00:00               # Time limit hrs:min:sec
#SBATCH --nodes 1
#SBATCH  --gres=gpu:3
#SBATCH --constraint=k80
#SBATCH -c 1
#SBATCH --mem-per-cpu=2G

