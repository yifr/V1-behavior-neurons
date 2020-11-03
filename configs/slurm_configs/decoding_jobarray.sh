#!/bin/bash
#SBATCH --job-name=latent_decoding_plaw   # Job name
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jf3338@columbia.edu   # Where to send mail
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --account=zi
#SBATCH --nodes 1
#SBATCH  --cpus 1
#SBATCH --mem-per-cpu 2G
#SBATCH --array=0-898%100

DIR=$HOME/behavenet
IDX=$(($SLURM_ARRAY_TASK_ID + 901))
echo "Launching Experiment ", $IDX
python behavenet/fitting/decoder_grid_search.py --data_config $DIR/slurm_logs/temp_configs/data_config-exp$IDX.json --model_config $DIR/slurm_logs/temp_configs/empty_config.json --compute_config $DIR/slurm_logs/temp_configs/empty_config.json --training_config $DIR/slurm_logs/temp_configs/empty_config.json

