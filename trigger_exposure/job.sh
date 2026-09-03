#!/bin/bash

# SLURM options:

#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --output=log/%j.log   # Standard output and error log

#SBATCH --partition=htc               # Partition choice (htc by default)

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=2000                    # Memory in MB per default
#SBATCH --time=1-00:00:00             # Max time limit = 7 days
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

thisdir= $(pwd)
log=${thisdir}/log
mkdir -p ${log}

filter=fir
trig_params=${thisdir}/dict_trig_params_fir.csv
data_dir=/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_20170331_220000_RUN1_CD_DC2-CoreasDC2_1rc4_AN_filenumber

python ${thisdir}/judge_trigger_with_ADCtrace_like_experiment.py ${filter} ${trig_params} ${data_dir}
