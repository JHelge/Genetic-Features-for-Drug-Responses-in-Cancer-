#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=example

# 1530 jobs will run in this array at the same time
#SBATCH --array=1-1530


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de


source /prj/ml-ident-canc/CLA/cla_env/bin/activate

echo "$SLURM_ARRAY_TASK_ID"

python3 /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/GeneExpRun/aBioInf100.py $SLURM_ARRAY_TASK_ID
exit 0

