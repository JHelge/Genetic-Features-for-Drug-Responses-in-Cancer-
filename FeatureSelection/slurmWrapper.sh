#!/bin/bash -l

#####################
# job-array IC50_FEATURES_ONLY #
#####################

#SBATCH --job-name=IC50_FEATURES_ONLY

## 1530 jobs will run in this array at the same time
#SBATCH --array=100-149
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de


source /prj/ml-ident-canc/CLA/cla_env/bin/activate
# Basiswert und Schrittweite für den Job-Index festlegen
basis=1003
schrittweite=10

# Berechnen Sie den eigentlichen Job-Index basierend auf dem SLURM_ARRAY_TASK_ID
job_index=$(($basis + $SLURM_ARRAY_TASK_ID * $schrittweite))


echo "In Bash-Script starte Job mit Index $job_index"


python3 /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/IC50_FEATURES_ONLY/GeneExpRun/aBioInf100.py $job_index


exit 0

