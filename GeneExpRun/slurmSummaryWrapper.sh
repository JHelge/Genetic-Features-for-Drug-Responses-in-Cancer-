#!/bin/bash -l

#####################
# job-array IC50_FEATURES_ONLY #
#####################

#SBATCH --job-name=IC50_FEATURES_ONLY

## 1530 jobs will run in this array at the same time
#SBATCH --array=0-1499
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.schlueter@uni-bielefeld.de


source /prj/ml-ident-canc/CLA/cla_env/bin/activate
# Basiswert und Schrittweite für den Job-Index festlegen
basis=1003
schrittweite=1

# Berechnen Sie den eigentlichen Job-Index basierend auf dem SLURM_ARRAY_TASK_ID
job_index=$(($basis + $SLURM_ARRAY_TASK_ID * $schrittweite))

# Corrected if condition
if (( $job_index % 10 != 3 )); then
    echo "In Bash script starting job with index $job_index"
    # Uncomment the following line if the script should execute a Python script
    python3 /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/IC50_FEATURES_ONLY/GeneExpRun/summaryMulti.py $job_index
else 
    echo "Already done: $job_index"
fi


exit 0

