#!/bin/bash
#$ -cwd
#$ -M j.schlueter@uni-bielefeld.de
##$ -m abe
#$ -b n
#$ -o /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/IC50_REGRESSION/GeneExpRun/tmp.out
#$ -e /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/IC50_REGRESSION/GeneExpRun/tmp.err
#$ -i /dev/null 
##$ -o /dev/null
##$ -e /dev/null
##$ -l h_vmem=1G
##$ -l h_rt=00:01:00
##$ -l s_rt=00:01:00
#$ -P fair_share
#$ -l idle=1 
##$ -l nodes=1
#$ -t 1003-1102:10

import /vol/codine-8.3/default/common/settings.sh
source /prj/ml-ident-canc/CLA/cla_env/bin/activate

echo "$SGE_TASK_ID"

python3 /prj/ml-ident-canc/CLA/EnsembleFeatureSelection/IC50_REGRESSION/GeneExpRun/aBioInf100.py $SGE_TASK_ID
exit 0

