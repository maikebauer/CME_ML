#!/bin/bash
#
#  usage: sbatch ./gpu_test.scrpt          
#SBATCH -J Training_Summed                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2
#SBATCH --gres=gpu:1                   #or --gres=gpu:1 if you only want to use half a node
#SBATCH --output=output.txt
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=maikebauer42@gmail.com
module purge

nvidia-smi
module load miniconda3
eval "$(conda shell.bash hook)"

conda activate cme_ml

mode="Train" #Test
epoch=60

model_run="model_summed_aug.py"
eval_run="evaluation_summed_aug.py"

if [ "$mode" = "Train" ]
then
    python "$model_run"

elif [ "$mode" = "Test" ]
then
    python "$eval_run" "$epoch"
fi

