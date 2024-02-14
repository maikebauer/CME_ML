#!/bin/bash
#  
#SBATCH -J Train_CNN                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2
#SBATCH --gres=gpu:1                   #or --gres=gpu:1 if you only want to use half a node
#SBATCH --output=output.txt
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=maikebauer42@gmail.com
export PYTORCH_ENABLE_MPS_FALLBACK=1

module purge

nvidia-smi
module load miniconda3
eval "$(conda shell.bash hook)"

conda activate cme_ml

mode="Train" #Test

declare -a epoch=(0 5 10 15 20 30 40 50 60 70 80 90 100 120 140 160 180 200 250 300 350 400) #0 5 10 15 20 30 40 50 60 70 80 90 100 120 140 160 180 200 250 300 350 400

model_run="model_torch.py"
eval_run="evaluation.py"
eval_folder="run_14022024_140713_model_resnet34"

backbone="resnet34"

if [ "$mode" = "Train" ]
then
    python "$model_run" "$backbone"

elif [ "$mode" = "Test" ]
then
    for i in "${epoch[@]}"
    do
       python "$eval_run" "$i" "$eval_folder"
    done
fi


