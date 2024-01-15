export PYTORCH_ENABLE_MPS_FALLBACK=1

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
