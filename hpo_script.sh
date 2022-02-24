# !/bin/sh

if grep -q "sweep" "sweep_status.txt"; then
    echo  Sweep Found.
    fi

python tools/train.py -f exps/yolox_custom.py -c pretrained_weights/yolox_s.pth -d 1 -b 32 --gpus 1 --fp16 -expn helmet_hpo

# Use while loop to keep executing the training until the sweep is complete.