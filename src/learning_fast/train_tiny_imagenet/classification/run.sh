#!/bin/bash

python train.py --data-path ~/Projects/data/imagenet64 --model='mobilenet_v3_large' --workers 4\ 
                --epochs 600 --opt rmsprop --batch-size 128 --lr 0.064 --wd 0.00001\ 
                --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2

# python train_quantization.py --data-path ~/Projects/data/imagenet64 --model='mobilenet_v3_large' \
#     --wd 0.00001 --lr 0.001 --workers=8