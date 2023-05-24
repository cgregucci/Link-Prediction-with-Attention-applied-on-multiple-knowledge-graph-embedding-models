#!/bin/bash
source set_env.sh
source hyp_kg_env/bin/activate
export CUDA_VISIBLE_DEVICES=3
python run.py \
          --dataset NELL-995-h100 \
            --model SEA \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --neg_sample_size 250 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --batch_size 500 \
            --dtype single \
            --double_neg 
            
            