#! /bin/bash

## Train models for classifying 8-20 gestures in CapgMyo dataset
for((i=20; i <= 20; i++));
do
    python -m emg.app --model cnn \
                      --gesture_num $i \
                      --lr 0.001 \
                      --epoch 60 \
                      --train_batch_size 256 \
                      --valid_batch_size 1024 \
                      --stop_patience 5 \
                      --log_interval 100
done

# lstm model with bn
#for((i=8; i <= 20; i++));
#do
#    python -m emg.app --model lstm \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 60 \
#                      --train_batch_size 256 \
#                      --val_batch_size 1024 \
#                      --stop_patience 12 \
#                      --log_interval 100 \
#                      --load_model False
#done

## seq2seq model with bn
#for((i=8; i <= 20; i++));
#do
#    python -m emg.app --model seq2seq \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 60 \
#                      --train_batch_size 256 \
#                      --val_batch_size 1024 \
#                      --stop_patience 7 \
#                      --log_interval 100 \
#                      --load_model False
#done

# mlp model, mainly used for test code
#for((i=8; i <= 8; i++));
#do
#    python -m emg.app --model mlp \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 60 \
#                      --train_batch_size 256 \
#                      --val_batch_size 1024 \
#                      --stop_patience 7 \
#                      --log_interval 100 \
#                      --load_model False
#done
