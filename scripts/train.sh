#! /bin/bash

## Train models for classifying 8-20 gestures in CapgMyo dataset
#for((i=20; i <= 20; i++));
#do
#    python -m emg.app --model cnn \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 60 \
#                      --train_batch_size 256 \
#                      --valid_batch_size 1024 \
#                      --stop_patience 5 \
#                      --log_interval 100
#done

# lstm model
#for((i=8; i <= 20; i++));
#do
#    python -m emg.app --model lstm \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 60 \
#                      --train_batch_size 256 \
#                      --valid_batch_size 1024 \
#                      --stop_patience 12 \
#                      --log_interval 100
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
#for((i=8; i <= 20; i++));
#do
#    python -m emg.app --model mlp \
#                      --gesture_num $i \
#                      --lr 0.001 \
#                      --epoch 100 \
#                      --train_batch_size 512 \
#                      --valid_batch_size 2048 \
#                      --stop_patience 7 \
#                      --log_interval 100
#done

python -m emg.train mlp \
                 --suffix test-arg \
                 --sub_folder test8 \
                 --gesture_num 8 \
                 --epoch 4 \
                 --train_batch_size 512 \
                 --valid_batch_size 2048 \
                 --lr 0.001


# c3d model
#python -m emg.app --model c3d \
#                  --suffix default \
#                  --sub_folder default \
#                  --gesture_num 8 \
#                  --lr 0.001 \
#                  --epoch 1 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 7 \
#                  --log_interval 100

