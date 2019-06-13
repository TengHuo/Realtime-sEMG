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

python -m emg.train lstm \
                    --suffix adam-compare \
                    --sub_folder 3layer-0.5dp \
                    --gesture_num 8 \
                    --epoch 200 \
                    --train_batch_size 256 \
                    --valid_batch_size 1024 \
                    --lr 0.001 \
                    --lr_step 70

#python -m emg.train --model lstm \
#                    --suffix regularization \
#                    --sub_folder e-4 \
#                    --gesture_num 8 \
#                    --lr 0.001 \
#                    --epoch 200 \
#                    --train_batch_size 256 \
#                    --valid_batch_size 1024 \
#                    --stop_patience 12 \
#                    --log_interval 100
#
#python -m emg.train --model lstm \
#                    --suffix regularization \
#                    --sub_folder without \
#                    --gesture_num 8 \
#                    --lr 0.001 \
#                    --epoch 200 \
#                    --train_batch_size 256 \
#                    --valid_batch_size 1024 \
#                    --stop_patience 12 \
#                    --log_interval 100

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

#python -m emg.train mlp \
#                  --suffix test-args \
#                  --sub_folder test \
#                  --gesture_num 8 \
#                  --epoch 4 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --lr 0.001

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

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --lr 0.001 \
#                  --epoch 60 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-10 \
#                  --gesture_num 10 \
#                  --lr 0.001 \
#                  --epoch 1 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-12 \
#                  --gesture_num 12 \
#                  --lr 0.001 \
#                  --epoch 60 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-14 \
#                  --gesture_num 14 \
#                  --lr 0.001 \
#                  --epoch 1 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-16 \
#                  --gesture_num 16 \
#                  --lr 0.001 \
#                  --epoch 60 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-18 \
#                  --gesture_num 18 \
#                  --lr 0.001 \
#                  --epoch 60 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100

#python -m emg.train --model c3d \
#                  --suffix compare-gesture \
#                  --sub_folder gesture-20 \
#                  --gesture_num 20 \
#                  --lr 0.001 \
#                  --epoch 60 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --stop_patience 12 \
#                  --log_interval 100


