#!/bin/bash

rm -rf /var/tmp/logs/
mkdir -p /var/tmp/logs/

echo "Starting training for ViT"
taskset -c 33-49 python vit_training.py --model_name ViT --gpu 0 > /var/tmp/logs/ViT.log 2>&1 &
PID1=$!
echo "ViT training initiated with PID: $PID1"

echo "Starting training for VGG"
taskset -c 9-16 python train.py --model_name VGG --gpu 0 > /var/tmp/logs/VGG.log 2>&1 &
PID2=$!
echo "VGG training initiated with PID: $PID2"

echo "Starting training for ResNet"
taskset -c 17-24 python train.py --model_name ResNet --gpu 0 > /var/tmp/logs/ResNet.log 2>&1 &
PID3=$!
echo "ResNet training initiated with PID: $PID3"

echo "Starting training for EfficientNet"
taskset -c 25-32 python train.py --model_name EfficientNet --gpu 0 > /var/tmp/logs/EfficientNet.log 2>&1 &
PID4=$!
echo "EfficientNet training initiated with PID: $PID4"

echo "Starting training for AlexNet"
taskset -c 0-8 python train.py --model_name AlexNet --gpu 0 > /var/tmp/logs/AlexNet.log 2>&1 &
PID5=$!
echo "AlexNet training initiated with PID: $PID5"

echo "Waiting for all training jobs to complete..."
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "All training jobs completed"
