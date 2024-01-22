#!/bin/bash

# Sequential training of models
echo "Training AlexNet..."
nohup python train.py --model_name AlexNet --gpu 0 > /var/tmp/logs/AlexNet.log 2>&1

echo "Training VGG..."
nohup python train.py --model_name VGG --gpu 0 > /var/tmp/logs/VGG.log 2>&1

echo "Training ResNet..."
nohup python train.py --model_name ResNet --gpu 0 > /var/tmp/logs/ResNet.log 2>&1

echo "Training EfficientNet..."
nohup python train.py --model_name EfficientNet --gpu 0 > /var/tmp/logs/EfficientNet.log 2>&1

echo "All models have been trained."

