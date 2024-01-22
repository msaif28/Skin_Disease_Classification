# Skin_Disease_Classification




pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu120

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

nohup python train.py --model_name AlexNet --gpu 0 > /var/tmp/AlexNet.log 2>&1 &
nohup python train.py --model_name VGG --gpu 1 > /var/tmp/VGG.log 2>&1 &
nohup python train.py --model_name ResNet --gpu 2 > /var/tmp/ResNet.log 2>&1 & 
nohup python train.py --model_name EfficientNet --gpu 0 > /var/tmp/EfficientNet.log 2>&1 & 
nohup python train.py --model_name ViT --gpu 1 > /var/tmp/ViT.log 2>&1 & 


