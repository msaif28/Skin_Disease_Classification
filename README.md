# Skin_Disease_Classification




pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu120

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
