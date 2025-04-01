import torch

print("当前设备：", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
