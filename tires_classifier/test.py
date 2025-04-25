import torch
import torchvision

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("CUDA is available. Using GPU.")
else:
  device = torch.device("cpu")
  print("CUDA is not available. Using CPU.")

PATH_TO_IMAGES = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset_split"

dataset.class_to_idx

dummy_input = torch.randn(1, 3, 224, 224).to(device)

dummy_input.shape
