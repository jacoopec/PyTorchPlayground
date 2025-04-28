# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BATCH_SIZE = 32

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
# print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=False,
    transform=ToTensor()
)

# image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(label);

train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")