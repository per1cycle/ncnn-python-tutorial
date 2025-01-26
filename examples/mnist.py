import ncnn

# load mnist dataset from torch.
import torch
import torchvision

training_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    )

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    )

# show the image using matplotlib
import matplotlib.pyplot as plt

# class SimpleMnist(ncnn.Layer):
#     def __init__(self):
#         super(SimpleMnist, self).__init__()
#         self.conv1 = ncnn.Convolution(1, 32, 5, 1, 2)
#         self.conv2 = ncnn.Convolution(32, 64, 5, 1, 2)
#         self.fc1 = ncnn.Linear(7*7*64, 1024)
#         self.fc2 = ncnn.Linear(1024, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = ncnn.ReLU(x)
#         x = ncnn.MaxPool(x, 2, 2)
#         x = self.conv2(x)
#         x = ncnn.ReLU(x)
#         x = ncnn.MaxPool(x, 2, 2)
#         x = ncnn.Reshape(x, 1, 1, -1)
#         x = self.fc1(x)
#         x = ncnn.ReLU(x)
#         x = self.fc2(x)
#         return x
    
model = SimpleMnist()

def train_step():

    pass 