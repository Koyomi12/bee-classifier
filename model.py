import torch
import torch.nn as nn
import torch.nn.functional as F


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 2 * 2, 16)
        self.fc2 = nn.Linear(16, 2)
        # self.fc1 = nn.Linear(16 * 2 * 2, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = Model()
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 2)
#         nn.ReLU = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(nn.ReLU(self.conv1(x)))
#         x = self.pool(nn.ReLU(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = nn.ReLU(self.fc1(x))
#         x = self.fc2(x)
#         return x


# https://www.doczamora.com/cats-vs-dogs-binary-classifier-with-pytorch-cnn
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         # convolutional layers (3,16,32)
#         self.conv1 = nn.Conv2d(
#             in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1
#         )
#         self.conv3 = nn.Conv2d(
#             in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
#         )
#
#         # conected layers
#         self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500)
#         self.fc2 = nn.Linear(in_features=500, out_features=50)
#         self.fc3 = nn.Linear(in_features=50, out_features=2)
#
#     def forward(self, X):
#         X = F.relu(self.conv1(X))
#         X = F.max_pool2d(X, 2)
#
#         X = F.relu(self.conv2(X))
#         X = F.max_pool2d(X, 2)
#
#         X = F.relu(self.conv3(X))
#         X = F.max_pool2d(X, 2)
#
#         X = X.view(X.shape[0], -1)
#         X = F.relu(self.fc1(X))
#         X = F.relu(self.fc2(X))
#         X = self.fc3(X)
#
#         return X


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.conv4 = nn.Conv2d(128, 256, 5)
#
#         self.fc1 = nn.Linear(256, 2)
#
#         self.pool = nn.MaxPool2d(2, 2)
#
#     def forward(self, x):
#         x = self.pool(nn.ReLU(self.conv1(x)))
#         x = self.pool(nn.ReLU(self.conv2(x)))
#         x = self.pool(nn.ReLU(self.conv3(x)))
#         x = self.pool(nn.ReLU(self.conv4(x)))
#         bs, _, _, _ = x.shape
#         x = self.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
#         x = self.fc1(x)
#         return x
