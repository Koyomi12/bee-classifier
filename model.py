import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
class TaggedBeeClassificationModel(nn.Module):
    def __init__(self):
        super(TaggedBeeClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=3)
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

    @staticmethod
    def postprocess_predictions(outputs):
        probabilities = F.softmax(outputs, 1)
        predictions = torch.argmax(outputs, 1)
        confidences = probabilities[np.arange(probabilities.shape[0]), predictions]

        predictions = predictions.detach().cpu().numpy()
        confidences = confidences.detach().cpu().numpy()

        return predictions, confidences
