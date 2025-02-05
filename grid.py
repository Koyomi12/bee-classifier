import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import test_dataloader
from model import Model
from utils import classes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.load_state_dict(torch.load("output/model.pth", weights_only=True))
    model.to(device)
    model.eval()

    for step, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, 1)
        print(len(images))

        figure = plt.figure(figsize=(8, 8), layout="constrained")
        cols, rows = 11, 11
        for i, (image, label, prediction) in enumerate(
            zip(images, labels, predictions)
        ):
            image = image.cpu()
            image = image / 2 + 0.5  # unnormalize
            npimg = image.numpy()

            figure.add_subplot(rows, cols, i + 1)
            plt.title(
                f"{classes[label]} | {classes[prediction]}",
                fontsize=8,
                color="red" if label != prediction else "black",
            )
            plt.axis("off")
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
