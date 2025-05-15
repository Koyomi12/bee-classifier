import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator

from datasets import train_dataloader, validation_dataloader
from hyperparameters import epochs, learning_rate
from model import TaggedBeeClassificationModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaggedBeeClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_losses = train(train_dataloader, model, criterion, optimizer, device)
        train_loss_values.append(sum(train_losses) / len(train_losses))
        validation_loss, validation_accuracy = validate(
            validation_dataloader, model, criterion, device
        )
        validation_loss_values.append(validation_loss)
        validation_accuracy_values.append(validation_accuracy)
    print("Done!")
    save_plots(train_loss_values, validation_loss_values, validation_accuracy_values)
    torch.save(model.state_dict(), "output/model.pth")


def train(dataloader, model, loss_fn, optimizer, device):
    epoch_losses = []
    model.train()
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Compute prediction error
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backpropagation
        loss.backward()
        epoch_losses.append(loss.item() * images.shape[0])
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            size = len(dataloader.dataset)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return epoch_losses


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            validation_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    validation_loss /= num_batches
    accuracy = 100 * correct / size
    print(
        f"Validation Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {validation_loss:>8f} \n"
    )
    return validation_loss, accuracy


def save_plots(train_loss_values, validation_loss_values, validation_accuracy_values):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(np.linspace(1, epochs, epochs).astype(int), train_loss_values)
    ax1.set_title("Training Loss")
    ax2.plot(np.linspace(1, epochs, epochs).astype(int), validation_loss_values)
    ax2.set_title("Validation Loss")
    ax3.plot(np.linspace(1, epochs, epochs).astype(int), validation_accuracy_values)
    ax3.set_title("Validation Accuracy")
    ax1.xaxis.set_major_locator(MaxNLocator(epochs, integer=True))
    plt.xlabel("epochs")
    plt.tight_layout()
    plt.savefig("output/visualizations/training_graphs.png")


if __name__ == "__main__":
    main()
