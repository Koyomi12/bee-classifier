import torch

from datasets import test_dataloader
from model import TaggedBeeClassificationModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaggedBeeClassificationModel()
    model.load_state_dict(torch.load("output/model.pth", weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the {total} test images: {accuracy:.2f}%")
