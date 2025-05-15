import sklearn
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
        y_true = []
        y_pred = []
        y_score = []
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions, confidences, tagged_probabilities = (
                model.postprocess_predictions(outputs, numpy=False)
            )
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predictions.tolist()
            y_score += confidences.tolist()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the {total} test images: {accuracy:.2f}%")

        # Reverse labels because we want 0 = tagged to be positive
        y_true = [0 if y == 1 else 1 for y in y_true]
        y_pred = [0 if y == 1 else 1 for y in y_pred]

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        print(f"tp: {tp}")
        print(f"tn: {tn}")
        print(f"fp: {fp}")
        print(f"fn: {fn}")
        f1_score = sklearn.metrics.f1_score(y_true, y_pred)
        print(f"F1 score: {f1_score}")
        roc_auc_score = sklearn.metrics.roc_auc_score(
            y_true, tagged_probabilities.cpu()
        )
        print(f"ROC AUC score: {roc_auc_score}")
        print(sklearn.metrics.classification_report(y_true, y_pred, digits=2))
