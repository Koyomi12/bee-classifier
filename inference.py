from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

from model import TaggedBeeClassificationModel

TAGGED = "tagged"
UNTAGGED = "untagged"
class_labels = (TAGGED, UNTAGGED)


class TaggedBeeClassifierConvNet:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TaggedBeeClassificationModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def classify_single_image(self, image):
        """Classifies an image."""
        transform = transforms.Compose(
            [
                transforms.Grayscale(1),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        with torch.inference_mode():
            output = self.model(image_tensor)
            prediction, confidence, _ = self.model.postprocess_predictions(output)
            return prediction[0], confidence[0]

    def classify_images_from_directory(self, image_dir: Path | str, batch_size):
        """Classifies images in given directory."""
        transform = transforms.Compose(
            [
                transforms.Grayscale(1),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        dataset = datasets.ImageFolder(image_dir, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=True)
        paths = np.array([path for path, _ in dataset.imgs])
        with torch.inference_mode():
            all_predictions = np.array([], dtype=int)
            all_confidences = np.array([])
            for inputs, _ in tqdm(dataloader):
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                predictions, confidences = self.model.postprocess_predictions(outputs)
                all_predictions = np.concatenate(
                    (all_predictions, predictions), dtype=int
                )
                all_confidences = np.concatenate((all_confidences, confidences))
            return all_predictions, all_confidences, paths
