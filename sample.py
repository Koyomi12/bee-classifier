import random
from pathlib import Path

from PIL import Image

from inference import TaggedBeeClassifierConvNet, class_labels, dict_to_csv

if __name__ == "__main__":
    paths = list(Path("/home/niklas/bee-data/cropped/").rglob("*"))
    samples = random.sample(paths, 100)
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    predictions = []
    for sample in samples:
        with Image.open(sample) as image:
            prediction, confidence = classifier.classify_single_image(image)
            predictions.append(class_labels[prediction[0]])
    data = {"sample_path": samples, "class": predictions}
    dict_to_csv(data, "output/samples.csv")
