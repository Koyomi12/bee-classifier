import csv
from collections import namedtuple
from pathlib import Path

import cv2

from inference import TAGGED, UNTAGGED

THRESHOLD_VALUE = 129  # value that works best on training dataset
SAMPLE_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-classifier/output/samples.csv"
)

Evaluation = namedtuple("Evaluation", ["total", "accuracy"])


def main():
    samples = load_samples_csv(SAMPLE_PATH)
    for sample in samples:
        label = label_image(Path(sample["sample_path"]))
        sample[f"pixel_threshold_label_at_{THRESHOLD_VALUE}"] = label
        dictlist_to_csv(samples, "output/samples.csv")
    confusion_matrix = evaluate_samples(samples)
    print(confusion_matrix)


def load_samples_csv(path: Path):
    with open(path) as csvfile:
        return list(csv.DictReader(csvfile))


def label_image(image_path: Path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = UNTAGGED
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > THRESHOLD_VALUE:
                label = TAGGED
    return label


def dictlist_to_csv(data: list[dict[str, str]], filename: Path):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(data[0].keys())
        writer.writerows([row.values() for row in data])


def evaluate_samples(sample_data):
    confusion_matrix = dict(
        true_positive=0, false_negative=0, false_positive=0, true_negative=0
    )
    for sample in sample_data:
        label = sample[f"pixel_threshold_label_at_{THRESHOLD_VALUE}"]
        if (
            label == TAGGED
            and sample["manual_evaluation_based_on_first_frame"] == TAGGED
        ):
            confusion_matrix["true_positive"] += 1
        elif (
            label == UNTAGGED
            and sample["manual_evaluation_based_on_first_frame"] == TAGGED
        ):
            confusion_matrix["false_negative"] += 1
        elif (
            label == TAGGED
            and sample["manual_evaluation_based_on_first_frame"] == UNTAGGED
        ):
            confusion_matrix["false_positive"] += 1
        elif (
            label == UNTAGGED
            and sample["manual_evaluation_based_on_first_frame"] == UNTAGGED
        ):
            confusion_matrix["true_negative"] += 1
    return confusion_matrix


def evaluate_test_images(image_dir: Path, threshold_value: int):
    mistake_count = 0
    image_paths = list(image_dir.rglob("*.png"))
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        tagged = False
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] > threshold_value:
                    tagged = True
        if (
            tagged
            and "/tagged/" not in str(image_path)
            or not tagged
            and "/untagged/" not in str(image_path)
        ):
            mistake_count += 1
    total = len(image_paths)
    correct = total - mistake_count
    accuracy = 100 * correct / total
    return Evaluation(total=total, accuracy=accuracy)


if __name__ == "__main__":
    main()
