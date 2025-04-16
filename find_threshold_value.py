from collections import namedtuple
from pathlib import Path

import cv2

from evaluate_pixel_threshold import Evaluation, evaluate_test_images

TRAIN_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/train/"
)
VALIDATION_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/validation/"
)
TEST_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/test/"
)

ThresholdEvaluation = namedtuple(
    "ThresholdEvaluation", ("threshold_value",) + Evaluation._fields
)


def main():
    threshold_value, total, accuracy = find_best_threshold_value(
        TRAIN_PATH, VALIDATION_PATH
    )
    print(
        f"Accuracy of the model on the {total} training and validation images using a threshold value of {threshold_value}: {accuracy:.2f}%"
    )
    threshold_value = 129
    total, accuracy = evaluate_test_images(TEST_PATH, threshold_value)
    print(f"Accuracy of the model on the {total} test images: {accuracy:.2f}%")


def find_best_threshold_value(train_image_dir: Path, validation_image_dir: Path):
    train_image_paths = list(train_image_dir.rglob("*.png"))
    validation_image_paths = list(validation_image_dir.rglob("*.png"))
    image_paths = train_image_paths + validation_image_paths
    mistakes = []
    for value in range(256):
        mistake_count = 0
        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            tagged = False
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i, j] > value:
                        tagged = True
            if tagged and "/tagged/" not in str(image_path):
                mistake_count += 1
                # print(f"FALSE POSITIVE - {image_path.name}")
            elif not tagged and "/untagged/" not in str(image_path):
                # print(f"FALSE NEGATIVE - {image_path.name}")
                mistake_count += 1
        mistakes.append((value, mistake_count))

    lowest = mistakes[0]
    for value, mistake_count in mistakes:
        if mistake_count < lowest[1]:
            lowest = (value, mistake_count)
    total = len(image_paths)
    correct = total - lowest[1]
    accuracy = 100 * correct / total
    return ThresholdEvaluation(
        threshold_value=lowest[0], total=total, accuracy=accuracy
    )


if __name__ == "__main__":
    main()
