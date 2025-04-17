import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from cv2.typing import MatLike
from PIL import Image
from tqdm import tqdm

from crop_images import crop_center
from daily_data_processing import IMAGE_SIZE
from evaluate_pixel_threshold import THRESHOLD_VALUE, load_samples_csv
from inference import TAGGED, UNTAGGED

CSV_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-data/samples/2024-08-30/data.csv"
)
ZIP_PATH = Path("/home/niklas/Documents/dev/uni/bees/bee-data/zipped/2024-08-30.zip")


def main():
    asdf = []
    with ZipFile(ZIP_PATH) as zip_file:
        img_seq_filenames = list(
            filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
        )
        for img_seq_filename in tqdm(img_seq_filenames):
            # Find matching metadata file
            metadata_filename = img_seq_filename.replace("frames.apng", "waggle.json")
            with zip_file.open(metadata_filename) as metadata_file:
                json_data = json.load(metadata_file)
                waggle_id = str(json_data["waggle_id"])
                asdf.append(dict(waggle_id=waggle_id, video_seq_file=img_seq_filename))

        confusion_matrix = dict(
            true_positive=0, false_negative=0, false_positive=0, true_negative=0
        )
        samples = load_samples_csv(CSV_PATH)
        for sample in samples:
            match = next(
                (item for item in asdf if item["waggle_id"] == sample["waggle_id"]),
                None,
            )
            if match is None:
                print(f"{sample['waggle_id']}")
                continue

            with zip_file.open(match["video_seq_file"]) as video_file:
                with Image.open(video_file) as image:
                    cropped_image = crop_center(image, IMAGE_SIZE, IMAGE_SIZE)
            img = np.array(cropped_image)
            label = label_image(img)
            if (
                label == TAGGED
                and sample["category_label"] == TAGGED
                and sample["corrected_category_label"] == ""
            ):
                confusion_matrix["true_positive"] += 1
            elif (
                label == TAGGED
                and sample["category_label"] == UNTAGGED
                and sample["corrected_category_label"] == TAGGED
            ):
                confusion_matrix["true_positive"] += 1
            elif (
                label == UNTAGGED
                and sample["category_label"] == UNTAGGED
                and sample["corrected_category_label"] == ""
            ):
                confusion_matrix["true_negative"] += 1
            elif (
                label == UNTAGGED
                and sample["category_label"] == TAGGED
                and sample["corrected_category_label"] == UNTAGGED
            ):
                confusion_matrix["true_negative"] += 1
            elif (
                label == TAGGED
                and sample["category_label"] == UNTAGGED
                and sample["corrected_category_label"] == ""
            ):
                confusion_matrix["false_positive"] += 1
            elif (
                label == TAGGED
                and sample["category_label"] == TAGGED
                and sample["corrected_category_label"] == UNTAGGED
            ):
                confusion_matrix["false_positive"] += 1
            elif (
                label == UNTAGGED
                and sample["category_label"] == TAGGED
                and sample["corrected_category_label"] == ""
            ):
                confusion_matrix["false_negative"] += 1
            elif (
                label == UNTAGGED
                and sample["category_label"] == UNTAGGED
                and sample["corrected_category_label"] == TAGGED
            ):
                confusion_matrix["false_negative"] += 1
        print(confusion_matrix)


def label_image(image: MatLike):
    label = UNTAGGED
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > THRESHOLD_VALUE:
                label = TAGGED
    return label


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


if __name__ == "__main__":
    main()
