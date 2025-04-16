import json
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm
from PIL import Image

from crop_images import crop_center
from inference import TaggedBeeClassifierConvNet, class_labels

OUTPUT_WIDTH = OUTPUT_HEIGHT = 50
TARGET = Path("/home/niklas/Documents/dev/uni/bees/bee-data/tmp")
PATH_TO_ALL_ZIPS = Path("/home/niklas/Documents/dev/uni/bees/bee-data/zipped")
TAGGED_DANCE_DIR = "tagged-dances"
UNTAGGED_DANCE_DIR = "untagged-dances"
TAGGED = "tagged"


def main():
    classifier = TaggedBeeClassifierConvNet("output/model.pth")

    # path_to_all_zips = Path("/mnt/trove/wdd/wdd_output_2024/cam0/2024/")
    for path_to_zip in tqdm(PATH_TO_ALL_ZIPS.rglob("*")):
        if not str(path_to_zip).endswith(".zip"):
            continue
        daily_target = TARGET / path_to_zip.name.replace(".zip", "")
        with ZipFile(path_to_zip) as zip_file:
            data = process_daily_data(zip_file, daily_target, classifier)
            df = pd.DataFrame(data)
            df.to_csv(daily_target / "data.csv", index=False)


def process_daily_data(
    zip_file: ZipFile, target_dir: Path, classifier: TaggedBeeClassifierConvNet
):
    tagged_target_dir = target_dir / TAGGED_DANCE_DIR
    untagged_target_dir = target_dir / UNTAGGED_DANCE_DIR
    day_dance_ids = []
    waggle_ids = []
    predictions = []
    confidences = []
    count = 1
    video_filenames = list(
        filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
    )

    for video_filename in tqdm(video_filenames):
        # Find matching metadata file
        metadata_filename = video_filename.replace("frames.apng", "waggle.json")
        with zip_file.open(metadata_filename) as metadata_file:
            json_data = json.load(metadata_file)
            # We only care about waggles
            if json_data["predicted_class_label"] != "waggle":
                continue
            waggle_ids.append(json_data["waggle_id"])
        with zip_file.open(video_filename) as video_file:
            with Image.open(video_file) as image:
                cropped_image = crop_center(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                prediction, confidence = classifier.classify_single_image(cropped_image)
                predictions.append(prediction)
                confidences.append(confidence)
        day_dance_id = f"{count:04d}"
        day_dance_ids.append(day_dance_id)
        # Save video file
        zip_file.getinfo(video_filename).filename = day_dance_id + ".apng"
        if class_labels[prediction] == TAGGED:
            tagged_target_dir.mkdir(parents=True, exist_ok=True)
            zip_file.extract(video_filename, tagged_target_dir)
        else:
            untagged_target_dir.mkdir(parents=True, exist_ok=True)
            zip_file.extract(video_filename, untagged_target_dir)
        count += 1

    data = {
        "day_dance_id": np.array(day_dance_ids),
        "waggle_id": np.array(waggle_ids),
        "category": np.array([predictions[i] for i, _ in enumerate(predictions)]),
        "category_label": np.array(
            [class_labels[predictions[i]] for i, _ in enumerate(predictions)]
        ),
        "confidence": np.array(confidences),
        "corrected_category": np.empty_like(day_dance_ids),
        "corrected_category_label": np.empty_like(day_dance_ids),
    }
    return data


if __name__ == "__main__":
    main()
