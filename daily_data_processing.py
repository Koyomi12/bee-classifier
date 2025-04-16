import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from crop_images import crop_center
from inference import TaggedBeeClassifierConvNet, class_labels

OUTPUT_WIDTH = OUTPUT_HEIGHT = 50
ZIPS_PATH = Path("/mnt/trove/wdd/wdd_output_2024/cam0/2024")
TARGET = Path("/home/niklas/processed-bee-data")
TAGGED_DANCE_DIR = "tagged-dances"
UNTAGGED_DANCE_DIR = "untagged-dances"
TAGGED = "tagged"


def main():
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    for zip_path in ZIPS_PATH.rglob("*"):
        if not zip_path.suffix == ".zip":
            continue
        daily_target = TARGET / zip_path.stem
        # Ignore days that were already processed.
        if daily_target.exists():
            continue
        with ZipFile(zip_path) as zip_file:
            day_dance_ids = []
            waggle_ids = []
            predictions = []
            confidences = []

            tagged_target_dir = daily_target / TAGGED_DANCE_DIR
            untagged_target_dir = daily_target / UNTAGGED_DANCE_DIR
            video_filenames = list(
                filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
            )
            for count, video_filename in enumerate(tqdm(video_filenames), start=1):
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
                        prediction, confidence = classifier.classify_single_image(
                            cropped_image
                        )
                        predictions.append(prediction)
                        confidences.append(confidence)
                day_dance_id = f"{count:04d}"
                day_dance_ids.append(day_dance_id)
                # Save video file
                # Because we don't want to keep the nested directory structure
                # which the files within the zip file are in, we assign a new
                # name to the filename attribute of a video file, i.e.
                # "12/44/8/frames.apng", -> "0001.apng". This leads to a flat
                # directory structure.
                zip_file.getinfo(video_filename).filename = day_dance_id + ".apng"
                if class_labels[prediction] == TAGGED:
                    extract_file(video_filename, zip_file, tagged_target_dir)
                else:
                    extract_file(video_filename, zip_file, untagged_target_dir)

            data = {
                "day_dance_id": np.array(day_dance_ids),
                "waggle_id": np.array(waggle_ids),
                "category": np.array(
                    [predictions[i] for i, _ in enumerate(predictions)]
                ),
                "category_label": np.array(
                    [class_labels[predictions[i]] for i, _ in enumerate(predictions)]
                ),
                "confidence": np.array(confidences),
                "corrected_category": np.empty_like(day_dance_ids),
                "corrected_category_label": np.empty_like(day_dance_ids),
            }

            # Save to csv file
            df = pd.DataFrame(data)
            df.to_csv(daily_target / "data.csv", index=False)


def extract_file(filename: str, zip_file: ZipFile, target_dir: Path):
    """Extracts a file from a zip file into a directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_file.extract(filename, target_dir)


if __name__ == "__main__":
    main()
