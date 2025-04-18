import datetime
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from ffmpeg import FFmpeg
from PIL import Image
from tqdm import tqdm

from crop_images import crop_center
from inference import TaggedBeeClassifierConvNet, class_labels

IMAGE_SIZE = 50
ZIPS_PATH = Path("/mnt/trove/wdd/wdd_output_2024/cam0/2024")
TARGET = Path("/home/niklas/processed-bee-data")
TAGGED_DANCE_DIR = "tagged-dances"
UNTAGGED_DANCE_DIR = "untagged-dances"
TAGGED = "tagged"
FRAMEDIR_WDD = Path("/mnt/trove/wdd/wdd_output_2024/fullframes")


def main():
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    for zip_path in tqdm(list(ZIPS_PATH.rglob("*"))):
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
            dance_types = []

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
                    # We only care about waggles, so filter the rest out. Also,
                    # the model thinks the bright pixels of the wooden frame on
                    # the comb are tags, so we ignore those detections.
                if json_data["predicted_class_label"] != "waggle" or is_wood_in_frame(
                    json_data
                ):
                    continue
                with zip_file.open(video_filename) as video_file:
                    with Image.open(video_file) as image:
                        cropped_image = crop_center(image, IMAGE_SIZE, IMAGE_SIZE)
                prediction, confidence = classifier.classify_single_image(cropped_image)
                day_dance_id = f"{count:04d}"
                day_dance_ids.append(day_dance_id)
                waggle_ids.append(json_data["waggle_id"])
                predictions.append(prediction)
                confidences.append(confidence)
                dance_types.append(json_data["predicted_class_label"])
                # Save video file
                # Because we don't want to keep the nested directory structure
                # which the files within the zip file are in, we assign a new
                # name to the filename attribute of a video file, i.e.
                # "12/44/8/frames.apng", -> "0001.apng". This leads to a flat
                # directory structure.
                zip_file.getinfo(video_filename).filename = day_dance_id + ".apng"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_file.extract(video_filename, tmp_dir)
                    input = Path(tmp_dir) / (day_dance_id + ".apng")
                    if class_labels[prediction] == TAGGED:
                        output = tagged_target_dir / (day_dance_id + ".mp4")
                    else:
                        output = untagged_target_dir / (day_dance_id + ".mp4")
                    encode_video(input, output)

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
                "dance_type": np.array(dance_types),
                "corrected_dance_type": np.empty_like(day_dance_ids),
            }

            # Save to csv file
            df = pd.DataFrame(data)
            df.to_csv(daily_target / "data.csv", index=False)


def is_wood_in_frame(json_data):
    """
    Estimates whether the cropped image of the corresponding dance shows a part
    of the wooden frame on the comb based on the position of the dance.
    """
    df_markers_wdd = pd.read_csv(FRAMEDIR_WDD / "df_markers.csv")
    wdd_markers = get_marker_coordinates_by_timestamp(
        detection_timestamp=json_data["timestamp_begin"],
        df_markers=df_markers_wdd,
    )
    if wdd_markers is None:
        return False

    # Values were determined from a single wdd image.
    wood_offset_x = 100
    wood_offset_top = 80
    wood_offset_bot = 220

    # These borders span an area on the comb that is within the wooden frame.
    border_left = max(
        [wdd_markers[0][0] + wood_offset_x, wdd_markers[2][0] + wood_offset_x]
    )
    border_top = max(
        [wdd_markers[0][1] + wood_offset_top, wdd_markers[1][1] + wood_offset_top]
    )
    border_right = min(
        [wdd_markers[1][0] - wood_offset_x, wdd_markers[3][0] - wood_offset_x]
    )
    border_bot = min(
        [wdd_markers[2][1] - wood_offset_bot, wdd_markers[3][1] - wood_offset_bot]
    )

    center_x, center_y = json_data["roi_center"]
    # Correct the -125 offset of the roi coordinates in the metadata
    correction_offset = 125
    center_x += correction_offset
    center_y += correction_offset

    # Apply offset to account for size of cropped image that the model works on
    frame_offset = IMAGE_SIZE // 2
    return (
        center_x - frame_offset <= border_left
        or center_x + frame_offset >= border_right
        or center_y - frame_offset <= border_top
        or center_y + frame_offset >= border_bot
    )


def get_marker_coordinates_by_timestamp(detection_timestamp, df_markers):
    """Gives the most recent marker coordinates from before the dance detection."""
    df_markers["timestamp"] = pd.to_datetime(df_markers["timestamp"])
    marker_timestamps = sorted(df_markers["timestamp"].unique())
    timestamp_to_show = None
    for marker_timestamp in marker_timestamps:
        if marker_timestamp <= datetime.datetime.fromisoformat(detection_timestamp):
            timestamp_to_show = marker_timestamp
    if timestamp_to_show is None:
        return None
    dfsel = df_markers.loc[df_markers["timestamp"] == timestamp_to_show]
    marker_coords = [(row["x"], row["y"]) for _, row in dfsel.iterrows()]
    return marker_coords


def encode_video(input: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(input))
        .output(str(output), {"codec:v": "libx264"}, crf=18, pix_fmt="yuv420p")
    )
    ffmpeg.execute()


if __name__ == "__main__":
    main()
