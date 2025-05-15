import csv
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

from inference import TaggedBeeClassifierConvNet, class_labels, dict_to_csv

ZIPPED_WDD_PATH = Path("/mnt/trove/wdd/wdd_output_2024/cam0/")


def get_samples(cropped_images_path: Path, k=100):
    paths = list(cropped_images_path.rglob("*"))
    samples = random.sample(paths, k)
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    predictions = []
    for sample in samples:
        with Image.open(sample) as image:
            prediction, confidence = classifier.classify_single_image(image)
            predictions.append(class_labels[prediction[0]])
    data = {"sample_path": samples, "category_label": predictions}
    dict_to_csv(data, "output/samples.csv")


def extract_samples(samples_csv_path: Path, output_path: Path):
    with open(samples_csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sample_path = row["sample_path"]
            # "/home/niklas/bee-data/cropped/2024-08-29/10_47_14.png"
            # "/mnt/trove/wdd/wdd_output_2024/cam0/2024/8/2024-08-29.zip"
            year_pattern = r"20\d{2}"
            after_year_lookahead = r"(?=-\d{2}-\d{2}/\d{2}_\d{2}_\d{2}.png)"
            # TODO: add lookbehinds and lookaheads
            # month_pattern = r"-(\d{2})-"
            month_pattern = r"-0([1-9])|(1[0-2])-"
            date_pattern = r"20\d{2}-\d{2}-\d{2}"
            time_num_pattern = r"\d+_\d+_\d+.png"
            year = re.search(year_pattern, sample_path).group()
            month = re.search(month_pattern, sample_path).group(1)
            date = re.search(date_pattern, sample_path).group()
            time_number = re.search(time_num_pattern, sample_path).group()
            sample_filename = re.sub("_", "/", time_number).replace(
                ".png", "/frames.apng"
            )

            zip_path = ZIPPED_WDD_PATH.joinpath(year, month, date + ".zip")
            archive = zipfile.ZipFile(zip_path)
            for file in archive.namelist():
                if file == sample_filename:
                    destination_dir = output_path / date
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    archive.extract(file, destination_dir)
                    continue


if __name__ == "__main__":
    # get_samples(Path("/home/niklas/bee-data/cropped/"), 100)
    samples_csv_path = Path.cwd() / "output" / "samples.csv"
    destination_path = Path.cwd() / "output" / "samples"
    extract_samples(samples_csv_path, destination_path)
