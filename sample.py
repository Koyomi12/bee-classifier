import csv
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

from inference import TaggedBeeClassifierConvNet, class_labels

# TODO: turn this into argparse argument
ZIPPED_WDD_PATH = Path("/mnt/trove/wdd/wdd_output_2024/cam0/")


def get_samples(cropped_images_path: Path, k=100):
    """
    Randomly samples and classifies k cropped images, then writes the results
    to a samples.csv file.
    """
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
    """
    Reads the samples.csv file, finds the data in the zipped WDD archives
    and extracts the video snippets to the output path.
    """
    with open(samples_csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sample_path = row["sample_path"]

            year_pattern = r"20\d{2}"
            month_pattern = r"-0([1-9])|(1[0-2])-"
            date_pattern = r"20\d{2}-\d{2}-\d{2}"
            time_num_pattern = r"\d+_\d+_\d+.png"

            year_match = re.search(year_pattern, sample_path)
            month_match = re.search(month_pattern, sample_path)
            date_match = re.search(date_pattern, sample_path)
            time_number_match = re.search(time_num_pattern, sample_path)

            if (
                year_match is None
                or month_match is None
                or date_match is None
                or time_number_match is None
            ):
                raise ValueError(
                    f"Failed to extract date/time info from path: {sample_path}"
                )

            year = year_match.group()
            month = month_match.group(1)
            date = date_match.group()
            time_number = time_number_match.group()

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


def dict_to_csv(data, filename: Path | str) -> None:
    filename = Path(filename)
    with filename.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))


# TODO: move this into main function
if __name__ == "__main__":
    # TODO: uncomment this
    # get_samples(Path("/home/niklas/bee-data/cropped/"), 100)
    # TODO: make sure to check if file already exists and if it does, don't overwrite it
    samples_csv_path = Path.cwd() / "output" / "samples.csv"
    destination_path = Path.cwd() / "output" / "samples"
    extract_samples(samples_csv_path, destination_path)
