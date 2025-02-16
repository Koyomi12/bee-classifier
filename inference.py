import csv
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from PIL.Image import Image as PILImage
from tqdm import tqdm

from model import TaggedBeeClassificationModel

class_labels = ("marked", "unmarked")


class TaggedBeeClassifierConvNet:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TaggedBeeClassificationModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def classify_single_image(self, image):
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
            prediction, confidence = self.model.postprocess_predictions(output)
            return prediction, confidence

    def classify_images_from_directory(self, image_dir: Path | str, batch_size):
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


def create_cropped_images(
    zips_dir: Path | str,
    target_dir: Path | str,
    output_width: int,
    output_height: int,
):
    for path_to_zip in tqdm(zips_dir.rglob("*")):
        if not str(path_to_zip).endswith(".zip"):
            continue
        zip_filename = path_to_zip.name.replace(".zip", "")
        current_target_dir = Path(target_dir) / zip_filename
        current_target_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(path_to_zip) as zip_file:
            video_filenames = filter(
                lambda filename: filename.endswith(".apng"), zip_file.namelist()
            )
            for filename in video_filenames:
                with zip_file.open(filename) as video_file:
                    with Image.open(video_file) as image:
                        cropped_image = crop_center(image, output_width, output_height)
                        filename = video_file.name.replace(
                            "/frames.apng", ".png"
                        ).replace("/", "_")
                        target_path = current_target_dir / filename
                        cropped_image.save(target_path)


def crop_center(image: PILImage, output_width: int, output_height: int):
    image_width, image_height = image.size
    left = (image_width - output_width) // 2
    top = (image_height - output_height) // 2
    return image.crop((left, top, left + output_width, top + output_height))


def generate_plots_pdfs(df: pd.DataFrame):
    df = df.sort_values(by=["class", "confidence"])
    count = df.shape[0]
    plots_per_page = 50
    num_pages = int(np.ceil(count / plots_per_page))
    rows, cols = (10, 5)
    unique_dates = df["date"].unique()
    df_dict = {elem: pd.DataFrame() for elem in unique_dates}
    for key in df_dict.keys():
        df_dict[key] = df[:][df["date"] == key]
        filename = f"output/visualizations/{key}.pdf"
        pdf_pages = PdfPages(filename)
        day_df = df_dict[key]
        figures = [
            plt.figure(i, figsize=(8.27, 11.69), dpi=100) for i in range(num_pages)
        ]
        last_tagged_idx = 0
        last_untagged_idx = count - 1
        current_idx = None
        for _, row in day_df.iterrows():
            if row["class_label"] == class_labels[0]:
                current_idx = last_tagged_idx
                last_tagged_idx += 1
            else:
                current_idx = last_untagged_idx
                last_untagged_idx -= 1
            current_page = current_idx // plots_per_page
            # print(
            #     i,
            #     current_idx,
            #     current_page,
            #     (current_idx // grid_size[1]) % grid_size[0],
            #     current_idx % grid_size[1],
            # )
            plt.figure(current_page)
            plt.subplot2grid(
                (rows, cols),
                (
                    (current_idx // cols) % rows,
                    current_idx % cols,
                ),
            )
            with Image.open(row["cropped_image_path"]) as cropped_image:
                plt.imshow(cropped_image, cmap="gray")
                plt.axis("off")
        for figure in figures:
            figure.savefig(pdf_pages, format="pdf")
        pdf_pages.close()
        plt.close("all")


# benefit: don't have to save cropped images
def old_generate_plots(
    classifier: TaggedBeeClassifierConvNet,
    zips_dir: Path | str,
    output_width: int,
    output_height: int,
):
    predictions = []
    confidences = []
    paths = []
    for path_to_zip in tqdm(zips_dir.rglob("*")):
        with ZipFile(path_to_zip) as zip_file:
            video_filenames = list(
                filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
            )
            count = len(video_filenames)
            plots_per_page = 50
            num_pages = int(np.ceil(count / plots_per_page))
            rows, cols = (10, 5)
            pdf_pages = PdfPages(f"output/{path_to_zip.stem}.pdf")
            last_tagged_idx = 0
            last_untagged_idx = count - 1
            current_idx = None
            figures = [
                plt.figure(i, figsize=(8.27, 11.69), dpi=100) for i in range(num_pages)
            ]
            # for filename in tqdm(video_filenames):
            for filename in video_filenames:
                with zip_file.open(filename) as video_file:
                    with Image.open(video_file) as image:
                        cropped_image = crop_center(image, output_width, output_height)
                        prediction, confidence = classifier.classify_single_image(
                            cropped_image
                        )
                        predictions.append(prediction)
                        confidences.append(confidence)
                        paths.append(Path(path_to_zip.stem) / filename)

                        if class_labels[prediction[0]] == "marked":
                            current_idx = last_tagged_idx
                            last_tagged_idx += 1
                        else:
                            current_idx = last_untagged_idx
                            last_untagged_idx -= 1
                        current_page = current_idx // plots_per_page
                        # print(
                        #     i,
                        #     current_idx,
                        #     current_page,
                        #     (current_idx // grid_size[1]) % grid_size[0],
                        #     current_idx % grid_size[1],
                        # )
                        plt.figure(current_page)
                        plt.subplot2grid(
                            (rows, cols),
                            (
                                (current_idx // cols) % rows,
                                current_idx % cols,
                            ),
                        )
                        plt.imshow(cropped_image, cmap="gray")
                        plt.axis("off")
                        # plt.tight_layout()
            for figure in figures:
                figure.savefig(pdf_pages, format="pdf")
            pdf_pages.close()
            plt.close("all")
    data = {
        "class": np.array([predictions[i][0] for i, _ in enumerate(predictions)]),
        "class_label": np.array(
            [class_labels[predictions[i][0]] for i, _ in enumerate(predictions)]
        ),
        "confidence": np.array([confidences[i][0] for i, _ in enumerate(confidences)]),
        "cropped_image_path": np.array(paths),
        "date": np.array(
            [re.search(r"20\d{2}-\d{2}-\d{2}", str(path)).group(0) for path in paths]
        ),
    }
    return data


def run_classifier_one_by_one(
    classifier: TaggedBeeClassifierConvNet, cropped_image_dir: Path
):
    # TODO: Could run the entire pipeline here without saving the cropped images. At the end, we take the labels and probabilities and images to create a grid and save that file.
    # np.append is slower than appending on a list
    predictions = []
    confidences = []
    paths: list[Path] = []
    for path in tqdm(cropped_image_dir.rglob("*")):
        if path.suffix == ".png":
            with Image.open(path) as image:
                prediction, confidence = classifier.classify_single_image(image)
                predictions.append(prediction)
                confidences.append(confidence)
                paths.append(path)
    data = {
        "class": np.array([predictions[i][0] for i, _ in enumerate(predictions)]),
        "class_label": np.array(
            [class_labels[predictions[i][0]] for i, _ in enumerate(predictions)]
        ),
        "confidence": np.array([confidences[i][0] for i, _ in enumerate(confidences)]),
        "cropped_image_path": np.array(paths),
        "date": np.array(
            [re.search(r"20\d{2}-\d{2}-\d{2}", str(path)).group(0) for path in paths]
        ),
    }
    return data


def run_classifier_on_all(
    classifier: TaggedBeeClassifierConvNet, cropped_image_dir: Path
):
    predictions, confidences, paths = classifier.classify_images_from_directory(
        cropped_image_dir, 128
    )
    data = {
        "class": predictions,
        "class_label": np.array([class_labels[pred] for pred in predictions]),
        "confidence": confidences,
        "cropped_image_path": paths,
        "date": np.array(
            [re.search(r"20\d{2}-\d{2}-\d{2}", str(path)).group(0) for path in paths]
        ),
    }
    return data


def data_to_dataframe(data: dict[str, any]):
    df = pd.DataFrame(data)
    return df


# doesn't work with numpy array
# def save_json(data: dict[str, any], path: Path):
#     result = json.dumps(data)
#     json.dump(result, path)


def df_to_json(df: pd.DataFrame, filename: Path):
    df.to_json(filename)


def dict_to_csv(data, filename: Path):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))


def df_to_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path)


if __name__ == "__main__":
    # 2024/8/2024-08-29.zip
    zipped_dir = Path("/mnt/trove/wdd/wdd_output_2024/cam0/2024/")
    if not zipped_dir.is_dir():
        zipped_dir = Path("/home/niklas/Documents/dev/uni/bees/bee-data/zipped/")

    cropped_image_dir = Path("/home/niklas/bee-data/cropped/")
    if not cropped_image_dir.is_dir():
        cropped_image_dir = Path(
            "/home/niklas/Documents/dev/uni/bees/bee-data/cropped/"
        )

    create_cropped_images(
        zipped_dir,
        cropped_image_dir,
        50,
        50,
    )

    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    # data = run_classifier_one_by_one(classifier, cropped_image_dir)
    # dict_to_csv(data, Path("output/data_one_by_one.csv"))

    data = run_classifier_on_all(classifier, cropped_image_dir)
    # dict_to_csv(data, Path("output/data.csv"))
    data = data_to_dataframe(data)
    generate_plots_pdfs(data)

    # does it all
    # old_generate_plots(
    #     classifier, Path("/home/niklas/Documents/dev/uni/bees/bee-data/zipped/"), 50, 50
    # )
