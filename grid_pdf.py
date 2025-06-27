import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm

from image_cropping import crop_center
from inference import TaggedBeeClassifierConvNet

TAGGED = "tagged"
UNTAGGED = "untagged"
class_labels = (TAGGED, UNTAGGED)


# TODO: probably move this to experiments, but keep this for the Koyomi owned repositories as it was declared to be in this repo in the bachelor's thesis. Remove it for the biorobotics repo.
def main():
    cropped_image_dir = Path("/home/niklas/bee-data/cropped/")
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    data = run_classifier_on_all(classifier, cropped_image_dir)
    data = data_to_dataframe(data)
    generate_plots_pdfs(data)


def data_to_dataframe(data: dict[str, any]):
    df = pd.DataFrame(data)
    return df


def generate_plots_pdfs(df: pd.DataFrame):
    """
    Creates a multi-page PDF containing a grid of cropped images sorted by tag
    status and the model's confidence.
    """
    df = df.sort_values(by=["class", "confidence"])
    count = df.shape[0]
    plots_per_page = 50
    num_pages = int(np.ceil(count / plots_per_page))
    rows, cols = (10, 5)
    unique_dates = df["date"].unique()
    df_dict = {elem: pd.DataFrame() for elem in unique_dates}
    for key in df_dict.keys():
        df_dict[key] = df[:][df["date"] == key]
        filename = f"output/visualizations/daily-cropped-image-grids/{key}.pdf"
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


def run_classifier_one_by_one(
    classifier: TaggedBeeClassifierConvNet, cropped_image_dir: Path
):
    """
    Applies the classifier to all images in a given directory one by one.
    """
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
    """
    Applies the classifier to all images in a given directory containing
    cropped images. Uses batches.
    """
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


# benefit: don't have to save cropped images
# def old_generate_plots(
#     classifier: TaggedBeeClassifierConvNet,
#     zips_dir: Path | str,
#     output_width: int,
#     output_height: int,
# ):
#     predictions = []
#     confidences = []
#     paths = []
#     for path_to_zip in tqdm(zips_dir.rglob("*")):
#         with ZipFile(path_to_zip) as zip_file:
#             video_filenames = list(
#                 filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
#             )
#             count = len(video_filenames)
#             plots_per_page = 50
#             num_pages = int(np.ceil(count / plots_per_page))
#             rows, cols = (10, 5)
#             pdf_pages = PdfPages(f"output/{path_to_zip.stem}.pdf")
#             last_tagged_idx = 0
#             last_untagged_idx = count - 1
#             current_idx = None
#             figures = [
#                 plt.figure(i, figsize=(8.27, 11.69), dpi=100) for i in range(num_pages)
#             ]
#             # for filename in tqdm(video_filenames):
#             for filename in video_filenames:
#                 with zip_file.open(filename) as video_file:
#                     with Image.open(video_file) as image:
#                         cropped_image = crop_center(image, output_width, output_height)
#                         prediction, confidence = classifier.classify_single_image(
#                             cropped_image
#                         )
#                         predictions.append(prediction)
#                         confidences.append(confidence)
#                         paths.append(Path(path_to_zip.stem) / filename)
#
#                         if class_labels[prediction[0]] == "tagged":
#                             current_idx = last_tagged_idx
#                             last_tagged_idx += 1
#                         else:
#                             current_idx = last_untagged_idx
#                             last_untagged_idx -= 1
#                         current_page = current_idx // plots_per_page
#                         plt.figure(current_page)
#                         plt.subplot2grid(
#                             (rows, cols),
#                             (
#                                 (current_idx // cols) % rows,
#                                 current_idx % cols,
#                             ),
#                         )
#                         plt.imshow(cropped_image, cmap="gray")
#                         plt.axis("off")
#             for figure in figures:
#                 figure.savefig(pdf_pages, format="pdf")
#             pdf_pages.close()
#             plt.close("all")
#     data = {
#         "class": np.array([predictions[i][0] for i, _ in enumerate(predictions)]),
#         "class_label": np.array(
#             [class_labels[predictions[i][0]] for i, _ in enumerate(predictions)]
#         ),
#         "confidence": np.array([confidences[i][0] for i, _ in enumerate(confidences)]),
#         "cropped_image_path": np.array(paths),
#         "date": np.array(
#             [re.search(r"20\d{2}-\d{2}-\d{2}", str(path)).group(0) for path in paths]
#         ),
#     }
#     return data


if __name__ == "__main__":
    main()
