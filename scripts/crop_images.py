import argparse
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm

from daily_data_processing import IMAGE_SIZE
from image_cropping import crop_center


def main():
    parser = init_argparse()
    args = parser.parse_args()

    create_cropped_images(
        zips_dir=args.zipped_wdd_data_dir,
        target_dir=args.cropped_image_output_dir,
        output_width=IMAGE_SIZE,
        output_height=IMAGE_SIZE,
    )


def create_cropped_images(
    zips_dir: Path | str,
    target_dir: Path | str,
    output_width: int,
    output_height: int,
):
    """Extracts and crops first frame from WDD video snippets."""
    zips_dir = Path(zips_dir)
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

                        # Flatten output directory structure
                        filename = video_file.name.replace(
                            "/frames.apng", ".png"
                        ).replace("/", "_")

                        target_path = current_target_dir / filename
                        cropped_image.save(target_path)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s <zipped_wdd_data_dir> <cropped_image_output_dir>",
        description=("Extracts and crops the first frame from APNG video snippets"),
    )
    parser.add_argument(
        "zipped_wdd_data_dir",
        type=Path,
        help="Path to directory containing zip archives of .APNG video snippets",
    )
    parser.add_argument(
        "cropped_image_output_dir",
        type=Path,
        help="Path to directory containing zip archives of .APNG video snippets",
    )
    return parser


if __name__ == "__main__":
    main()
