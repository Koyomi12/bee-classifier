from pathlib import Path
from zipfile import ZipFile

import tqdm
from PIL import Image
from PIL.Image import Image as PILImage


def main():
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


if __name__ == "__main__":
    main()
