import imageio.v3 as iio
from pathlib import Path


def copy_frames(root, destination):
    root = Path(root)
    for path in root.rglob("*"):
        if path.name == "frames.apng":
            source = path.parent / path.name
            destination = Path(destination) / str(source).replace(
                str(root) + "/", ""
            ).replace("/", "-")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())


def create_cropped_image(file, destination_dir, x, w, y, h):
    image = iio.imread(file, index=0)
    cropped_image = image[x : x + w, y : y + h]
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    filename = file.name.replace("-frames.apng", ".png")
    iio.imwrite(Path(destination_dir) / filename, cropped_image)


if __name__ == "__main__":
    data_dir = "/home/niklas/Documents/dev/uni/bees/bee-data/"
    frames_dir = "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/frames"
    # copy_frames(
    #     data_dir,
    #     frames_dir,
    # )
    frame_size = (250, 250)
    crop_size = (75, 75)
    x = frame_size[0] // 2 - crop_size[0] // 2
    y = frame_size[1] // 2 - crop_size[1] // 2
    cropped_image_dir = (
        "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/"
        + str(crop_size[0])
        + "x"
        + str(crop_size[1])
    )
    for file in Path(frames_dir).iterdir():
        create_cropped_image(file, cropped_image_dir, x, crop_size[0], y, crop_size[1])
