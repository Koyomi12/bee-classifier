from pathlib import Path


def create_images(root, destination):
    root = Path(root)
    for path in root.rglob("*"):
        if path.name == "frames.apng":
            source = path.parent / path.name
            destination = Path(destination) / str(source).replace(
                str(root) + "/", ""
            ).replace("/", "-")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())


if __name__ == "__main__":
    data_dir = "/home/niklas/Documents/dev/uni/bees/bee-data/"
    frames_dir = "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/frames"
    create_images(
        data_dir,
        frames_dir,
    )
