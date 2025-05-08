import random
from pathlib import Path


def move_file(source, destination):
    if destination.exists():
        print(f"{destination} already exists")
    else:
        source.replace(destination)


if __name__ == "__main__":
    image_dir = Path.joinpath(Path.cwd(), "data", "cropped", "50x50_1")
    marked_image_dir = image_dir / "marked"
    unmarked_image_dir = image_dir / "unmarked"

    marked_images = list(marked_image_dir.iterdir())
    unmarked_images = list(unmarked_image_dir.iterdir())

    assert len(marked_images) == 397 == len(unmarked_images)
    num_train_images_per_class = 277
    num_test_images_per_class = 60
    num_validation_images_per_class = 60

    # move train images
    train_marked = random.sample(marked_images, k=num_train_images_per_class)
    train_unmarked = random.sample(unmarked_images, k=num_train_images_per_class)

    train_marked_image_dir = Path.joinpath(image_dir, "train", "marked")
    train_marked_image_dir.mkdir(parents=True, exist_ok=True)

    train_unmarked_image_dir = Path.joinpath(image_dir, "train", "unmarked")
    train_unmarked_image_dir.mkdir(parents=True, exist_ok=True)

    for file in train_marked:
        move_file(file, train_marked_image_dir / file.name)

    for file in train_unmarked:
        move_file(file, train_unmarked_image_dir / file.name)

    # move test images
    marked_images = list(marked_image_dir.iterdir())
    unmarked_images = list(unmarked_image_dir.iterdir())

    test_marked = random.sample(marked_images, k=num_test_images_per_class)
    test_unmarked = random.sample(unmarked_images, k=num_test_images_per_class)

    test_marked_image_dir = Path.joinpath(image_dir, "test", "marked")
    test_marked_image_dir.mkdir(parents=True, exist_ok=True)

    test_unmarked_image_dir = Path.joinpath(image_dir, "test", "unmarked")
    test_unmarked_image_dir.mkdir(parents=True, exist_ok=True)

    for file in test_marked:
        move_file(file, test_marked_image_dir / file.name)

    for file in test_unmarked:
        move_file(file, test_unmarked_image_dir / file.name)

    # move validation images
    (image_dir / "validation").mkdir(parents=True, exist_ok=True)

    marked_image_dir.rename(Path.joinpath(image_dir, "validation", "marked"))
    unmarked_image_dir.rename(Path.joinpath(image_dir, "validation", "unmarked"))
