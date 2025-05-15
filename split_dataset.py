import random
from pathlib import Path


def main():
    """Splits dataset into train, validation and test sets."""
    image_dir = Path.joinpath(Path.cwd(), "data", "cropped", "50x50_1")
    tagged_image_dir = image_dir / "tagged"
    untagged_image_dir = image_dir / "untagged"

    tagged_images = list(tagged_image_dir.iterdir())
    untagged_images = list(untagged_image_dir.iterdir())

    assert len(tagged_images) == 397 == len(untagged_images)
    num_train_images_per_class = 277
    num_test_images_per_class = 60

    # move train images
    train_tagged = random.sample(tagged_images, k=num_train_images_per_class)
    train_untagged = random.sample(untagged_images, k=num_train_images_per_class)

    train_tagged_image_dir = Path.joinpath(image_dir, "train", "tagged")
    train_tagged_image_dir.mkdir(parents=True, exist_ok=True)

    train_untagged_image_dir = Path.joinpath(image_dir, "train", "untagged")
    train_untagged_image_dir.mkdir(parents=True, exist_ok=True)

    for file in train_tagged:
        move_file(file, train_tagged_image_dir / file.name)

    for file in train_untagged:
        move_file(file, train_untagged_image_dir / file.name)

    # move test images
    tagged_images = list(tagged_image_dir.iterdir())
    untagged_images = list(untagged_image_dir.iterdir())

    test_tagged = random.sample(tagged_images, k=num_test_images_per_class)
    test_untagged = random.sample(untagged_images, k=num_test_images_per_class)

    test_tagged_image_dir = Path.joinpath(image_dir, "test", "tagged")
    test_tagged_image_dir.mkdir(parents=True, exist_ok=True)

    test_untagged_image_dir = Path.joinpath(image_dir, "test", "untagged")
    test_untagged_image_dir.mkdir(parents=True, exist_ok=True)

    for file in test_tagged:
        move_file(file, test_tagged_image_dir / file.name)

    for file in test_untagged:
        move_file(file, test_untagged_image_dir / file.name)

    # move validation images
    (image_dir / "validation").mkdir(parents=True, exist_ok=True)

    tagged_image_dir.rename(Path.joinpath(image_dir, "validation", "tagged"))
    untagged_image_dir.rename(Path.joinpath(image_dir, "validation", "untagged"))


def move_file(source, destination):
    if destination.exists():
        print(f"{destination} already exists")
    else:
        source.replace(destination)


if __name__ == "__main__":
    main()
