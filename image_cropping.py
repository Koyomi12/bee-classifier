from PIL.Image import Image as PILImage


def crop_center(image: PILImage, output_width: int, output_height: int):
    image_width, image_height = image.size
    left = (image_width - output_width) // 2
    top = (image_height - output_height) // 2
    return image.crop((left, top, left + output_width, top + output_height))
