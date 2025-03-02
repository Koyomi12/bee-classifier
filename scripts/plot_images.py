from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid

if __name__ == "__main__":
    all_images_path = Path(
        "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_updated/close"
    )

    images = [cv2.imread(image) for image in all_images_path.glob("*.png")]

    # fig = plt.figure(figsize=(4, 4))
    # grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
    #
    # for ax, im in zip(grid, images):
    #     ax.imshow(im)
    #
    # plt.show()
    #
    # images[0].save(pdf_path)

    chunk_size = 50
    image_chunks = [
        images[i : i + chunk_size] for i in range(0, len(images), chunk_size)
    ]

    figs = []
    for chunk in image_chunks:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        grid = ImageGrid(fig, 111, nrows_ncols=(10, 5), axes_pad=0.1)
        for ax, im in zip(grid, chunk):
            ax.imshow(im)
            ax.axis("off")
        figs.append(fig)

    with PdfPages("bees.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close()
