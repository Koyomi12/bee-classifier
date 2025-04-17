import datetime
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from converter import Converter
from asdf import Converter


def load_marker_csv():
    framedir_hd = Path(
        "/home/niklas/Documents/dev/uni/bees/bee-data/single_camera_frames/"
    )
    df_markers_hd = pd.read_csv(framedir_hd / "df_markers.csv")
    return df_markers_hd


# "2024-08-16T13:18:03.151660+00:00"
def get_marker_coordinates_by_timestamp(detection_timestamp, df_markers):
    df_markers["timestamp"] = pd.to_datetime(df_markers["timestamp"])
    timestamps = df_markers["timestamp"].unique()
    timestamp_to_show = None
    for timestamp in timestamps:
        if datetime.datetime.fromisoformat(detection_timestamp) >= timestamp:
            timestamp_to_show = timestamp
    if timestamp_to_show is None:
        print("No marker data for that time")
        sys.exit(0)
    dfsel = df_markers[df_markers["timestamp"] == timestamp_to_show]
    marker_coords = [(row["x"], row["y"]) for _, row in dfsel.iterrows()]
    return marker_coords


def print_marker_coords_for_timestampe():
    detection_coords_hd = (657, 2493)
    detection_timestamp = "20240831T134440.480245.377Z"

    # framedir_hd = Path("/mnt/trove/wdd/wdd_videos_2024/single_camera_frames/")
    # framedir_wdd = Path("/mnt/trove/wdd/wdd_output_2024/fullframes/")
    framedir_hd = Path(
        "/home/niklas/Documents/dev/uni/bees/bee-data/single_camera_frames/"
    )
    framedir_wdd = Path("/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/")

    df_markers_hd = pd.read_csv(framedir_hd / "df_markers.csv")
    df_markers_wdd = pd.read_csv(framedir_wdd / "df_markers.csv")

    df_markers_hd["timestamp"] = pd.to_datetime(df_markers_hd["timestamp"])
    df_markers_wdd["timestamp"] = pd.to_datetime(df_markers_wdd["timestamp"])

    timestamps = df_markers_wdd["timestamp"].unique()
    timestamp_to_show = None
    for timestamp in timestamps:
        if datetime.datetime.fromisoformat(detection_timestamp) >= timestamp:
            timestamp_to_show = timestamp
    if timestamp_to_show is None:
        print("No marker data for that time")
        sys.exit(0)
    dfsel = df_markers_wdd[df_markers_wdd["timestamp"] == timestamp_to_show]
    marker_coords_wdd = [(row["x"], row["y"]) for _, row in dfsel.iterrows()]

    timestamps = df_markers_hd["timestamp"].unique()
    timestamp_to_show = None
    for timestamp in timestamps:
        if datetime.datetime.fromisoformat(detection_timestamp) >= timestamp:
            timestamp_to_show = timestamp
    if timestamp_to_show is None:
        print("No marker data for that time")
        sys.exit(0)
    dfsel = df_markers_hd[df_markers_hd["timestamp"] == timestamp_to_show]
    marker_coords_hd = [(row["x"], row["y"]) for _, row in dfsel.iterrows()]
    print("hd: ", marker_coords_hd)
    print("wdd: ", marker_coords_wdd)


def draw_roi():
    # this needs to choose the correct row instead
    # filename = marker_df.iloc[0]["filename"]

    waggle_paths = [
        "/home/niklas/Documents/dev/uni/bees/bee-data/unzipped/2024-08-16/13/18/2/",
        "/home/niklas/Documents/dev/uni/bees/bee-data/unzipped/2024-08-16/14/10/16/",
        "/home/niklas/Documents/dev/uni/bees/bee-data/unzipped/2024-08-16/14/13/10/",
    ]
    image_paths = [
        "/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/cam0-2024-08-16T13_12_36.259721+00_00.png",
        "/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/cam0-2024-08-16T13_12_36.259721+00_00.png",
        "/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/cam0-2024-08-16T14_12_41.766871+00_00.png",
    ]
    _, ax = plt.subplots(len(waggle_paths), 2, gridspec_kw={"width_ratios": [1, 2]})
    for i, (waggle_path, image_path) in enumerate(zip(waggle_paths, image_paths)):
        image = mpimg.imread(image_path)
        metadata_path = Path(waggle_path) / "waggle.json"
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        apng_image_path = Path(waggle_path) / "frames.apng"
        apng_image = mpimg.imread(apng_image_path)
        # plt.figure(figsize=(10, 10))
        ax[i, 0].imshow(apng_image, cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(image, cmap="gray")

        roi_coordinates = metadata["roi_coordinates"]
        xs = [i for i in range(roi_coordinates[0][0], roi_coordinates[1][0] + 1)]
        ys = [i for i in range(roi_coordinates[0][1], roi_coordinates[1][1] + 1)]
        x, y = (
            xs + xs + [xs[0] for _ in ys] + [xs[-1] for _ in ys],
            [ys[0] for _ in xs] + [ys[-1] for _ in xs] + ys + ys,
        )
        ax[i, 1].scatter(x, y, c="red", s=0.1)
        ax[i, 1].annotate(
            "ROI from metadata",
            (xs[0], ys[0]),
            color="red",
            fontsize=12,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )
        ax[i, 1].scatter(
            [x_coord + 125 for x_coord in x],
            [y_coord + 125 for y_coord in y],
            c="green",
            s=0.1,
        )
        ax[i, 1].annotate(
            "ROI as it should be?",
            (xs[0] + 125, ys[0] + 125),
            color="green",
            fontsize=12,
            fontweight="bold",
            xytext=(50, 5),
            textcoords="offset points",
        )
        center_x, center_y = metadata["roi_center"]
        ax[i, 1].scatter(center_x, center_y, c="red", s=10)
        ax[i, 1].axis("off")
    plt.show()


def draw_points():
    framedir_hd = Path(
        "/home/niklas/Documents/dev/uni/bees/bee-data/single_camera_frames/"
    )
    framedir_wdd = Path("/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/")

    df_markers_hd = pd.read_csv(framedir_hd / "df_markers.csv")
    df_markers_wdd = pd.read_csv(framedir_wdd / "df_markers.csv")

    hd_markers = get_marker_coordinates_by_timestamp(
        "2024-09-02 12:02:39+00:00", df_markers_hd
    )
    wdd_markers = get_marker_coordinates_by_timestamp(
        "2024-09-02 12:02:36.710553+00:00", df_markers_wdd
    )
    hd_markers = [hd_markers[2], hd_markers[0], hd_markers[3], hd_markers[1]]
    # hd_markers[2] = wdd_markers[0]
    # hd_markers[0] = wdd_markers[1]
    # hd_markers[3] = wdd_markers[2]
    # hd_markers[1] = wdd_markers[3]
    converter = Converter(hd_markers, wdd_markers)

    image_path_hd = "/home/niklas/Documents/dev/uni/bees/bee-data/single_camera_frames/cam-1_20240902T120239.577384.924Z--20240902T120739.png"
    image_path_wdd = "/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/cam0-2024-09-02T12_02_36.710553+00_00.png"
    image_hd = mpimg.imread(image_path_hd)
    image_wdd = mpimg.imread(image_path_wdd)
    # rotated_image = np.rot90(image_hd, 3)
    _, ax = plt.subplots(1, 2)
    # ax[0].imshow(rotated_image, cmap="gray")
    ax[0].imshow(image_hd, cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(image_wdd, cmap="gray")
    ax[1].axis("off")
    # hd_x, hd_y = (2400, 0)
    # hd_x, hd_y = (2400, 3000)
    # hd_x, hd_y = (2500, 2900)
    # TODO: increasing x here means it moves down in the hd image -- I need to think about how to do this
    # hd_x, hd_y = (2600, 2900)
    hd_x, hd_y = (2150, 2750)
    # hd_x, hd_y = hd_markers[3]
    ax[0].scatter([hd_y], [hd_x], c="red", s=5)
    print(hd_markers)
    print(wdd_markers)
    wdd_x, wdd_y = converter.transform_coordinates(hd_y, hd_x)
    # wdd_x, wdd_y = wdd_markers[2]
    ax[1].scatter([wdd_x], [wdd_y], c="red", s=5)
    plt.show()


def find_wood():
    framedir_wdd = Path("/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/")

    df_markers_wdd = pd.read_csv(framedir_wdd / "df_markers.csv")

    wdd_markers = get_marker_coordinates_by_timestamp(
        "2024-09-02 12:02:36.710553+00:00", df_markers_wdd
    )
    image_path_wdd = "/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/cam0-2024-09-02T12_02_36.710553+00_00.png"
    image_wdd = mpimg.imread(image_path_wdd)
    # rotated_image = np.rot90(image_hd, 3)
    _, ax = plt.subplots()
    # ax[0].imshow(rotated_image, cmap="gray")
    ax.imshow(image_wdd, cmap="gray")
    ax.axis("off")
    print(wdd_markers)
    wdd_x, wdd_y = (
        [
            wdd_markers[0][0] + 100,
            wdd_markers[1][0] - 100,
            wdd_markers[2][0] + 100,
            wdd_markers[3][0] - 100,
        ],
        [
            wdd_markers[0][1] + 80,
            wdd_markers[1][1] + 80,
            wdd_markers[2][1] - 220,
            wdd_markers[3][1] - 220,
        ],
    )
    # wdd_x, wdd_y = wdd_markers[2]
    ax.scatter(wdd_x, wdd_y, c="red", s=5)
    plt.show()


if __name__ == "__main__":
    # find_wood()

    draw_roi()
    #
    # draw_points()

    # framedir_hd = Path(
    #     "/home/niklas/Documents/dev/uni/bees/bee-data/single_camera_frames/"
    # )
    # framedir_wdd = Path("/home/niklas/Documents/dev/uni/bees/bee-data/fullframes/")
    #
    # df_markers_hd = pd.read_csv(framedir_hd / "df_markers.csv")
    # df_markers_wdd = pd.read_csv(framedir_wdd / "df_markers.csv")
    #
    # hd_markers = get_marker_coordinates_by_timestamp(
    #     "2024-08-16T13:18:03.151660+00:00", df_markers_hd
    # )
    # wdd_markers = get_marker_coordinates_by_timestamp(
    #     "2024-08-16T13:18:03.151660+00:00", df_markers_wdd
    # )
    # converter = Converter(hd_markers)
    # sys.exit(0)

    # converter = Converter(
    #     marker_coords_hd[0],
    #     marker_coords_hd[1],
    #     marker_coords_wdd[0],
    #     marker_coords_wdd[1],
    #     distance_cm=39,
    # )
    # detection_coords_wdd = converter.convert_from_a_to_b(detection_coords_hd)


# cam_id,timestamp,marker_num,x,y,marker_type,score,filename
