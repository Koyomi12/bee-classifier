import datetime
import json
import math
from pathlib import Path
from zipfile import ZipFile

import cv2
import pandas as pd
from cv2.typing import MatLike
from tqdm import tqdm

from coordinate_transformation import CoordinateConverter
from daily_data_processing import (
    FRAMEDIR_WDD,
    TAGGED,
    get_marker_coordinates_by_timestamp,
)

MANUALLY_ANNOTATED_DATA_PATH = Path(
    "/home/niklas/Documents/dev/uni/bees/bee-data/manual_annotations/Final_weltwinkel_tunnel_richtungsvektor.xlsx"
)
FRAMEDIR_HD = Path("/mnt/trove/wdd/wdd_videos_2024/single_camera_frames/")
WDD_PATH = Path("/mnt/trove/wdd/")
PROCESSED_DATA_PATH = Path("/mnt/local_storage/processed-bee-data/")
CUT_VIDEO_DIR = Path("/mnt/local_storage/dance_videos_tunnel_2024/")

HD_CAM_RESOLUTION = (3520, 4608)
HD_CAM_FPS = 15

TIME_DELTA = datetime.timedelta(seconds=1)
COORDINATE_EPSILON = 100


def main():
    tunnel_bees_df = pd.read_excel(MANUALLY_ANNOTATED_DATA_PATH)
    results = dict()
    for cut_video_rel_path_str in tunnel_bees_df["video_name"].unique():
        # The names look like this:
        # "./cam-1_20240904T122936.14581.115Z--20240904T123435.952057.695Z_14.30_40.mp4".
        # When there are multiple fractional second sections and the first
        # section doesn't have at least 6 digits, the fromisoformat function
        # can't parse the timestamp. If there is only one fractional second
        # section, the number of digits there doesn't matter. In this case, the
        # number of digits in the first fractional second section seems to
        # vary, so we truncate the second section.
        cut_video_rel_path = Path(cut_video_rel_path_str)
        video_filename_parts = cut_video_rel_path.name.split("_")
        original_video_name = (
            video_filename_parts[0]
            + "_"
            + video_filename_parts[1]
            # One of the relative filepaths is missing a "Z".
            + ("" if video_filename_parts[1].endswith("Z") else "Z")
            + cut_video_rel_path.suffix
        )
        timestampstr = original_video_name.replace("cam-1_", "").partition("--")[0]
        timestampstr = truncate_higher_precision_timestamp(timestampstr)
        timestamp = datetime.datetime.fromisoformat(timestampstr)
        datestr = timestamp.strftime("%Y-%m-%d")
        zip_path = (
            WDD_PATH
            / ("wdd_output_" + str(timestamp.year))
            / "cam0"
            / timestamp.year
            / timestamp.month
            / (datestr + ".zip")
        )
        with ZipFile(zip_path) as zip_file:
            files = zip_file.namelist()
            metadata_filenames = list(
                filter(lambda filename: filename.endswith(".json"), files)
            )
            jsons = []
            for metadata_filename in tqdm(metadata_filenames):
                with zip_file.open(metadata_filename) as metadata_file:
                    json_data = json.load(metadata_file)
                    jsons.append(json_data)
        all_candidates = pd.DataFrame(jsons)
        all_candidates = all_candidates.astype({"waggle_id": "string"})

        original_video_path = (
            WDD_PATH
            / ("wdd_videos_" + str(timestamp.year))
            / timestamp.strftime("%Y%m%d")
            / "cam-1"
            / original_video_name
        )
        if original_video_path.exists():
            try:
                video_fps = get_video_fps(original_video_path)
            except Exception as e:
                video_fps = HD_CAM_FPS
                print(e, f"\nWill assume {video_fps} FPS")
        else:
            video_fps = HD_CAM_FPS

        cut_video_path = CUT_VIDEO_DIR / Path(cut_video_rel_path).name

        results[original_video_name] = []

        # Iterate over each waggle run of a manually annotated dance
        manually_annotated_rows = tunnel_bees_df.loc[
            tunnel_bees_df["video_name"] == cut_video_rel_path_str
        ]
        for _, manually_annotated_row in manually_annotated_rows.iterrows():
            try:
                frame_offset = get_first_matching_frame_index(
                    original_video=original_video_path, cut_video=cut_video_path
                )
            except Exception as e:
                print(e)
                continue

            candidates_after_timestamp_check = filtered_by_timestamp(
                candidates=all_candidates,
                timestamp=timestamp,
                delta=TIME_DELTA,
                start_frame=frame_offset
                + manually_annotated_row["waggle_start_frames"],
                video_fps=video_fps,
            )

            if candidates_after_timestamp_check.shape[0] == 0:
                continue

            # narrow candidates by excluding all waggles that aren't labeled as tagged and waggle
            # we use the data that was manully corrected using the bee tag corrector here
            data_csv_path = Path.joinpath(PROCESSED_DATA_PATH, datestr, "data.csv")
            csv_df = pd.read_csv(
                data_csv_path,
                # can remove those we don't care about or add confidence as well
                dtype={
                    "day_dance_id": "string",
                    "waggle_id": "string",
                    "category": "Int64",
                    "category_label": "string",
                    "corrected_category": "Int64",
                    "corrected_category_label": "string",
                    "dance_type": "string",
                    "corrected_dance_type": "string",
                },
            )
            candidates_after_tagged_waggle_check = []
            for _, candidate in candidates_after_timestamp_check.iterrows():
                csv_row = csv_df.loc[csv_df["waggle_id"] == candidate["waggle_id"]]
                csv_row.reset_index(drop=True, inplace=True)
                # Waggle IDs are unique, so this should always be a single row
                assert csv_row.shape[0] <= 1

                if csv_row.shape[0] == 0:
                    continue
                elif is_tagged(csv_row) and is_waggle(csv_row):
                    candidates_after_tagged_waggle_check.append(candidate)
                else:
                    print(f"tagged: {is_tagged(csv_row)}")
                    print(f"waggle: {is_waggle(csv_row)}")

            # use coordinates to narrow candidates further
            df_markers_wdd = pd.read_csv(FRAMEDIR_WDD / "df_markers.csv")
            df_markers_hd = pd.read_csv(FRAMEDIR_HD / "df_markers.csv")
            candidates_after_coordinate_check = [
                candidate
                for candidate in candidates_after_tagged_waggle_check
                if is_matching_coordinates(
                    automatically_annotated_data=candidate,
                    manually_annotated_data=manually_annotated_row,
                    df_markers_hd=df_markers_hd,
                    df_markers_wdd=df_markers_wdd,
                )
            ]

            results[original_video_name].append(
                {
                    # "waggle_index": manually_annotated_row["waggle_index"],
                    "manual_annotation": dict(
                        waggle_index=manually_annotated_row["waggle_index"],
                        waggle_start_position=(
                            manually_annotated_row["waggle_start_positions_x"],
                            manually_annotated_row["waggle_start_positions_y"],
                        ),
                        thorax_position=(
                            manually_annotated_row["thorax_positions_x"],
                            manually_annotated_row["thorax_positions_y"],
                        ),
                        waggle_angle_deg=manually_annotated_row[
                            " Positiver Tanzwinkel in Grad  "
                        ],
                    ),
                    "candidates": [
                        dict(
                            waggle_id=candidate["waggle_id"],
                            timestamp_begin=candidate["timestamp_begin"],
                            roi_center=candidate["roi_center"],
                            waggle_angle_deg=(candidate["waggle_angle"] / math.pi * 180)
                            % 360,
                            waggle_duration=candidate["waggle_duration"],
                        )
                        for candidate in candidates_after_coordinate_check
                    ],
                }
            )
    with open("output/find_waggles.json", "w") as fp:
        json.dump(results, fp, indent=2)


def truncate_higher_precision_timestamp(timestamp: str) -> str:
    """
    Truncates a timestamp string with multiple sections for fractional seconds
    down to one fractional second section.
    """
    precision_section_count = timestamp.count(".")
    if precision_section_count >= 2:
        sections = timestamp.split(".")
        return sections[0] + "." + sections[1] + "Z"
    else:
        return timestamp


def filtered_by_timestamp(
    candidates: pd.DataFrame,
    timestamp: datetime.datetime,
    delta: datetime.timedelta,
    start_frame: int,
    video_fps: int,
):
    # adjust timestamp by frame index
    seconds_played = start_frame / video_fps
    time_change = datetime.timedelta(seconds=seconds_played)
    adjusted_time = timestamp + time_change
    # use a delta to allow for some inaccuracy
    start_time = adjusted_time - delta

    # start_frame of last waggle run + the 2 seconds
    # actually just use start_time plus two seconds here because we are iterating over each waggle run
    end_time = adjusted_time + delta

    # find waggle detection candidates by adjusted timestamp
    # TODO: iterate over the metadata_df for the day to find detections by timestamp
    candidates["timestamp_begin"] = pd.to_datetime(
        candidates["timestamp_begin"]
    )  # maybe do this while loading it in the first place?
    # print(candidates.dtypes)
    return candidates.loc[
        (candidates["timestamp_begin"] >= start_time)
        & (candidates["timestamp_begin"] <= end_time)
    ]


def is_tagged(df: pd.DataFrame) -> bool:
    return (
        df.at[0, "category_label"] == TAGGED
        and pd.isna(df.at[0, "corrected_category_label"])
        or not pd.isna(df.at[0, "corrected_category_label"])
        and df.at[0, "corrected_category_label"] == TAGGED
    )


def is_waggle(df: pd.DataFrame) -> bool:
    return (
        df.at[0, "dance_type"] == "waggle"
        and pd.isna(df.at[0, "corrected_dance_type"])
        or not pd.isna(df.at[0, "corrected_dance_type"])
        and df.at[0, "corrected_dance_type"] == "waggle"
    )


def is_matching_coordinates(
    automatically_annotated_data,
    manually_annotated_data,
    df_markers_hd,
    df_markers_wdd,
    epsilon=COORDINATE_EPSILON,
) -> bool:
    wdd_markers = get_marker_coordinates_by_timestamp(
        detection_timestamp=datetime.datetime.isoformat(
            automatically_annotated_data["timestamp_begin"]
        ),
        df_markers=df_markers_wdd,
    )
    hd_markers = get_marker_coordinates_by_timestamp(
        detection_timestamp=datetime.datetime.isoformat(
            automatically_annotated_data["timestamp_begin"]
        ),
        df_markers=df_markers_hd,
    )
    converter = CoordinateConverter(src_markers=hd_markers, dst_markers=wdd_markers)
    # manually annotated coordinates
    # The dances were annotated using the clockwise rotation flag which means
    # we need to reverse the rotation to get compatible coordinates.
    x_hd_rotated = manually_annotated_data["waggle_start_positions_x"]
    y_hd_rotated = manually_annotated_data["waggle_start_positions_y"]
    x_hd, y_hd = unrotate((x_hd_rotated, y_hd_rotated))
    x_wdd, y_wdd = converter.transform_coordinates(x=x_hd, y=y_hd)
    x, y = automatically_annotated_data["roi_center"]
    x_with_offset, y_with_offset = fix_padding_error((x, y))
    return is_adjacent(
        coordinates1=(x_wdd, y_wdd),
        coordinates2=(x_with_offset, y_with_offset),
        epsilon=epsilon,
    )


def unrotate(rotated_coordinates: tuple[int, int]) -> tuple[int, int]:
    (rotated_x, rotated_y) = rotated_coordinates
    x = rotated_y
    y = HD_CAM_RESOLUTION[1] - rotated_x
    return (x, y)


def fix_padding_error(coordinates: tuple[int, int]) -> tuple[int, int]:
    """Adds a 125 pixel offset."""
    x, y = coordinates
    return (x + 125, y + 125)


def is_adjacent(
    coordinates1: tuple[int, int],
    coordinates2: tuple[int, int],
    epsilon: int = COORDINATE_EPSILON,
) -> bool:
    x1, y1 = coordinates1
    x2, y2 = coordinates2
    return abs(x1 - x2) <= epsilon and abs(y1 - y2) <= epsilon


def get_video_fps(video_path: Path) -> int:
    """Extracts FPS from video metadata."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if fps > 0:
        return round(fps)
    else:
        raise ValueError(f"FPS for {video_path} is negative: {fps}")


def get_first_matching_frame_index(original_video: Path, cut_video: Path) -> int:
    """Finds the index of the frame in the original video that matches the first frame of the cut video."""

    cut_hd_video_starting_frame = get_starting_frame(cut_video)

    capture = cv2.VideoCapture(original_video)
    if not capture.isOpened():
        raise Exception(f"Unable to open: {original_video}")
    match_found = False
    frame_index = 0
    while capture.isOpened():
        frame_exists, frame = capture.read()
        if not frame_exists:
            break
        if (frame == cut_hd_video_starting_frame).all():
            match_found = True
            break
        frame_index += 1
    capture.release()
    cv2.destroyAllWindows()

    if match_found:
        return frame_index
    else:
        raise Exception(f"Unable to find matching frame for {cut_video}")


def get_starting_frame(video: Path) -> MatLike:
    """Returns the first frame of video."""
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise Exception(f"Unable to open: {video}")

    frame_exists, frame = capture.read()
    if not frame_exists:
        raise Exception(f"No frame in: {video}")

    cut_video_starting_frame = frame
    capture.release()
    cv2.destroyAllWindows()
    return cut_video_starting_frame


if __name__ == "__main__":
    # get_first_matching_frame_index(
    #     Path(
    #         "/home/niklas/Documents/dev/uni/bees/bee-data/wdd_videos/20240902/cam-1/cam-1_20240902T144239.709998.334Z--20240902T144739.647474.914Z.mp4"
    #     ),
    #     Path(
    #         "/home/niklas/Documents/dev/uni/bees/bee-data/wdd_videos_cut/cam-1_20240902T144239.709998.334Z--20240902T144739.647474.914Z_16.44_o4.mp4"
    #     ),
    # )
    main()
