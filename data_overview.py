import json
from zipfile import ZipFile

from tqdm import tqdm

from daily_data_processing import ZIPS_PATH, is_wood_in_frame


def main():
    """
    Gives a daily overview of the number of detections, videos  per dance type
    and how many videos are filtered out by the wood filter.
    """
    data = dict()
    zip_files = sorted(list(ZIPS_PATH.rglob("*")))
    for zip_path in tqdm(zip_files):
        if not zip_path.suffix == ".zip":
            continue
        day_data = {
            "waggle": {"detections": 0, "videos": 0, "wood filter": 0},
            "activating": {"detections": 0, "videos": 0, "wood filter": 0},
            "ventilating": {"detections": 0, "videos": 0, "wood filter": 0},
            "other": {"detections": 0, "videos": 0, "wood filter": 0},
        }
        with ZipFile(zip_path) as zip_file:
            files = zip_file.namelist()
            metadata_filenames = list(
                filter(lambda filename: filename.endswith(".json"), files)
            )
            for metadata_filename in tqdm(metadata_filenames):
                with zip_file.open(metadata_filename) as metadata_file:
                    json_data = json.load(metadata_file)
                label = json_data["predicted_class_label"]
                day_data[label]["detections"] += 1
                if is_wood_in_frame(json_data):
                    day_data[label]["wood filter"] += 1

                # find matching video file
                video_filename = metadata_filename.replace("waggle.json", "frames.apng")
                if video_filename in files:
                    day_data[label]["videos"] += 1
        date_str = zip_path.stem
        data[date_str] = day_data
    with open("output/data_overview.json", "w") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    main()
