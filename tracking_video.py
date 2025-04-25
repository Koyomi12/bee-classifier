import tempfile

# this functions create a video showing beesbook tag detections overlayed and uses the 3 steps I told you about.
# 1) extract video frames to a temp directory
# 2) overlay on the video frames by writing in place
# 3) create a new video from these frames (then delete the temp files)

# -- use tmp to extract all frames from video
# -- draw roi and bee location on video frames using PIL or opencv
# --- frame timings might be relevant for bee location
# -- use ffmpeg to combine


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pass


if __name__ == "__main__":
    main()
