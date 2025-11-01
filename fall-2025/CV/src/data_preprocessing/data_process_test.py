import json
from video_preprocessor import VideoPreprocessor, CoordinateSystem

if __name__ == "__main__":
    with open("gloss_map.json") as f:
        gloss_map = json.load(f)

    vp = VideoPreprocessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        include_pose=True,
        include_face=False,   # start simple (pose only)
        include_hands=True
    )

    out = vp.process_video(
        video_path="test_videos/hello.mp4",
        output_dir="processed_out",
        gloss_mapping=gloss_map,
        save_annotated_video=True,
        coordinate_systems=[CoordinateSystem.ORIGINAL, CoordinateSystem.SHOULDER_CENTERED]
    )
    print(out)