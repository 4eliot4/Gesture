import argparse
import csv
from pathlib import Path

from video_preprocessor import VideoPreprocessor


def load_gloss_mapping(csv_path):
    mapping = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row["label"]).strip()
            gloss = str(row["gloss"]).strip()
            mapping[video_id] = gloss
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labels.csv (label,gloss)")
    parser.add_argument("--video", required=True, help="Path to the video (e.g. ./videos/195.mp4)")
    parser.add_argument("--out", default="./output", help="Folder to save parquet files")
    parser.add_argument("--annotate", action="store_true", help="Save annotated video")
    args = parser.parse_args()

    gloss_mapping = load_gloss_mapping(args.csv)

    video_path = Path(args.video)
    video_id = video_path.stem
    if video_id not in gloss_mapping:
        raise KeyError(f"{video_id} not in CSV labels")

    vp = VideoPreprocessor(
        include_pose=True,
        include_face=True,
        include_hands=True
    )
    result = vp.process_video(
        video_path=str(video_path),
        output_dir=args.out,
        gloss_mapping=gloss_mapping,
        save_annotated_video=args.annotate,
    )

    print("Video:", result["video_name"])
    print("Frames:", result["total_frames"])



if __name__ == "__main__":
    main()