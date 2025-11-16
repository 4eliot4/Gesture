#!/usr/bin/env python3
"""
Video preprocessing script with command-line arguments.
Processes sign language videos using MediaPipe Holistic to extract landmarks.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from src.data_preprocessing.video_preprocessor import VideoPreprocessor


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Process sign language videos to extract pose/face/hand landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output paths
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing input videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to folder where processed data will be saved",
    )

    # Body part selection
    parser.add_argument(
        "--include_pose",
        action="store_true",
        default=True,
        help="Include pose landmarks in extraction",
    )
    parser.add_argument(
        "--include_face",
        action="store_true",
        default=False,
        help="Include face landmarks in extraction",
    )
    parser.add_argument(
        "--include_hands",
        action="store_true",
        default=True,
        help="Include hand landmarks in extraction",
    )

    # MediaPipe confidence thresholds
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for MediaPipe (0.0 to 1.0)",
    )


    # Video extensions
    parser.add_argument(
        "--video_extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to process",
    )

    # Output options
    parser.add_argument(
        "--save_annotated_video",
        action="store_true",
        default=False,
        help="Save annotated videos with landmarks drawn (can be slow and use disk space)",
    )

    # Dataset info JSON
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to dataset JSON file to map video IDs to gloss names (e.g., data/WLASL_v0.3.json)",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Validate input paths
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata from JSON
    df_wlasl = pd.read_json(args.dataset_json)
    df_exploded = df_wlasl.explode("instances")
    df_wlasl_info = pd.concat(
        [df_exploded["gloss"], df_exploded["instances"].apply(pd.Series)], axis=1
    )

    # Create metadata list with gloss, video_id, and signer_id
    metadata_list = []
    for _, row in df_wlasl_info.iterrows():
        metadata_list.append({
            "gloss": row["gloss"],
            "video_id": str(row["video_id"]),
            "signer_id": row["signer_id"]
        })

    print(f"Loaded metadata for {len(metadata_list)} videos from {args.dataset_json}")

    # Create VideoPreprocessor instance
    preprocessor = VideoPreprocessor(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        include_pose=args.include_pose,
        include_face=args.include_face,
        include_hands=args.include_hands,
    )

    # Process all videos in the folder
    try:
        preprocessor.process_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            metadata_list=metadata_list,
            video_extensions=args.video_extensions,
            save_annotated_video=args.save_annotated_video
        )
        print("\n" + "=" * 80)
        print("Processing completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
