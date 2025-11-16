import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from typing import Dict, List

# NOTE: We use the normalized landmarks (normalized based on height and width of the image)


class VideoPreprocessor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        include_pose: bool = True,
        include_face: bool = True,
        include_hands: bool = True,
    ):
        """
        Initialize MediaPipe Holistic model

        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            include_pose: Whether to include pose landmarks in extraction
            include_face: Whether to include face landmarks in extraction
            include_hands: Whether to include hand landmarks in extraction
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Store confidence parameters
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize Holistic model
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Body part selection flags
        self.include_pose = include_pose
        self.include_face = include_face
        self.include_hands = include_hands

    def calculate_visibility_score(self, landmark, margin: float = 0.1) -> float:
        """
        Returns visibility score [0,1] based on how far landmark is from boundaries.
        Useful for detecting when hands/faces are leaving the frame.

        Args:
            landmark: MediaPipe landmark object with x, y attributes (normalized [0,1])
            margin: how close to edge before we consider it "leaving"

        Returns:
            float: visibility score between 0.0 (outside/at edge) and 1.0 (well inside frame)
        """
        x, y = landmark.x, landmark.y

        # If clearly outside frame
        if x < 0 or x > 1 or y < 0 or y > 1:
            return 0.0

        # Calculate distance from nearest edge
        dist_to_edge = min(x, 1 - x, y, 1 - y)

        # Normalize: 0 at boundary, 1.0 at margin distance inward
        visibility = min(dist_to_edge / margin, 1.0)

        return visibility

    def extract_landmarks(self, results) -> Dict[str, np.ndarray]:
        """
        Extract ALL landmarks from MediaPipe Holistic (normalized coordinates [0,1])
        Returns dictionary with pose, face, left_hand, right_hand landmarks (based on include flags)
        Each landmark has [x, y, z, visibility] format
        """
        landmarks_data = {}

        # Pose landmarks - extract ALL pose landmarks
        if self.include_pose:
            if results.pose_landmarks:
                pose_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    pose_landmarks.append(
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    )
                landmarks_data["pose"] = np.array(pose_landmarks)
            else:
                # MediaPipe pose has 33 landmarks
                landmarks_data["pose"] = np.full((33, 4), np.nan)

        # Face landmarks - extract ALL face landmarks
        if self.include_face:
            if results.face_landmarks:
                face_landmarks = []
                for landmark in results.face_landmarks.landmark:
                    visibility = self.calculate_visibility_score(landmark)
                    face_landmarks.append(
                        [landmark.x, landmark.y, landmark.z, visibility]
                    )
                landmarks_data["face"] = np.array(face_landmarks)
            else:
                # MediaPipe face mesh has 468 landmarks
                landmarks_data["face"] = np.full((468, 4), np.nan)

        # Hand landmarks - extract ALL hand landmarks
        if self.include_hands:
            # Left hand landmarks
            if results.left_hand_landmarks:
                left_hand_landmarks = []
                for landmark in results.left_hand_landmarks.landmark:
                    visibility = self.calculate_visibility_score(landmark)
                    left_hand_landmarks.append(
                        [landmark.x, landmark.y, landmark.z, visibility]
                    )
                landmarks_data["left_hand"] = np.array(left_hand_landmarks)
            else:
                # MediaPipe hand has 21 landmarks
                landmarks_data["left_hand"] = np.full((21, 4), np.nan)

            # Right hand landmarks
            if results.right_hand_landmarks:
                right_hand_landmarks = []
                for landmark in results.right_hand_landmarks.landmark:
                    visibility = self.calculate_visibility_score(landmark)
                    right_hand_landmarks.append(
                        [landmark.x, landmark.y, landmark.z, visibility]
                    )
                landmarks_data["right_hand"] = np.array(right_hand_landmarks)
            else:
                # MediaPipe hand has 21 landmarks
                landmarks_data["right_hand"] = np.full((21, 4), np.nan)

        return landmarks_data

    def draw_landmarks_on_frame(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw all holistic landmarks on the frame
        """
        annotated_frame = frame.copy()

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_frame

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        gloss_mapping: Dict[str, str],
        save_annotated_video: bool = True,
    ) -> Dict:
        """
        Process a single video file

        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            gloss_mapping: Dictionary mapping video_id to gloss name
            save_annotated_video: Whether to save annotated video with landmarks drawn
        """
        video_name = Path(video_path).stem

        # video_name is the video_id (e.g., "12345" from "12345.mp4")
        gloss = gloss_mapping[video_name]

        output_dir_gloss = os.path.join(output_dir, gloss)
        os.makedirs(output_dir_gloss, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_name}")
        print(
            f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}"
        )

        # Setup output video writer if needed
        if save_annotated_video:
            output_video_path = os.path.join(
                output_dir_gloss, f"{video_name}_annotated.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )

        # Storage for landmarks
        original_landmarks = []

        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe Holistic
            results = self.holistic.process(rgb_frame)

            # Extract original landmarks
            frame_landmarks = self.extract_landmarks(results)
            original_landmarks.append(frame_landmarks)

            # Draw landmarks and save annotated video
            if save_annotated_video:
                rgb_frame.flags.writeable = True
                annotated_frame = self.draw_landmarks_on_frame(
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results
                )
                out_video.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        if save_annotated_video:
            out_video.release()
            print(f"Annotated video saved to: {output_video_path}")

        # Save dataset
        self.save_landmarks_dataset(
            original_landmarks,
            video_name,
            gloss,
            output_dir_gloss,
            coordinate_system="original",
        )

        return {
            "video_name": video_name,
            "total_frames": frame_count,
            "original_landmarks": original_landmarks,
            "fps": fps,
            "resolution": (frame_width, frame_height),
        }

    def save_landmarks_dataset(
        self,
        landmarks_data: List,
        video_name: str,
        gloss: str,
        output_dir: str,
        coordinate_system: str = "original",
    ):
        """
        Save landmarks dataset in Parquet format (tabular structure)

        Args:
            landmarks_data: List of frame landmarks (Dict with separate pose, face, left_hand, right_hand)
            video_name: Name of the video
            gloss: The gloss label for this video
            output_dir: Output directory
            coordinate_system: "original" (only option)
        """
        rows = []

        for frame_id, frame_landmarks in enumerate(landmarks_data):
            if frame_landmarks is None:
                continue

            # Process each body part
            for body_part in ["pose", "face", "left_hand", "right_hand"]:
                if body_part not in frame_landmarks:
                    continue

                landmarks = frame_landmarks[body_part]

                # Add each landmark as a row
                for landmark_index, coords in enumerate(landmarks):
                    x, y, z, visibility = coords

                    # Create landmark_id like "84-face-0"
                    landmark_id = f"{frame_id}-{body_part}-{landmark_index}"

                    rows.append(
                        {
                            "video_name": video_name,
                            "gloss": gloss,
                            "frame_id": frame_id,
                            "landmark_id": landmark_id,
                            "body_part": body_part,
                            "landmark_index": landmark_index,
                            "x": x,
                            "y": y,
                            "z": z,
                            "visibility": visibility,
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save as Parquet
        output_path = os.path.join(output_dir, f"{video_name}_landmarks.parquet")
        df.to_parquet(output_path, index=False)

        print(f"Dataset saved to: {output_path}")
        print(f"Total landmarks: {len(df)} rows across {len(landmarks_data)} frames")

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        metadata_list: List[Dict],
        video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"],
        save_annotated_video: bool = True,
    ):
        """
        Process all videos in a folder

        Args:
            input_folder: Input folder containing videos
            output_folder: Output folder for processed data
            metadata_list: List of metadata dicts with gloss, video_id, signer_id, etc.
            video_extensions: List of video file extensions to process
            save_annotated_video: Whether to save annotated videos with landmarks drawn
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all video files first
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"*{ext}")))
            video_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        if not video_files:
            print(f"No video files found in {input_folder}")
            return

        # Create a set of video IDs that actually exist in the folder
        existing_video_ids = {Path(video_path).stem for video_path in video_files}
        print(f"Found {len(existing_video_ids)} video files in folder")

        # Filter metadata to only include videos that exist in the folder
        csv_rows = []
        gloss_mapping = {}

        for item in metadata_list:
            video_id = item["video_id"]
            gloss = item["gloss"]
            signer_id = item["signer_id"]

            # Only add to CSV if the video file actually exists
            if video_id in existing_video_ids:
                gloss_mapping[video_id] = gloss

                # Build relative path to parquet file (output_folder/gloss/video_id_landmarks.parquet)
                relative_path = f"{gloss}/{video_id}_landmarks.parquet"

                csv_rows.append({
                    "path": relative_path,
                    "participant_id": signer_id,
                    "sequence_id": video_id,
                    "sign": gloss
                })

        # Save CSV file
        csv_df = pd.DataFrame(csv_rows)
        csv_path = output_path / "train.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"CSV metadata saved to: {csv_path}")
        print(f"Total entries in CSV: {len(csv_rows)} (filtered from {len(metadata_list)} total metadata entries)")

        print(f"Save annotated videos: {save_annotated_video}")

        # Process videos sequentially
        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")
            self.process_video(
                str(video_path),
                str(output_path),
                gloss_mapping=gloss_mapping,
                save_annotated_video=save_annotated_video,
            )

        print(f"\nAll videos processed! Output saved to: {output_folder}")
