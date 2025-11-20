import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import os
import hashlib
import inspect

from .utils.pose_utils import Landmark, LandmarkList, Pose, unify_normalized_landmarks_to_shoulder_center, build_unified_adjacency_matrix, get_hand_mapping
from . import FACE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS, POSE_CONNECTIONS, FACEMESH_CONTOURS, HAND_CONNECTIONS


class UnifiedSkeletonDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for loading skeleton data from Parquet files.

    Only loads individual Parquet files when accessed in __getitem__, not all at once.
    Each sample represents one video with landmarks across multiple frames.
    """

    def __init__(
        self,
        data_root: str,
        cache_dir: str,
        gloss_map: Dict[str, int],
        metadata_file: Optional[str] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        body_parts: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset loader.

        Args:
            data_root: Root directory containing gloss subdirectories with .parquet files
            gloss_map: map that contains gloss to lable
            metadata_file: Optional path to csv file that contains the metadata of the path to the files, glosses and other info (if None, loads from data_root/train.csv)
            min_frames: Minimum number of frames required (filter out shorter videos)
            max_frames: Maximum number of frames to use (truncate longer videos)
            body_parts: List of body parts to include (default: all - ['pose', 'face', 'left_hand', 'right_hand'])
        """
        self.data_root = Path(data_root)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.body_parts = body_parts or ["pose", "face", "left_hand", "right_hand"]
        self.gloss_map = gloss_map

        # Generate configuration hash for cache versioning
        self.config_hash = self._generate_config_hash()

        self.cache_dir = os.path.join(cache_dir, self.config_hash)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load metadata CSV
        if metadata_file is None:
            metadata_file = os.path.join(self.data_root, "train.csv")

        self.df_metadata = pd.read_csv(metadata_file)
        self.df_metadata = self.df_metadata[
            self.df_metadata["sign"].isin(self.gloss_map.keys())
        ]

        self.num_classes = len(gloss_map)

        # Build video files list from CSV
        self.video_files = self.df_metadata.to_dict(orient="records")

        # Build unified adjacency matrix once (same for all samples)
        self.adjacency_matrix = build_unified_adjacency_matrix(
            pose_landmarks=POSE_LANDMARKS if "pose" in self.body_parts else None,
            face_landmarks=FACE_LANDMARKS if "face" in self.body_parts else None,
            left_hand_landmarks=HAND_LANDMARKS if "left_hand" in self.body_parts else None,
            right_hand_landmarks=HAND_LANDMARKS if "right_hand" in self.body_parts else None,
            pose_connections=POSE_CONNECTIONS if "pose" in self.body_parts else None,
            face_connections=FACEMESH_CONTOURS if "face" in self.body_parts else None,
            hand_connections=HAND_CONNECTIONS if ("left_hand" in self.body_parts or "right_hand" in self.body_parts) else None,
        )

        print(f"Found {len(self.video_files)} videos from {self.data_root}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Body parts: {self.body_parts}")
        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        print(f"Cache directory: {self.cache_dir}")

    def _generate_config_hash(self) -> str:
        """
        Generate a hash based on:
        1. Configuration parameters
        2. The source code of this class (to detect implementation changes)

        This ensures cache invalidation when either parameters or implementation change.
        """
        # Get the source code of this class
        class_source = inspect.getsource(self.__class__)

        # Create a string with all configuration parameters
        config_str = (
            f"class_source={class_source}"
            f"body_parts={sorted(self.body_parts)}"
            f"min_frames={self.min_frames}"
            f"max_frames={self.max_frames}"
            f"gloss_map={sorted(self.gloss_map.items())}"
        )

        # Generate MD5 hash and return first 8 characters
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_files)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single video sample - loads the parquet file for this video only.

        Returns:
            Dictionary containing:
                - 'landmarks': Tensor of shape (num_frames, num_landmarks, 4) [x, y, z, visibility]
                - 'adjacency': Tensor of shape (num_landmarks, num_landmarks) - unified adjacency matrix
                - 'label': Integer label for the gloss
                - 'num_frames': Number of frames in this sample
                - 'video_name': String video name
                - 'gloss': String name of the gloss
        """
        path = os.path.join(self.cache_dir, f"{idx}.pt")

        if os.path.exists(path):
            return torch.load(path)

        video_info = self.video_files[idx]

        # Load parquet file for this video (join with data_root if path is relative)
        parquet_path = video_info["path"]
        if not os.path.isabs(parquet_path):
            parquet_path = os.path.join(self.data_root, parquet_path)
        df = pd.read_parquet(parquet_path)

        body_part_name = "type"

        # Filter by selected body parts
        df = df[df["type"].isin(self.body_parts)]

        # Filter by specific landmark indices based on body part
        # Assuming the parquet has a 'landmark_index' column
        mask = (
            ((df[body_part_name] == "pose") & (df["landmark_index"].isin(POSE_LANDMARKS))) |
            ((df[body_part_name] == "face") & (df["landmark_index"].isin(FACE_LANDMARKS))) |
            ((df[body_part_name] == "left_hand") & (df["landmark_index"].isin(HAND_LANDMARKS))) |
            ((df[body_part_name] == "right_hand") & (df["landmark_index"].isin(HAND_LANDMARKS)))
        )
        df = df[mask]

        # Get number of frames
        num_frames = df["frame"].nunique()

        # Apply filters (incomplete)
        if self.min_frames and num_frames < self.min_frames:
            # Return empty sample or handle as needed
            pass

        # Truncate frames if needed
        if self.max_frames and num_frames > self.max_frames:
            # Get the first max_frames sorted frame IDs
            valid_frame_ids = sorted(df["frame"].unique())[:self.max_frames]
            df = df[df["frame"].isin(valid_frame_ids)]
            num_frames = self.max_frames

        # Organize data by frame and apply shoulder centering
        all_frames_shoulder_centered = []

        for frame_id in sorted(df["frame"].unique()):
            frame_df = df[df["frame"] == frame_id].sort_values(
                ["type", "landmark_index"]
            )

            # Organize by body part
            pose_landmarks = []
            face_landmarks = []
            left_hand_landmarks = []
            right_hand_landmarks = []

            # Get mapping of actual hand positions to labeled hands
            # E.g., {"left_hand": "right_hand", "right_hand": "left_hand"} means labels are swapped
            hand_mapping = get_hand_mapping(frame_df, HAND_LANDMARKS)

            # Extract landmarks for each body part
            # IMPORTANT: Must iterate in the same order as defined in POSE_LANDMARKS, FACE_LANDMARKS, HAND_LANDMARKS
            # to match the adjacency matrix indices
            for body_part in self.body_parts:
                # For hands, use the mapping to determine which labeled hand to actually read from
                if body_part in ["left_hand", "right_hand"]:
                    labeled_hand = hand_mapping[body_part]
                    if labeled_hand is not None:
                        # Here we are saying in the position of the left hand we say that in realliity the right hand is the left hand, so instead we grab the left hand to add it to the body_part=left_hand
                        part_df = frame_df[frame_df["type"] == labeled_hand]
                    else:
                        part_df = frame_df[frame_df["type"] == body_part]
                else:
                    part_df = frame_df[frame_df["type"] == body_part]

                # Get the landmark indices list for this body part (defines the ordering)
                if body_part == "pose":
                    landmark_order = POSE_LANDMARKS
                elif body_part == "face":
                    landmark_order = FACE_LANDMARKS
                elif body_part == "left_hand":
                    landmark_order = HAND_LANDMARKS
                elif body_part == "right_hand":
                    landmark_order = HAND_LANDMARKS
                else:
                    continue

                # Extract landmarks in the order specified by landmark_order
                for landmark_idx in landmark_order:
                    row = part_df[part_df["landmark_index"] == landmark_idx]

                    if row.empty:
                        # Landmark not found, use zeros with visibility 0
                        landmark = Landmark(0.0, 0.0, 0.0, 0.0)
                    else:
                        row = row.iloc[0]
                        visibility = row.get("visibility", 1.0)
                        # Check if all coordinates are NaN (missing landmark)
                        if pd.isna(row["x"]) and pd.isna(row["y"]) and pd.isna(row["z"]):
                            visibility = 0.0

                        x = np.nan_to_num(row["x"], copy=False, nan=0.0)
                        y = np.nan_to_num(row["y"], copy=False, nan=0.0)
                        z = np.nan_to_num(row["z"], copy=False, nan=0.0)
                        landmark = Landmark(x, y, z, visibility)

                    if body_part == "pose":
                        pose_landmarks.append(landmark)
                    elif body_part == "face":
                        face_landmarks.append(landmark)
                    elif body_part == "left_hand":
                        left_hand_landmarks.append(landmark)
                    elif body_part == "right_hand":
                        right_hand_landmarks.append(landmark)

            # Create Pose object
            pose = Pose(
                face_landmarks=LandmarkList(face_landmarks) if face_landmarks else None,
                pose_landmarks=LandmarkList(pose_landmarks) if pose_landmarks else None,
                right_hand_landmarks=LandmarkList(right_hand_landmarks) if right_hand_landmarks else None,
                left_hand_landmarks=LandmarkList(left_hand_landmarks) if left_hand_landmarks else None
            )

            # Apply shoulder centering
            shoulder_centered = unify_normalized_landmarks_to_shoulder_center(pose)

            if shoulder_centered:
                # Concatenate all body parts in order: pose, face, left_hand, right_hand
                frame_coords = []
                for part_name in ["pose", "face", "left_hand", "right_hand"]:
                    if part_name in shoulder_centered and shoulder_centered[part_name]:
                        for lm in shoulder_centered[part_name]:
                            frame_coords.append([lm["x"], lm["y"], lm["z"], lm.get("visibility", 1.0)])


                all_frames_shoulder_centered.append(np.array(frame_coords))

        # Stack into (num_frames, num_landmarks, 4)
        if all_frames_shoulder_centered:
            landmarks = np.stack(all_frames_shoulder_centered, axis=0)
        else:
            # Fallback if shoulder centering fails
            landmarks = np.zeros((num_frames, 1, 4))

        result = {
            "landmarks": torch.from_numpy(landmarks).float(),
            "adjacency": torch.from_numpy(self.adjacency_matrix).float(),
            "label": torch.tensor(self.gloss_map[video_info["sign"]], dtype=torch.long),
            "num_frames": torch.tensor(num_frames, dtype=torch.long),
            "video_name": video_info["sequence_id"],
            "gloss": video_info["sign"],
        }

        # Save to cache atomically to avoid corruption from parallel workers
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix='.pt.tmp')
        try:
            os.close(tmp_fd)  # Close the file descriptor, torch.save will open it
            torch.save(result, tmp_path)
            os.replace(tmp_path, path)  # Atomic rename on POSIX systems
        except Exception as e:
            # Clean up temp file if something went wrong
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

        return result
