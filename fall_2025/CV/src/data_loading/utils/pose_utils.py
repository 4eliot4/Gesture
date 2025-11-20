from typing import Optional, Dict, List, Set, Tuple
import numpy as np
import pandas as pd


class Landmark:
    """Mimics MediaPipe Landmark structure"""
    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class LandmarkList:
    """Mimics MediaPipe NormalizedLandmarkList structure"""
    def __init__(self, landmarks: List[Landmark]):
        self.landmark = landmarks


class Pose:
    """
    Mimics MediaPipe Holistic results structure.

    Usage:
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                print(lm.x, lm.y, lm.z, lm.visibility)
    """
    def __init__(
        self,
        face_landmarks: Optional[LandmarkList] = None,
        pose_landmarks: Optional[LandmarkList] = None,
        right_hand_landmarks: Optional[LandmarkList] = None,
        left_hand_landmarks: Optional[LandmarkList] = None
    ):
        self.face_landmarks = face_landmarks
        self.pose_landmarks = pose_landmarks
        self.right_hand_landmarks = right_hand_landmarks
        self.left_hand_landmarks = left_hand_landmarks


# NOTE: theres one problem with working with the unified matrix, the problem is that we can't know which hand is in reality the left or right (because mediapipe just outputs based on position on camera), and mirroring is not enough because if right hand is now on the left side of the camera it will now say the contrary thing (now it is in the right position and not left)
def unify_normalized_landmarks_to_shoulder_center(results: Pose) -> Optional[Dict]:
        """
        Transform all MediaPipe Holistic landmarks to shoulder-centered coordinates.
        Shoulder center = midpoint between left shoulder (11) and right shoulder (12)

        Uses normalized coordinates [0,1] and transforms all body parts to a unified
        coordinate system centered at the shoulders.

        Note: z-coordinates have different scales but are combined using displacement vectors:
          - Pose z: relative to hips
          - Hand z: relative to wrist
          - Face z: relative to head center
        """
        if not results.pose_landmarks:
            return None

        pose_norm = results.pose_landmarks.landmark

        # Calculate shoulder center (NEW ORIGIN)
        left_shoulder = pose_norm[11]
        right_shoulder = pose_norm[12]

        shoulder_center = {
            "x": (left_shoulder.x + right_shoulder.x) / 2,
            "y": (left_shoulder.y + right_shoulder.y) / 2,
            "z": (left_shoulder.z + right_shoulder.z) / 2,
        }

        unified_data = {"pose": [], "face": [], "left_hand": [], "right_hand": []}

        # Transform pose to shoulder-centered
        for landmark in pose_norm:
            unified_data["pose"].append(
                {
                    "x": landmark.x - shoulder_center["x"],
                    "y": landmark.y - shoulder_center["y"],
                    "z": landmark.z - shoulder_center["z"],
                    "visibility": landmark.visibility,
                }
            )

        # Get shoulder-centered anchor points
        nose_shoulder_centered = unified_data["pose"][0]
        left_wrist_shoulder_centered = unified_data["pose"][15]
        right_wrist_shoulder_centered = unified_data["pose"][16]

        # Transform face using nose as anchor
        if results.face_landmarks:
            face_norm = results.face_landmarks.landmark
            nose_face = face_norm[1]  # Nose tip in face mesh

            for landmark in face_norm:
                # Calculate displacement vector from landmark to nose in face coordinate system
                offset_x = landmark.x - nose_face.x
                offset_y = landmark.y - nose_face.y
                offset_z = landmark.z - nose_face.z

                # Apply displacement to shoulder-centered nose position
                unified_data["face"].append(
                    {
                        "x": nose_shoulder_centered["x"] + offset_x,
                        "y": nose_shoulder_centered["y"] + offset_y,
                        "z": nose_shoulder_centered["z"] + offset_z,
                    }
                )

        # Transform left hand using wrist as anchor
        if results.left_hand_landmarks:
            left_hand_norm = results.left_hand_landmarks.landmark
            left_wrist_hand = left_hand_norm[0]

            for landmark in left_hand_norm:
                # Calculate displacement vector in hand's coordinate system
                offset_x = landmark.x - left_wrist_hand.x
                offset_y = landmark.y - left_wrist_hand.y
                offset_z = landmark.z - left_wrist_hand.z

                # Apply displacement to shoulder-centered wrist position
                unified_data["left_hand"].append(
                    {
                        "x": left_wrist_shoulder_centered["x"] + offset_x,
                        "y": left_wrist_shoulder_centered["y"] + offset_y,
                        "z": left_wrist_shoulder_centered["z"] + offset_z,
                    }
                )

        # Transform right hand using wrist as anchor
        if results.right_hand_landmarks:
            right_hand_norm = results.right_hand_landmarks.landmark
            right_wrist_hand = right_hand_norm[0]

            for landmark in right_hand_norm:
                # Calculate displacement vector in hand's coordinate system
                offset_x = landmark.x - right_wrist_hand.x
                offset_y = landmark.y - right_wrist_hand.y
                offset_z = landmark.z - right_wrist_hand.z

                # Apply displacement to shoulder-centered wrist position
                unified_data["right_hand"].append(
                    {
                        "x": right_wrist_shoulder_centered["x"] + offset_x,
                        "y": right_wrist_shoulder_centered["y"] + offset_y,
                        "z": right_wrist_shoulder_centered["z"] + offset_z,
                    }
                )

        return unified_data

def build_adjacency_matrix_from_connections(
    connections: Set[Tuple[int, int]],
    filtered_indices: List[int]
) -> np.ndarray:
    """
    Build an adjacency matrix for a body part using MediaPipe connections.

    Args:
        connections: Set of (start, end) tuples representing connections in original MediaPipe indices
        filtered_indices: List of landmark indices we want to keep

    Returns:
        Adjacency matrix of shape (len(filtered_indices), len(filtered_indices))
    """
    n = len(filtered_indices)
    adj = np.zeros((n, n), dtype=np.float32)

    # Create mapping from original index to filtered index
    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(filtered_indices)}

    # Fill adjacency matrix based on connections
    for start, end in connections:
        # Only add connection if both landmarks are in our filtered set
        if start in idx_map and end in idx_map:
            i, j = idx_map[start], idx_map[end]
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # Undirected graph

    return adj


def build_unified_adjacency_matrix(
    pose_landmarks: Optional[List[int]] = None,
    face_landmarks: Optional[List[int]] = None,
    left_hand_landmarks: Optional[List[int]] = None,
    right_hand_landmarks: Optional[List[int]] = None,
    pose_connections: Optional[Set[Tuple[int, int]]] = None,
    face_connections: Optional[Set[Tuple[int, int]]] = None,
    hand_connections: Optional[Set[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Build a unified adjacency matrix for the shoulder-centered coordinate system.
    This includes connections within each body part AND connections between body parts:
    - Hands to pose via wrists (hand wrist idx 0 connects to pose wrist idx 15/16)
    - Face to pose via nose (face nose idx 1 connects to pose nose idx 0)

    Args:
        pose_landmarks: List of pose landmark indices to include (None to exclude)
        face_landmarks: List of face landmark indices to include (None to exclude)
        left_hand_landmarks: List of left hand landmark indices to include (None to exclude)
        right_hand_landmarks: List of right hand landmark indices to include (None to exclude)
        pose_connections: Set of (start, end) tuples for pose connections (required if pose_landmarks provided)
        face_connections: Set of (start, end) tuples for face connections (required if face_landmarks provided)
        hand_connections: Set of (start, end) tuples for hand connections (required if hand landmarks provided)

    Returns:
        Unified adjacency matrix of shape (total_landmarks, total_landmarks)
    """
    # Calculate total size
    total_landmarks = 0
    if pose_landmarks is not None:
        total_landmarks += len(pose_landmarks)
    if face_landmarks is not None:
        total_landmarks += len(face_landmarks)
    if left_hand_landmarks is not None:
        total_landmarks += len(left_hand_landmarks)
    if right_hand_landmarks is not None:
        total_landmarks += len(right_hand_landmarks)

    # Initialize unified adjacency matrix
    unified_adj = np.zeros((total_landmarks, total_landmarks), dtype=np.float32)

    # Build individual adjacency matrices
    adj_matrices = {}
    if pose_landmarks is not None and pose_connections is not None:
        adj_matrices["pose"] = build_adjacency_matrix_from_connections(
            pose_connections, pose_landmarks
        )
    if face_landmarks is not None and face_connections is not None:
        adj_matrices["face"] = build_adjacency_matrix_from_connections(
            face_connections, face_landmarks
        )
    if left_hand_landmarks is not None and hand_connections is not None:
        adj_matrices["left_hand"] = build_adjacency_matrix_from_connections(
            hand_connections, left_hand_landmarks
        )
    if right_hand_landmarks is not None and hand_connections is not None:
        adj_matrices["right_hand"] = build_adjacency_matrix_from_connections(
            hand_connections, right_hand_landmarks
        )

    # Create index mappings from original MediaPipe indices to filtered indices
    pose_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(pose_landmarks)} if pose_landmarks is not None else {}
    face_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(face_landmarks)} if face_landmarks is not None else {}
    left_hand_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(left_hand_landmarks)} if left_hand_landmarks is not None else {}
    right_hand_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(right_hand_landmarks)} if right_hand_landmarks is not None else {}

    # Fill in the unified matrix block by block
    current_idx = 0
    indices_pos = {}

    if pose_landmarks is not None:
        pose_size = len(pose_landmarks)
        unified_adj[
            current_idx : current_idx + pose_size,
            current_idx : current_idx + pose_size,
        ] = adj_matrices["pose"]
        indices_pos["pose"] = (current_idx, current_idx + pose_size)
        current_idx += pose_size

    if face_landmarks is not None:
        face_size = len(face_landmarks)
        unified_adj[
            current_idx : current_idx + face_size,
            current_idx : current_idx + face_size,
        ] = adj_matrices["face"]
        indices_pos["face"] = (current_idx, current_idx + face_size)
        current_idx += face_size

    if left_hand_landmarks is not None:
        hand_size = len(left_hand_landmarks)
        unified_adj[
            current_idx : current_idx + hand_size,
            current_idx : current_idx + hand_size,
        ] = adj_matrices["left_hand"]
        indices_pos["left_hand"] = (current_idx, current_idx + hand_size)
        current_idx += hand_size

    if right_hand_landmarks is not None:
        hand_size = len(right_hand_landmarks)
        unified_adj[
            current_idx : current_idx + hand_size,
            current_idx : current_idx + hand_size,
        ] = adj_matrices["right_hand"]
        indices_pos["right_hand"] = (current_idx, current_idx + hand_size)

    # Add inter-body-part connections
    # Connect hands to pose (wrists)
    if pose_landmarks is not None:
        hand_wrist_orig = 0  # Hand wrist is always index 0 in MediaPipe
        left_wrist_pose_orig = 15  # Left wrist in pose
        right_wrist_pose_orig = 16  # Right wrist in pose

        # Connect left hand to pose left wrist
        if (left_hand_landmarks is not None and
            hand_wrist_orig in left_hand_idx_map and
            left_wrist_pose_orig in pose_idx_map):

            left_hand_wrist_unified = (
                indices_pos["left_hand"][0] + left_hand_idx_map[hand_wrist_orig]
            )
            pose_left_wrist_unified = (
                indices_pos["pose"][0] + pose_idx_map[left_wrist_pose_orig]
            )
            unified_adj[left_hand_wrist_unified, pose_left_wrist_unified] = 1.0
            unified_adj[pose_left_wrist_unified, left_hand_wrist_unified] = 1.0

        # Connect right hand to pose right wrist
        if (right_hand_landmarks is not None and
            hand_wrist_orig in right_hand_idx_map and
            right_wrist_pose_orig in pose_idx_map):

            right_hand_wrist_unified = (
                indices_pos["right_hand"][0] + right_hand_idx_map[hand_wrist_orig]
            )
            pose_right_wrist_unified = (
                indices_pos["pose"][0] + pose_idx_map[right_wrist_pose_orig]
            )
            unified_adj[right_hand_wrist_unified, pose_right_wrist_unified] = 1.0
            unified_adj[pose_right_wrist_unified, right_hand_wrist_unified] = 1.0

    # Connect face to pose (nose)
    if face_landmarks is not None and pose_landmarks is not None:
        face_nose_orig = 1  # Nose tip in face mesh
        pose_nose_orig = 0  # Nose in pose

        if face_nose_orig in face_idx_map and pose_nose_orig in pose_idx_map:
            face_nose_unified = indices_pos["face"][0] + face_idx_map[face_nose_orig]
            pose_nose_unified = indices_pos["pose"][0] + pose_idx_map[pose_nose_orig]
            unified_adj[face_nose_unified, pose_nose_unified] = 1.0
            unified_adj[pose_nose_unified, face_nose_unified] = 1.0

    return unified_adj

def get_hand_mapping(frame_df: pd.DataFrame, HAND_LANDMARKS: list[int]):
    """
    Determine which labeled hand corresponds to which actual hand based on spatial proximity to wrists.

    For each detected hand (whether labeled "left_hand" or "right_hand"), determine if it's
    actually closer to the left wrist or right wrist, and return a mapping.

    Args:
        frame_df: DataFrame for a single frame with columns: type, landmark_index, x, y, z
        HAND_LANDMARKS: List of hand landmark indices to use

    Returns:
        dict: Mapping where keys are the actual hand positions ("left_hand", "right_hand")
              and values are the labels in the data ("left_hand", "right_hand", or None)
              Example: {"left_hand": "right_hand", "right_hand": "left_hand"} means swapped
                      {"left_hand": "left_hand", "right_hand": None} means only left detected
    """
    mapping: Dict[str, Optional[str]] = {"left_hand": None, "right_hand": None}

    # Get wrists from pose (landmark 15=left, 16=right)
    left_wrist_row = frame_df[
        (frame_df["type"] == "pose") & (frame_df["landmark_index"] == 15)
    ]
    right_wrist_row = frame_df[
        (frame_df["type"] == "pose") & (frame_df["landmark_index"] == 16)
    ]

    if left_wrist_row.empty or right_wrist_row.empty:
        # Can't determine without wrists, return default
        mapping["left_hand"] = (
            "left_hand" if "left_hand" in frame_df["type"].values else None
        )
        mapping["right_hand"] = (
            "right_hand" if "right_hand" in frame_df["type"].values else None
        )
        return mapping

    left_wrist = left_wrist_row.iloc[0]
    right_wrist = right_wrist_row.iloc[0]

    # Check if wrist coordinates are valid
    if (
        pd.isna(left_wrist["x"])
        or pd.isna(left_wrist["y"])
        or pd.isna(right_wrist["x"])
        or pd.isna(right_wrist["y"])
    ):
        mapping["left_hand"] = (
            "left_hand" if "left_hand" in frame_df["type"].values else None
        )
        mapping["right_hand"] = (
            "right_hand" if "right_hand" in frame_df["type"].values else None
        )
        return mapping

    # Get palm positions for both hands
    left_hand_palm_row = frame_df[
        (frame_df["type"] == "left_hand")
        & (frame_df["landmark_index"] == HAND_LANDMARKS[0])
    ]
    right_hand_palm_row = frame_df[
        (frame_df["type"] == "right_hand")
        & (frame_df["landmark_index"] == HAND_LANDMARKS[0])
    ]

    # Calculate distances for all combinations
    distances = {}

    left_palm = left_hand_palm_row.iloc[0]
    if not (pd.isna(left_palm["x"]) or pd.isna(left_palm["y"])):
        distances[("left_hand", "left_wrist")] = np.sqrt(
            (left_wrist["x"] - left_palm["x"]) ** 2
            + (left_wrist["y"] - left_palm["y"]) ** 2
        )
        distances[("left_hand", "right_wrist")] = np.sqrt(
            (right_wrist["x"] - left_palm["x"]) ** 2
            + (right_wrist["y"] - left_palm["y"]) ** 2
        )

    right_palm = right_hand_palm_row.iloc[0]
    if not (pd.isna(right_palm["x"]) or pd.isna(right_palm["y"])):
        distances[("right_hand", "left_wrist")] = np.sqrt(
            (left_wrist["x"] - right_palm["x"]) ** 2
            + (left_wrist["y"] - right_palm["y"]) ** 2
        )
        distances[("right_hand", "right_wrist")] = np.sqrt(
            (right_wrist["x"] - right_palm["x"]) ** 2
            + (right_wrist["y"] - right_palm["y"]) ** 2
        )

    # If both hands detected, find the optimal assignment
    if len(distances) == 4:
        # Compare the two possible assignments:
        # Assignment 1: left_hand -> left_wrist, right_hand -> right_wrist (no swap)
        # Assignment 2: left_hand -> right_wrist, right_hand -> left_wrist (swap)

        assignment1_cost = (
            distances[("left_hand", "left_wrist")]
            + distances[("right_hand", "right_wrist")]
        )
        assignment2_cost = (
            distances[("left_hand", "right_wrist")]
            + distances[("right_hand", "left_wrist")]
        )

        if assignment1_cost <= assignment2_cost:
            # No swap needed
            mapping["left_hand"] = "left_hand"
            mapping["right_hand"] = "right_hand"
        else:
            # Swap needed
            mapping["left_hand"] = "right_hand"
            mapping["right_hand"] = "left_hand"
    elif len(distances) == 2:
        # Only one hand detected, assign it to the closest wrist
        for labeled_hand in ["left_hand", "right_hand"]:
            if (labeled_hand, "left_wrist") in distances and (
                labeled_hand,
                "right_wrist",
            ) in distances:
                if (
                    distances[(labeled_hand, "left_wrist")]
                    < distances[(labeled_hand, "right_wrist")]
                ):
                    mapping["left_hand"] = labeled_hand
                else:
                    mapping["right_hand"] = labeled_hand

    return mapping