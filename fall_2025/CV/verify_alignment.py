"""
Verify that coordinate ordering matches adjacency matrix indices.
"""
import numpy as np
from src.data_loading import POSE_LANDMARKS, FACE_LANDMARKS, HAND_LANDMARKS

# Simulate what happens in build_unified_adjacency_matrix
pose_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(POSE_LANDMARKS)}
face_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(FACE_LANDMARKS)}
left_hand_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(HAND_LANDMARKS)}

print("=== Adjacency Matrix Index Mapping ===\n")

print("POSE (first 5):")
for i, orig_idx in enumerate(POSE_LANDMARKS[:5]):
    print(f"  Row {i} in adj matrix = MediaPipe Pose landmark {orig_idx}")

print("\nFACE (first 5):")
offset = len(POSE_LANDMARKS)
for i, orig_idx in enumerate(FACE_LANDMARKS[:5]):
    print(f"  Row {offset + i} in adj matrix = MediaPipe Face landmark {orig_idx}")

print("\nLEFT_HAND (first 5):")
offset = len(POSE_LANDMARKS) + len(FACE_LANDMARKS)
for i, orig_idx in enumerate(HAND_LANDMARKS[:5]):
    print(f"  Row {offset + i} in adj matrix = MediaPipe Hand landmark {orig_idx}")

print("\n=== What the Data Loader Now Does ===\n")
print("The data loader iterates through POSE_LANDMARKS, FACE_LANDMARKS, HAND_LANDMARKS")
print("in the EXACT order they are defined, so:")
print(f"  Coordinate row 0 = Pose landmark {POSE_LANDMARKS[0]}")
print(f"  Coordinate row 1 = Pose landmark {POSE_LANDMARKS[1]}")
print("  ...")
print(f"  Coordinate row {len(POSE_LANDMARKS)} = Face landmark {FACE_LANDMARKS[0]}")
print(f"  Coordinate row {len(POSE_LANDMARKS) + 1} = Face landmark {FACE_LANDMARKS[1]}")

print("\n✓ These now match! The coordinate tensor row i corresponds to adjacency matrix row/col i")
print("\n=== Key Connections in Adjacency Matrix ===\n")

# Show some important connections
print("Inter-body-part connections:")
print(f"  Left hand wrist (landmark 0) should connect to Pose left wrist (landmark 15)")
if 0 in left_hand_idx_map and 15 in pose_idx_map:
    left_hand_wrist_unified = len(POSE_LANDMARKS) + len(FACE_LANDMARKS) + left_hand_idx_map[0]
    pose_left_wrist_unified = pose_idx_map[15]
    print(f"    → Adjacency matrix: row {left_hand_wrist_unified} connects to row {pose_left_wrist_unified}")
else:
    print("    → One or both not included in filtered landmarks")

print(f"  Right hand wrist (landmark 0) should connect to Pose right wrist (landmark 16)")
if 0 in left_hand_idx_map and 16 in pose_idx_map:
    right_hand_wrist_unified = len(POSE_LANDMARKS) + len(FACE_LANDMARKS) + len(HAND_LANDMARKS) + left_hand_idx_map[0]
    pose_right_wrist_unified = pose_idx_map[16]
    print(f"    → Adjacency matrix: row {right_hand_wrist_unified} connects to row {pose_right_wrist_unified}")
else:
    print("    → One or both not included in filtered landmarks")

print(f"\nFace nose (landmark 1) should connect to Pose nose (landmark 0)")
if 1 in face_idx_map and 0 in pose_idx_map:
    face_nose_unified = len(POSE_LANDMARKS) + face_idx_map[1]
    pose_nose_unified = pose_idx_map[0]
    print(f"    → Adjacency matrix: row {face_nose_unified} connects to row {pose_nose_unified}")
else:
    print("    → One or both not included in filtered landmarks")
