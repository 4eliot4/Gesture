import numpy as np
import torch
import random
from scipy.interpolate import interp1d
import pandas as pd
from typing import Union, Dict, List, Callable, Optional
import argparse
from pathlib import Path


class TemporalAugmentation:
    """Temporal augmentations for landmark sequences."""

    @staticmethod
    def random_resample(result, min_scale=0.5, max_scale=1.5):
        """
        Randomly resample the sequence to 0.5x ~ 1.5x of original length.

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', etc.
            min_scale: Minimum scaling factor (0.5 = half speed)
            max_scale: Maximum scaling factor (1.5 = 1.5x speed)

        Returns:
            Resampled result with same structure
        """
        scale = random.uniform(min_scale, max_scale)
        
        # Work with landmarks tensor
        landmarks = result['landmarks']  # Shape: (num_frames, num_landmarks, 3)
        original_length = landmarks.shape[0]
        new_length = max(1, int(original_length * scale))
        
        if new_length == original_length:
            return result
        
        # Convert tensor to numpy for interpolation
        landmarks_np = landmarks.numpy()
        
        # Interpolate along time dimension
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        resampled_landmarks = []
        for landmark_idx in range(landmarks_np.shape[1]):  # For each landmark point
            resampled_point = []
            for dim in range(3):  # x, y, z
                interp_func = interp1d(old_indices, landmarks_np[:, landmark_idx, dim], kind='linear')
                resampled_dim = interp_func(new_indices)
                resampled_point.append(resampled_dim)
            resampled_landmarks.append(np.array(resampled_point).T)
        
        resampled_landmarks = np.stack(resampled_landmarks, axis=1)  # (new_length, num_landmarks, 3)
        
        # Create new result with resampled landmarks
        result_resampled = result.copy()
        result_resampled['landmarks'] = torch.from_numpy(resampled_landmarks).float()
        result_resampled['num_frames'] = torch.tensor(new_length, dtype=torch.long)

        # If parquet metadata exists, update frames to match new length and
        # clear row_id_map because original row_ids no longer map to resampled frames.
        if 'parquet_meta' in result_resampled:
            meta = result_resampled['parquet_meta'].copy()
            meta['frames'] = list(range(new_length))
            # keep type_order and counts as-is, but drop any row_id mapping
            meta['row_id_map'] = {}
            result_resampled['parquet_meta'] = meta

        return result_resampled



    @staticmethod
    def random_masking(result, mask_prob=0.15, mask_length=5):
        """
        Randomly mask out temporal segments by setting landmarks to zero.

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', etc.
            mask_prob: Probability of starting a mask at any frame
            mask_length: Length of consecutive frames to mask

        Returns:
            Masked result
        """
        landmarks = result['landmarks'].clone()  # Shape: (num_frames, num_landmarks, 3)
        seq_length = landmarks.shape[0]
        
        # Determine mask positions
        i = 0
        while i < seq_length:
            if random.random() < mask_prob:
                # Mask this segment
                for j in range(i, min(i + mask_length, seq_length)):
                    landmarks[j] = torch.zeros_like(landmarks[j])
                i += mask_length
            else:
                i += 1
        
        result_masked = result.copy()
        result_masked['landmarks'] = landmarks
        return result_masked


class SpatialAugmentation:
    """Spatial augmentations for landmark coordinates."""

    @staticmethod
    def horizontal_flip(result):
        """
        Flip landmarks horizontally.

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', etc.

        Returns:
            Horizontally flipped result
        """
        landmarks = result['landmarks'].clone()  # Shape: (num_frames, num_landmarks, 3)
        landmarks[..., 0] = 1-landmarks[..., 0]  # Flip x-coordinate
        
        result_flipped = result.copy()
        result_flipped['landmarks'] = landmarks
        return result_flipped

    @staticmethod
    def random_affine(result, scale_range=(0.8, 1.2), shift_range=0.1,
                      rotate_range=15, shear_range=10):
        """
        Apply random affine transformation (scale, shift, rotate, shear).

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', etc.
            scale_range: (min, max) uniform scaling factor
            shift_range: Maximum shift in each direction (normalized coordinates)
            rotate_range: Maximum rotation in degrees
            shear_range: Maximum shear in degrees

        Returns:
            Transformed result
        """
        # Random parameters
        scale = random.uniform(*scale_range)
        shift_x = random.uniform(-shift_range, shift_range)
        shift_y = random.uniform(-shift_range, shift_range)
        shift_z = random.uniform(-shift_range, shift_range)
        rotate_deg = random.uniform(-rotate_range, rotate_range)
        shear_deg = random.uniform(-shear_range, shear_range)

        # Convert to radians
        theta = np.radians(rotate_deg)
        shear = np.radians(shear_deg)

        # Build affine transformation matrix (2D rotation + shear + scale)
        # For 3D, we'll rotate around z-axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Combined transformation matrix for x-y plane
        transform_matrix = np.array([
            [scale * cos_theta, -scale * sin_theta + shear],
            [scale * sin_theta, scale * cos_theta]
        ])

        landmarks = result['landmarks'].clone()  # Shape: (num_frames, num_landmarks, 3)
        landmarks_np = landmarks.numpy()
        
        # Apply transformation to each frame and landmark
        for frame_idx in range(landmarks_np.shape[0]):
            for landmark_idx in range(landmarks_np.shape[1]):
                point = np.array([landmarks_np[frame_idx, landmark_idx, 0], 
                                 landmarks_np[frame_idx, landmark_idx, 1]])
                transformed_point = transform_matrix @ point
                landmarks_np[frame_idx, landmark_idx, 0] = transformed_point[0] + shift_x
                landmarks_np[frame_idx, landmark_idx, 1] = transformed_point[1] + shift_y
                landmarks_np[frame_idx, landmark_idx, 2] = landmarks_np[frame_idx, landmark_idx, 2] * scale + shift_z
        
        result_transformed = result.copy()
        result_transformed['landmarks'] = torch.from_numpy(landmarks_np).float()
        return result_transformed

    @staticmethod
    def random_cutout(result, cutout_prob=0.2, num_landmarks=5):
        """
        Randomly set some landmarks to zero (spatial masking).

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', etc.
            cutout_prob: Probability of applying cutout
            num_landmarks: Number of random landmarks to zero out per frame

        Returns:
            Data with random landmarks zeroed out
        """

        landmarks = result['landmarks'].clone()  # Shape: (num_frames, num_landmarks, 3)
        num_total_landmarks = landmarks.shape[1]
        
        # For each frame, randomly select landmarks to zero out
        for frame_idx in range(landmarks.shape[0]):
            indices_to_zero = random.sample(range(num_total_landmarks),
                                           min(num_landmarks, num_total_landmarks))
            for idx in indices_to_zero:
                landmarks[frame_idx, idx] = torch.zeros(3)
        
        result_cutout = result.copy()
        result_cutout['landmarks'] = landmarks
        return result_cutout


class AugmentationPipeline:
    """
    Combine multiple augmentations into a pipeline.
    """

    def __init__(self, temporal_aug_prob=0.5, spatial_aug_prob=0.5):
        """
        Args:
            temporal_aug_prob: Probability of applying temporal augmentations
            spatial_aug_prob: Probability of applying spatial augmentations
        """
        self.temporal_aug_prob = temporal_aug_prob
        self.spatial_aug_prob = spatial_aug_prob

    def __call__(self, result):
        """
        Apply augmentations to result.

        Args:
            result: Dictionary with 'landmarks' (tensor), 'adjacency', 'label', 'num_frames', 
                   'video_name', 'gloss'

        Returns:
            Augmented result with same structure
        """
        data = result

        # If the input is parquet (path or DataFrame) or a dict containing parquet, convert it
        if 'landmarks' not in data:
            try:
                if isinstance(data, (pd.DataFrame, str)):
                    data = parquet_to_result(data)
                elif isinstance(data, dict) and 'parquet' in data:
                    data = parquet_to_result(data['parquet'])
            except Exception as e:
                raise ValueError(f"Unable to parse parquet input for augmentation: {e}")

        # Temporal augmentations
        if random.random() < self.temporal_aug_prob:
            # Random resample
            if random.random() < 0.5:
                data = TemporalAugmentation.random_resample(data)

            # Random masking
            if random.random() < 0.3:
                data = TemporalAugmentation.random_masking(data)

        # Spatial augmentations
        if random.random() < self.spatial_aug_prob:
            # Horizontal flip
            if random.random() < 0.5:
                data = SpatialAugmentation.horizontal_flip(data)

            # Random affine
            if random.random() < 0.7:
                data = SpatialAugmentation.random_affine(data)

            # Random cutout
            if random.random() < 0.3:
                data = SpatialAugmentation.random_cutout(data)

        return data


def parquet_to_result(parquet: Union[str, pd.DataFrame]):
    """
    Parse a parquet file (or DataFrame) with columns
    `frame,row_id,type,landmark_index,x,y,z` into a result dict
    containing 'landmarks' (torch.Tensor shape: [num_frames, num_landmarks, 3])
    and 'num_frames'. Preserves first-seen frame order and stores original
    rows for a round-trip that keeps the original row order.
    """
    if isinstance(parquet, str):
        df = pd.read_parquet(parquet)
    elif isinstance(parquet, pd.DataFrame):
        df = parquet.copy()
    else:
        raise TypeError("parquet must be a filepath or pandas.DataFrame")

    required = {"frame", "row_id", "type", "landmark_index", "x", "y", "z"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Parquet/DataFrame is missing required columns: {required - set(df.columns)}")

    # Determine types and preferred ordering (preserve first-seen order when possible)
    types_present = df['type'].drop_duplicates().tolist()
    preferred = ["face", "left_hand", "pose", "right_hand"]
    # keep preferred order but preserve appearance order for non-preferred
    type_order = [t for t in preferred if t in types_present] + [t for t in types_present if t not in preferred]

    # compute counts per type using groupby (vectorized)
    counts = {t: 0 for t in type_order}
    if not df.empty:
        max_idx = df.groupby('type')['landmark_index'].max()
        for t in type_order:
            if t in max_idx.index:
                counts[t] = int(max_idx[t]) + 1

    total_landmarks = sum(counts.values())

    # preserve frame order as first appearance in file
    frames = df['frame'].drop_duplicates().tolist()
    landmarks_array = np.zeros((len(frames), max(1, total_landmarks), 3), dtype=np.float32)

    if not df.empty:
        # Vectorized fill using categorical codes
        frame_codes = pd.Categorical(df['frame'], categories=frames).codes
        type_codes = pd.Categorical(df['type'], categories=type_order).codes
        li_vals = df['landmark_index'].to_numpy().astype(int)
        x_vals = df['x'].to_numpy().astype(float)
        y_vals = df['y'].to_numpy().astype(float)
        z_vals = df['z'].to_numpy().astype(float)

        # compute type offsets
        offsets_arr = np.array([sum(counts[t] for t in type_order[:i]) for i in range(len(type_order))], dtype=int)
        type_offset_arr = offsets_arr[type_codes]
        landmark_global_idx = type_offset_arr + li_vals

        # mask valid indices
        valid_mask = (landmark_global_idx >= 0) & (landmark_global_idx < landmarks_array.shape[1])
        if valid_mask.any():
            landmarks_array[frame_codes[valid_mask], landmark_global_idx[valid_mask], 0] = x_vals[valid_mask]
            landmarks_array[frame_codes[valid_mask], landmark_global_idx[valid_mask], 1] = y_vals[valid_mask]
            landmarks_array[frame_codes[valid_mask], landmark_global_idx[valid_mask], 2] = z_vals[valid_mask]

    meta = {
        'frames': frames,
        'type_order': type_order,
        'counts': counts,
    }

    # Store first-seen row_id per (frame, type, landmark_index) key for round-trip
    row_id_map = {}
    if not df.empty:
        keys = list(zip(df['frame'].astype(int).to_numpy(), df['type'].to_numpy(), df['landmark_index'].astype(int).to_numpy()))
        row_ids = df['row_id'].to_numpy()
        for k, rid in zip(keys, row_ids):
            if k not in row_id_map:
                row_id_map[k] = rid
    meta['row_id_map'] = row_id_map

    result = {
        'landmarks': torch.from_numpy(landmarks_array).float(),
        'num_frames': torch.tensor(len(frames), dtype=torch.long),
        'parquet_meta': meta,
    }

    return result


def result_to_parquet(result: Dict, out_path: Union[str, Path]):
    """
    Convert a `result` produced by `parquet_to_result` (or augmented version)
    back to a parquet file with columns `frame,row_id,type,landmark_index,x,y,z`.

    Args:
        result: dict with keys 'landmarks' (tensor) and 'parquet_meta'
        out_path: output file path
    """
    if 'parquet_meta' not in result:
        raise ValueError('result missing parquet_meta necessary to write parquet')

    meta = result['parquet_meta']
    frames: List[int] = meta['frames']
    type_order: List[str] = meta['type_order']
    counts: Dict[str, int] = meta['counts']
    row_id_map = meta.get('row_id_map', {})

    landmarks = result['landmarks'].numpy()  # (num_frames, total_landmarks, 3)

    # build output rows
    rows_frame = []
    rows_row_id = []
    rows_type = []
    rows_li = []
    rows_x = []
    rows_y = []
    rows_z = []

    for i, frame in enumerate(frames):
        offset = 0
        for t in type_order:
            n = counts.get(t, 0)
            for li in range(n):
                key = (int(frame), t, int(li))
                row_id = row_id_map.get(key, None)
                rows_frame.append(int(frame))
                rows_row_id.append(row_id)
                rows_type.append(t)
                rows_li.append(int(li))
                rows_x.append(float(landmarks[i, offset + li, 0]))
                rows_y.append(float(landmarks[i, offset + li, 1]))
                rows_z.append(float(landmarks[i, offset + li, 2]))
            offset += n

    out_df = pd.DataFrame({
        'frame': rows_frame,
        'row_id': rows_row_id,
        'type': rows_type,
        'landmark_index': rows_li,
        'x': rows_x,
        'y': rows_y,
        'z': rows_z,
    }, columns=['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)


# ---------- CLI and processing helpers ----------
def available_augmentations() -> Dict[str, Callable[[dict], dict]]:
    return {
        'rand_resample': TemporalAugmentation.random_resample,
        'rand_mask': TemporalAugmentation.random_masking,
        'hflip': SpatialAugmentation.horizontal_flip,
        'rand_affine': SpatialAugmentation.random_affine,
        'rand_cutout': SpatialAugmentation.random_cutout,
        'pipeline': lambda res: AugmentationPipeline()(res),
    }


def process_file(path: Path, aug_names: List[str], output_dir: Optional[Path] = None):
    """
    Process a single parquet file and save to output_dir.
    
    Args:
        path: path to input parquet file
        aug_names: list of augmentation names to apply
        output_dir: directory to save output file (defaults to same dir as input)
    
    Returns:
        path to output file
    """
    if output_dir is None:
        output_dir = path.parent
    
    df = pd.read_parquet(path)
    result = parquet_to_result(df)

    # Handle 'all' feature: expand to all augmentations except pipeline
    if 'all' in aug_names:
        all_augs = ['rand_resample', 'rand_mask', 'hflip', 'rand_affine', 'rand_cutout']
        aug_names = [a for a in aug_names if a != 'all'] + all_augs

    aug_map = available_augmentations()

    for aug in aug_names:
        if aug not in aug_map:
            raise ValueError(f"Unknown augmentation '{aug}'. Available: {list(aug_map.keys())}")
        result = aug_map[aug](result)

    # write output with suffix of joined augmentation names
    suffix = '_'.join(aug_names)
    out_name = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    out_path = output_dir / out_name.name
    
    # Ensure parquet_meta is present; if absent try to reuse original meta
    if 'parquet_meta' not in result:
        result['parquet_meta'] = parquet_to_result(df)['parquet_meta']
    result_to_parquet(result, out_path)
    return out_path


def process_directory(data_dir: Path, result_base: Path, aug_names: List[str], aug_suffix: str):
    """
    Recursively process all parquet files in a directory structure.
    Creates mirrored output directory structure with augmentation suffix.
    
    Args:
        data_dir: input directory to scan
        result_base: base result directory (root where suffixed dirs are created)
        aug_names: list of augmentation names to apply
        aug_suffix: suffix string from joined aug_names
    
    Returns:
        list of output file paths
    """
    written = []
    
    # Create output directory with suffix appended to the input directory name
    dir_suffix = f"{data_dir.name}_{aug_suffix}"
    output_dir = result_base / dir_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for entry in data_dir.iterdir():
        if entry.is_file() and entry.suffix in ('.parquet', '.pq'):
            out_path = process_file(entry, aug_names, output_dir=output_dir)
            written.append(out_path)
        elif entry.is_dir():
            # Recursively process subdirectories
            sub_results = process_directory(entry, output_dir, aug_names, aug_suffix)
            written.extend(sub_results)
    
    return written


def main():
    parser = argparse.ArgumentParser(description='Apply augmentations to parquet landmark files')
    parser.add_argument('--data', required=True, help='Path to parquet file or folder')
    parser.add_argument('--result_path', required=False, default=None,
                        help='Parent directory for output (defaults to parent of --data)')
    parser.add_argument('--augmentations', '-a', required=False, default='pipeline',
                        help='Comma-separated augmentation names, "all" for all augmentations, or "pipeline" (default)')
    args = parser.parse_args()

    data_path = Path(args.data)
    
    # Determine result_path parent: use provided one or default to parent of data
    if args.result_path:
        result_parent = Path(args.result_path)
    else:
        result_parent = data_path.parent
    
    aug_names = [s.strip() for s in args.augmentations.split(',') if s.strip()]
    aug_suffix = '_'.join(aug_names)

    if data_path.is_file():
        # Single file: output to result_parent directory
        result_parent.mkdir(parents=True, exist_ok=True)
        out = process_file(data_path, aug_names, output_dir=result_parent)
        print(f'Wrote: {out}')
    elif data_path.is_dir():
        # Directory: create suffixed directory in result_parent and recursively process
        written = process_directory(data_path, result_parent, aug_names, aug_suffix)
        print('Wrote:')
        for p in written:
            print(p)
    else:
        raise ValueError('Provided --data path is not a file or directory')


if __name__ == '__main__':
    main()