import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import logging
import random

# Import from utils.py in the same directory
from .utils import MICRO_EMOTION_MAP, NUM_CLASSES, EMOTION_IDX_TO_NAME, get_image_path, logger

class MicroExpressionDataset(Dataset):
    def __init__(self, metadata_df, data_root, 
                 mode='sequence', # 'sequence' for LSTM, 'single_frame' for CNN
                 sequence_length=32, # Used in 'sequence' mode
                 image_size=(128, 128), # Target image size (h, w)
                 spatial_transform=None, # Applied per frame
                 temporal_transform=None # Applied per sequence (Not used initially)
                 ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing dataset metadata (e.g., CASME2).
            data_root (str): Root directory of the image data.
            mode (str): 'sequence' to load fixed-length sequences, 
                        'single_frame' to load individual frames.
            sequence_length (int): Number of frames per sequence in 'sequence' mode.
            image_size (tuple): Target (height, width) for each frame.
            spatial_transform (callable, optional): Transform applied to each individual frame.
            temporal_transform (callable, optional): Transform applied to the whole sequence.
        """
        self.metadata = metadata_df.copy()
        self.data_root = data_root
        self.mode = mode
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.class_counts = {i: 0 for i in range(NUM_CLASSES)}
        self.samples = self._create_samples()

        self._log_class_distribution()

    def _create_samples(self):
        samples = []
        skipped_count = 0
        
        # Pre-calculate label indices and apex frames for faster access
        self.metadata['label_idx'] = -1
        self.metadata['apex_frame_int'] = -1
        
        valid_indices = []
        for index, row in self.metadata.iterrows():
            try:
                label_str = str(row['Estimated Emotion']).lower()
                if label_str in MICRO_EMOTION_MAP:
                    label_idx = MICRO_EMOTION_MAP[label_str]
                    apex_frame = int(row['ApexFrame'])
                    self.metadata.loc[index, 'label_idx'] = label_idx
                    self.metadata.loc[index, 'apex_frame_int'] = apex_frame
                    valid_indices.append(index)
                else:
                    # logger.warning(f"Skipping sample {row['Subject']}/{row['Filename']}: Unknown emotion '{row['Estimated Emotion']}'")
                    skipped_count += 1
            except (ValueError, TypeError, KeyError) as e:
                # logger.warning(f"Skipping sample due to invalid data at index {index}: {e} (Row: {row.to_dict()})")
                skipped_count += 1
                
        self.metadata = self.metadata.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Filtered metadata: {len(self.metadata)} samples remaining after checking labels/apex. Skipped {skipped_count}.")

        # Create sample list based on mode
        for index, row in self.metadata.iterrows():
            subject = row['Subject']
            filename = row['Filename']
            label_idx = row['label_idx']
            apex_frame = row['apex_frame_int']
            
            # Find available frames for this sample to determine actual range
            min_frame, max_frame = self._find_available_frames(subject, filename)
            if min_frame is None:
                # logger.warning(f"Skipping sample {subject}/{filename}: No frames found in directory.")
                skipped_count += 1
                continue
                
            num_available_frames = max_frame - min_frame + 1

            if self.mode == 'sequence':
                # --- MODIFIED LOGIC --- 
                # If sequence_length is specified, check if enough frames are available.
                # If sequence_length is None (or not strictly positive), we intend to load the full sequence.
                should_add_sample = False
                if self.sequence_length is not None and self.sequence_length > 0:
                    if num_available_frames >= self.sequence_length:
                        should_add_sample = True
                    else:
                        # logger.warning(f"Skipping sequence {subject}/{filename}: Not enough frames ({num_available_frames} < {self.sequence_length}).")
                        skipped_count += 1
                elif num_available_frames > 0: # sequence_length is None or invalid, load full sequence if frames exist
                    should_add_sample = True
                else:
                    # logger.warning(f"Skipping sequence {subject}/{filename}: No frames found ({num_available_frames}).")
                    skipped_count += 1
                    
                if should_add_sample:
                    # Store metadata index, label, and frame range for sequence loading
                    samples.append({
                        'metadata_index': index,
                        'subject': subject,
                        'filename': filename,
                        'label': label_idx,
                        'min_frame': min_frame,
                        'max_frame': max_frame,
                        # 'apex_frame': apex_frame # Keep original apex if needed later
                    })
                    self.class_counts[label_idx] += 1 # Count sequence samples

            elif self.mode == 'single_frame':
                # For single frame mode, add each available frame as a sample
                for frame_num in range(min_frame, max_frame + 1):
                    samples.append({
                        'metadata_index': index, # Keep reference to original row if needed
                        'subject': subject,
                        'filename': filename,
                        'label': label_idx,
                        'frame_num': frame_num
                    })
                    # Note: Class counts here reflect frames, not sequences
                    self.class_counts[label_idx] += 1 # Count each frame for single_frame mode distribution
            
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose 'sequence' or 'single_frame'.")

        logger.info(f"Created {len(samples)} samples for mode '{self.mode}'. Total skipped: {skipped_count}.")
        if len(samples) == 0:
             logger.error("No samples were created! Check data paths, metadata, and mode settings.")
             
        return samples

    def _find_available_frames(self, subject, filename):
        """Scans the directory to find the actual min and max frame numbers."""
        # Construct the potential directory path
        try:
            sub_folder = f"sub{int(subject):02d}"
        except (ValueError, TypeError):
             if isinstance(subject, str) and subject.startswith("sub") and subject[3:].isdigit():
                 sub_folder = subject
             elif isinstance(subject, str) and subject.isdigit():
                  sub_folder = f"sub{int(subject):02d}"
             else:
                 sub_folder = str(subject)
                 
        dir_path1 = os.path.join(self.data_root, sub_folder, filename)
        dir_path2 = os.path.join(self.data_root, filename) # Fallback if no subject folder
        
        dir_path = dir_path1 if os.path.isdir(dir_path1) else (dir_path2 if os.path.isdir(dir_path2) else None)

        if dir_path is None or not os.path.isdir(dir_path):
            # logger.debug(f"Directory not found for {subject}/{filename}. Looked in {dir_path1}, {dir_path2}")
            return None, None

        min_frame = float('inf')
        max_frame = float('-inf')
        found_frames = False
        try:
            for img_filename in os.listdir(dir_path):
                if img_filename.lower().startswith('img') and img_filename.lower().endswith('.jpg'):
                    try:
                        frame_num = int(img_filename[3:-4]) # Extract number between 'img' and '.jpg'
                        min_frame = min(min_frame, frame_num)
                        max_frame = max(max_frame, frame_num)
                        found_frames = True
                    except ValueError:
                        # logger.debug(f"Could not parse frame number from {img_filename} in {dir_path}")
                        continue # Skip files that don't match the naming convention
        except FileNotFoundError:
             # logger.error(f"Error listing directory (previously checked exists): {dir_path}")
             return None, None

        if not found_frames:
            # logger.warning(f"No valid 'imgXXX.jpg' files found in {dir_path}")
            return None, None

        return min_frame, max_frame

    def _load_frame(self, subject, filename, frame_num):
        """Loads, converts to grayscale, resizes, and normalizes a single frame."""
        img_path = get_image_path(subject, filename, frame_num, self.data_root)
        if not os.path.exists(img_path):
             # logger.warning(f"Frame not found at calculated path: {img_path}. Returning None.")
             return None # Return None if frame doesn't exist
        try:
            img = Image.open(img_path).convert('L') # Convert to grayscale like the original project
            img = img.resize(self.image_size[::-1]) # PIL resize takes (width, height)
            img_np = np.array(img, dtype=np.float32)
            img_np /= 255.0 # Normalize to [0, 1]
            # Add channel dimension: (H, W) -> (H, W, 1)
            img_np = np.expand_dims(img_np, axis=-1)
            return img_np
        except Exception as e:
            logger.error(f"Error loading frame {img_path}: {e}")
            return None

    def _log_class_distribution(self):
        logger.info(f"--- Class Distribution (Mode: {self.mode}) ---")
        total_samples = sum(self.class_counts.values())
        if total_samples == 0:
            logger.warning("No samples loaded, cannot show class distribution.")
            return
            
        for idx, count in sorted(self.class_counts.items()):
            label_name = EMOTION_IDX_TO_NAME.get(idx, f"Unknown_Idx_{idx}")
            percentage = (count / total_samples * 100)
            logger.info(f"Class '{label_name}' (ID: {idx}): {count} samples ({percentage:.2f}%)")
        logger.info("---------------------------------------------")
        
    def get_class_counts(self):
        """Returns the count of samples per class based on the current mode."""
        return self.class_counts
        
    def get_labels(self):
         """Returns a list of labels for all samples in the current mode."""
         return [s['label'] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        label = sample_info['label']

        if self.mode == 'single_frame':
            subject = sample_info['subject']
            filename = sample_info['filename']
            frame_num = sample_info['frame_num']
            
            frame_data = self._load_frame(subject, filename, frame_num)
            if frame_data is None:
                 # Return None if frame loading failed, collate_fn should handle this
                 # logger.debug(f"Returning None for sample index {idx} (single_frame mode) due to loading error.")
                 return None, None 

            # Apply spatial transform if provided
            if self.spatial_transform:
                # Transform expects (C, H, W) or PIL image. Our data is (H, W, C)
                # Convert to (C, H, W) for torchvision transforms
                frame_data_tensor = torch.from_numpy(frame_data.transpose((2, 0, 1))) # HWC to CHW
                frame_data_tensor = self.spatial_transform(frame_data_tensor)
            else:
                 frame_data_tensor = torch.from_numpy(frame_data.transpose((2, 0, 1))) # HWC to CHW
            
            return frame_data_tensor, label

        elif self.mode == 'sequence':
            subject = sample_info['subject']
            filename = sample_info['filename']
            min_frame = sample_info['min_frame']
            max_frame = sample_info['max_frame']
            num_available = max_frame - min_frame + 1
            
            sequence_frames = []
            valid_sequence = True
            target_frame_nums = []
            
            # --- MODIFIED LOGIC --- 
            if self.sequence_length is not None and self.sequence_length > 0:
                # Sample fixed-length sequence (like original project / Stage 1 training if needed)
                if num_available < self.sequence_length:
                    # logger.warning(f"Sequence {subject}/{filename} has only {num_available} frames, less than required {self.sequence_length}. Skipping in getitem.")
                    return None, None, None # Return None tuple for sequence, label, info
                start_offset = random.randint(0, num_available - self.sequence_length)
                start_frame_num = min_frame + start_offset
                target_frame_nums = range(start_frame_num, start_frame_num + self.sequence_length)
            elif num_available > 0:
                # Load the full sequence (for feature extraction)
                target_frame_nums = range(min_frame, max_frame + 1)
            else:
                 # No frames available, shouldn't happen if _create_samples worked correctly
                 # logger.warning(f"Sequence {subject}/{filename} has no available frames in getitem. Skipping.")
                 return None, None, None
            
            # Load the target frames
            for frame_num in target_frame_nums:
                frame_data = self._load_frame(subject, filename, frame_num)
                if frame_data is None:
                    # logger.warning(f"Missing frame {frame_num} in sequence {subject}/{filename}. Invalidating sequence.")
                    valid_sequence = False
                    break 
                sequence_frames.append(frame_data) # Append numpy array (H, W, C)
            
            if not valid_sequence or not sequence_frames: # Check if sequence_frames is empty
                # logger.debug(f"Returning None for sample index {idx} (sequence mode) due to invalid sequence or missing frames.")
                return None, None, None

            # Stack frames into a sequence: list of (H, W, C) -> (T, H, W, C)
            sequence_np = np.stack(sequence_frames, axis=0) 
            current_sequence_length = sequence_np.shape[0] # Actual length T

            # Apply spatial transforms frame-by-frame (if any)
            if self.spatial_transform:
                transformed_frames = []
                for frame_idx in range(current_sequence_length):
                    frame_tensor = torch.from_numpy(sequence_np[frame_idx].transpose((2, 0, 1))) # HWC to CHW
                    transformed_frame = self.spatial_transform(frame_tensor)
                    transformed_frames.append(transformed_frame)
                sequence_tensor = torch.stack(transformed_frames, dim=0) # Stack creates (T, C, H, W)
            else:
                 sequence_tensor = torch.from_numpy(sequence_np.transpose((0, 3, 1, 2))) # THWC to TCHW

            # Apply temporal transforms (if any) - expects (T, C, H, W)
            # if self.temporal_transform:
            #    sequence_tensor = self.temporal_transform(sequence_tensor)

            # Return sample_info as well for feature extraction script
            return sequence_tensor, label, sample_info

        else:
            # Should not happen based on constructor check
            raise RuntimeError(f"Invalid dataset mode encountered in __getitem__: {self.mode}")

# Example Usage (for testing)
if __name__ == '__main__':
    # Create a dummy metadata DataFrame 
    # Replace with actual path to your metadata file
    metadata_path = 'D:/pycharm2/PythonProject2/data/CASME2_Metadata.xlsx' 
    data_root_path = 'D:/pycharm2/PythonProject2/data' 
    try:
        full_metadata = pd.read_excel(metadata_path)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        exit()
        
    print("--- Testing Sequence Mode ---")
    try:
        seq_dataset = MicroExpressionDataset(full_metadata, data_root_path, mode='sequence', sequence_length=16, image_size=(128, 128))
        if len(seq_dataset) > 0:
            seq_sample, seq_label, seq_info = seq_dataset[0] # Get the first sample
            if seq_sample is not None:
                 print(f"Sequence sample shape: {seq_sample.shape}") # Should be (T, C, H, W) = (16, 1, 128, 128)
                 print(f"Sequence label: {seq_label} ({EMOTION_IDX_TO_NAME.get(seq_label)})")
                 assert seq_sample.shape == (16, 1, 128, 128)
                 assert isinstance(seq_label, int)
                 print("Sequence mode test passed basic checks.")
            else:
                print("Sequence dataset created, but first sample failed to load.")
        else:
            print("Sequence dataset created, but it is empty. Check data paths and sequence length requirements.")
    except Exception as e:
         print(f"Error during Sequence Mode test: {e}")

    print("\n--- Testing Single Frame Mode ---")
    try:
        frame_dataset = MicroExpressionDataset(full_metadata, data_root_path, mode='single_frame', image_size=(128, 128))
        if len(frame_dataset) > 0:
            frame_sample, frame_label = frame_dataset[0] # Get the first frame sample
            if frame_sample is not None:
                 print(f"Frame sample shape: {frame_sample.shape}") # Should be (C, H, W) = (1, 128, 128)
                 print(f"Frame label: {frame_label} ({EMOTION_IDX_TO_NAME.get(frame_label)})")
                 assert frame_sample.shape == (1, 128, 128)
                 assert isinstance(frame_label, int)
                 print("Single Frame mode test passed basic checks.")
            else:
                 print("Frame dataset created, but first sample failed to load.")
        else:
             print("Frame dataset created, but it is empty. Check data paths.")
             
    except Exception as e:
         print(f"Error during Single Frame Mode test: {e}") 