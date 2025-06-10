import torch
from torch.utils.data import Dataset
import logging
from collections import Counter
import numpy as np
import random # Import random for cropping

# Import from utils.py in the same directory (using absolute import)
from utils import MICRO_EMOTION_MAP, NUM_CLASSES, EMOTION_IDX_TO_NAME, get_image_path, logger

class FeatureDataset(Dataset):
    def __init__(self, target_sequence_length,
                 preloaded_data=None, # Add preloaded_data argument
                 feature_file=None,   # Make feature_file optional
                 repeats_per_label=None, 
                 is_train=False,        
                 padding_value=0.0):
        """
        Loads pre-extracted features OR uses preloaded data, potentially expands dataset 
        by repeating samples, and applies random temporal cropping.

        Args:
            target_sequence_length (int): The desired fixed length for SUB-sequences
                                          after cropping or padding.
            preloaded_data (list, optional): A list of pre-processed data dictionaries.
                                             If provided, feature_file is ignored.
            feature_file (str, optional): Path to the .pt file containing features.
                                          Used only if preloaded_data is None.
            repeats_per_label (dict, optional): Dictionary mapping class index to
                                                repeat count. Used only if is_train=True.
            is_train (bool): If True, enables repeat sampling based on repeats_per_label.
            padding_value (float): Value used for padding sequences shorter than 
                                   target_sequence_length AFTER potential cropping.
        """
        # self.feature_file = feature_file # Keep track if needed, but don't use for loading if preloaded
        self.target_sequence_length = target_sequence_length
        self.repeats_per_label = repeats_per_label if repeats_per_label else {}
        self.is_train = is_train
        self.padding_value = padding_value
        
        # 1. Use preloaded data if available, otherwise load from file
        if preloaded_data is not None:
            logger.info(f"使用预加载的数据 ({len(preloaded_data)} 个样本)。")
            self.all_original_data = preloaded_data 
        elif feature_file is not None:
            logger.info(f"未提供预加载数据，尝试从文件加载: {feature_file}")
            self.feature_file = feature_file # Store path if loaded from file
            self.all_original_data = self._load_features()
        else:
            raise ValueError("必须提供 preloaded_data 或 feature_file。")

        if not self.all_original_data:
            raise ValueError("未能获取任何特征数据。")
            
        # 2. Removed: Logic based on original_indices. Now processes all data in self.all_original_data.
        #    The splitting/subset selection should happen *before* creating the FeatureDataset instance.
        
        # 3. Prepare the final data list and expanded indices for THIS instance
        #    Iterate directly over the items in self.all_original_data
        self.data = [] # This will hold indices into self.all_original_data
        self.expanded_indices_map = [] # Maps self.data index to original_data index and repeat#
        self.labels = [] # Labels corresponding to self.data
        
        for current_idx, item in enumerate(self.all_original_data):
            # No need for bounds check here as we iterate directly
            label = item['label']
            
            if self.is_train:
                repeat_count = self.repeats_per_label.get(label, 1) # Default repeat 1 time
                for r in range(repeat_count):
                    self.data.append(current_idx) # Store index within self.all_original_data
                    self.expanded_indices_map.append((current_idx, r))
                    self.labels.append(label)
            else: # Validation/Test: No repeats
                self.data.append(current_idx) # Store index within self.all_original_data
                self.expanded_indices_map.append((current_idx, 0))
                self.labels.append(label)

        if not self.data:
            logger.warning("数据集实例为空 (可能在重复采样后仍然为空)。")
            
        self.class_counts = Counter(self.labels) # Counts based on final (expanded) data
        # Log distribution will be called from outside if needed

    # --- Static Methods for Loading/Logging --- 
    @staticmethod
    def _load_features_static(feature_file):
        """Static version of _load_features for external use."""
        try:
            logger.info(f"(Static) 加载特征文件: {feature_file}")
            # Note: Might need `weights_only=True` in future PyTorch versions for security
            # For now, keeping default behaviour.
            feature_data = torch.load(feature_file) 
            if not isinstance(feature_data, list):
                raise TypeError("特征文件格式错误：应为字典列表 (list of dictionaries)。")
            logger.info(f"(Static) 成功加载 {len(feature_data)} 个原始特征序列.")
            return feature_data
        except FileNotFoundError:
            logger.error(f"(Static) 错误：特征文件未找到于 {feature_file}")
            return None # Return None instead of raising here for flexibility
        except Exception as e:
            logger.error(f"(Static) 加载或处理特征文件时出错 {feature_file}: {e}")
            return None
            
    @staticmethod
    def _log_original_distribution_static(original_labels, 
                                        emotion_map=None, # Optional map for names
                                        num_classes_override=None): # Optional class count
        """Static version to log distribution from a list of labels."""
        logger.info("--- 原始数据集类别分布 (用于划分) ---")
        total_samples = len(original_labels)
        if total_samples == 0:
            logger.warning("提供的原始标签列表为空。")
            return
            
        # Use provided map/count or defaults
        effective_emotion_map = emotion_map if emotion_map is not None else EMOTION_IDX_TO_NAME
        effective_num_classes = num_classes_override if num_classes_override is not None else NUM_CLASSES
            
        class_counts = Counter(original_labels)
        # Iterate based on the effective number of classes to show potentially missing ones
        for idx in range(effective_num_classes):
            count = class_counts.get(idx, 0) # Use get() to handle missing classes
            label_name = effective_emotion_map.get(idx, f"Unknown_Idx_{idx}")
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.info(f"Class '{label_name}' (ID: {idx}): {count} sequences ({percentage:.2f}%)")
        logger.info(f"总原始样本数 (过滤后): {total_samples}") # Updated log message
        logger.info("-------------------------------------")
        
    def _load_features(self):
        # This method is now only called if preloaded_data is None and feature_file is provided
        if not hasattr(self, 'feature_file') or self.feature_file is None:
             logger.error("_load_features called without a feature_file path.")
             return None
        try:
            logger.info(f"加载特征文件: {self.feature_file}")
            # Consider adding weights_only=True here for safety if PyTorch version supports it
            feature_data = torch.load(self.feature_file, weights_only=None) # Explicitly None for now
            if not isinstance(feature_data, list):
                raise TypeError("特征文件格式错误：应为字典列表 (list of dictionaries)。")
            logger.info(f"成功加载 {len(feature_data)} 个原始特征序列.") # Log count of original sequences
            return feature_data
        except FileNotFoundError:
            logger.error(f"错误：特征文件未找到于 {self.feature_file}")
            raise
        except Exception as e:
            logger.error(f"加载或处理特征文件时出错 {self.feature_file}: {e}")
            raise

    def _log_distribution(self, class_map=None, num_classes=None):
        logger.info(f"--- 数据集实例 ({'训练' if self.is_train else '验证/测试'}) 类别分布 --- ")
        total_samples = len(self.labels)
        if total_samples == 0:
            logger.warning("数据集实例为空，无法显示类别分布。")
            return
            
        # Use provided map/count or defaults from the module level
        effective_map = class_map if class_map is not None else EMOTION_IDX_TO_NAME
        effective_num_classes = num_classes if num_classes is not None else NUM_CLASSES
            
        # Iterate based on the effective number of classes to show all expected classes
        for idx in range(effective_num_classes):
            count = self.class_counts.get(idx, 0) # Use get() for potentially missing classes
            label_name = effective_map.get(idx, f"Unknown_Idx_{idx}")
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            # Determine repeats applied (only relevant for training set)
            repeats = self.repeats_per_label.get(idx, 1) if self.is_train else 1
            logger.info(f"Class '{label_name}' (ID: {idx}): {count} samples ({percentage:.2f}%) - Repeats applied: {repeats}")
            
        logger.info(f"总样本数 (含重复/采样): {total_samples}") # Updated log text slightly
        logger.info("--------------------------------------------------")

    def get_sampler_weights(self): # Renamed for clarity
        """Calculates weights for WeightedRandomSampler based on the expanded labels."""
        if not self.is_train:
            logger.warning("Sampler weights requested for non-training dataset instance.")
            return None
            
        num_samples = len(self.labels)
        if num_samples == 0:
            return None, 0
            
        class_weights_calc = {cls: 1.0 / count if count > 0 else 0 
                              for cls, count in self.class_counts.items()}
                              
        # Need to handle classes potentially missing from NUM_CLASSES if dataset is small
        # However, sampler expects weights for *every* sample in this dataset instance.
        sample_weights = [class_weights_calc.get(label, 0) for label in self.labels]
        
        logger.info(f"为 WeightedRandomSampler 计算得到的样本权重数量: {len(sample_weights)}")
        # logger.debug(f"Sample weights snippet: {sample_weights[:10]}") # Optional debug
        return sample_weights, num_samples

    def __len__(self): # Returns the length of the possibly expanded dataset
        return len(self.data)

    def __getitem__(self, idx):
        # 'idx' is an index into self.data (the potentially expanded list)
        # We need the corresponding index into self.all_original_data
        original_data_idx = self.data[idx] 
        item = self.all_original_data[original_data_idx] # Access the correct item from the instance's data
        full_feature_tensor = item['feature'] # Shape: (T_original, feature_dim)
        label = item['label'] # Get the label (should be the remapped one)
        
        # --- Cropping/Padding logic remains the same --- 
        T_original = full_feature_tensor.shape[0]
        feature_dim = full_feature_tensor.shape[1]
        T_target = self.target_sequence_length
        
        if T_original >= T_target:
            max_start_idx = T_original - T_target
            start_idx = random.randint(0, max_start_idx) 
            sub_sequence = full_feature_tensor[start_idx : start_idx + T_target, :]
        else: 
            sub_sequence = torch.full((T_target, feature_dim), 
                                      self.padding_value,
                                      dtype=full_feature_tensor.dtype)
            sub_sequence[:T_original, :] = full_feature_tensor

        return sub_sequence, label

# Example Usage (for testing)
if __name__ == '__main__':
    # Assume features have been extracted to this file
    feature_file_path = '18_2/features/casme2_cnn_features_balanced.pt'
    target_len = 64

    print(f"--- Testing FeatureDataset (Target Length: {target_len}) ---")
    if not os.path.exists(feature_file_path):
        print(f"错误: 特征文件 {feature_file_path} 未找到. 请先运行 extract_features.py.")
    else:
        try:
            feature_dataset = FeatureDataset(target_sequence_length=target_len)
            
            if len(feature_dataset) > 0:
                print(f"数据集大小: {len(feature_dataset)}")
                # Get a sample
                sample_feature, sample_label = feature_dataset[0]
                print(f"样本 0 特征形状: {sample_feature.shape}") # Should be (target_len, feature_dim)
                print(f"样本 0 标签: {sample_label}")
                assert sample_feature.shape[0] == target_len
                assert isinstance(sample_label, int)
                
                # Check another sample to see potential padding/truncation difference
                sample_feature_5, sample_label_5 = feature_dataset[5]
                print(f"样本 5 特征形状: {sample_feature_5.shape}")
                print(f"样本 5 标签: {sample_label_5}")
                assert sample_feature_5.shape[0] == target_len
                
                # Test class weight calculation
                weights = feature_dataset.get_class_weights()
                print(f"计算得到的类别权重: {weights}")
                assert len(weights) == len(feature_dataset.class_counts) # Should match number of classes found
                
                print("\nFeatureDataset 测试通过.")
            else:
                print("特征数据集已创建但为空。")
                
        except Exception as e:
            print(f"测试 FeatureDataset 时出错: {e}") 