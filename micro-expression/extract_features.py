import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime

# --- 导入自定义模块 ---
# Assuming these are in the same parent directory or Python path is set up
from .utils import MICRO_EMOTION_MAP, NUM_CLASSES, EMOTION_IDX_TO_NAME, logger, setup_logging
from .dataset import MicroExpressionDataset # Make sure this dataset works in 'sequence' mode
from .transforms import create_spatial_transforms # Need the non-augmenting transforms
from .models import CNNModel # The Stage 1 model

# --- Custom Collate Function for Padding Sequences ---
def pad_collate_fn(batch):
    """Pads sequences in a batch to the maximum length in that batch."""
    # Filter out None samples (e.g., from loading errors)
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        # logger.warning("Collate function received an empty batch after filtering Nones.")
        return None # Return None if batch is empty after filtering

    # Separate sequences, labels, and sample_info
    # Assume batch items are (sequence_tensor, label, sample_info)
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sample_infos = [item[2] for item in batch]

    # Pad sequences. pad_sequence expects (T, *) shape.
    # Our sequences are (T, C, H, W)
    # We need padding_value=0.0 for float tensors
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return padded_sequences, labels_tensor, sample_infos


def extract_features(cfg):
    """Extracts features from sequences using the trained Stage 1 CNN."""
    # --- 环境设置 ---
    if cfg['environment']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['environment']['device'])
    logger.info(f"使用设备: {device}")

    # --- 加载元数据 ---
    logger.info("加载元数据...")
    metadata_file_config = cfg['data']['metadata_file']
    data_path_config = cfg['data']['data_path']
    if os.path.isabs(metadata_file_config):
        metadata_path = metadata_file_config
    else:
        logger.warning(f"元数据文件路径 '{metadata_file_config}' 是相对路径. 将与 data_path '{data_path_config}' 连接. 建议在配置中使用绝对路径.")
        metadata_path = os.path.join(data_path_config, metadata_file_config)

    try:
        metadata_df = pd.read_excel(metadata_path)
        logger.info(f"元数据加载成功，包含 {len(metadata_df)} 条记录.")
    except FileNotFoundError:
        logger.error(f"错误：元数据文件未找到于 {metadata_path}")
        return
    except Exception as e:
        logger.error(f"加载元数据时出错: {e}")
        return
        
    # --- 加载 Stage 1 CNN 模型 ---
    logger.info("加载 Stage 1 CNN 模型...")
    cnn_checkpoint_path = cfg['cnn_model']['checkpoint_path']
    input_channels = cfg['cnn_model'].get('input_channels', 1)
    
    # Initialize the model structure (num_classes doesn't matter for feature extraction)
    cnn_model = CNNModel(num_classes=NUM_CLASSES, input_channels=input_channels).to(device)
    
    try:
        # Load the state dictionary
        checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        # Handle potential nested state_dict keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
             state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        cnn_model.load_state_dict(state_dict)
        logger.info(f"成功从 {cnn_checkpoint_path} 加载 CNN 模型权重。")
    except FileNotFoundError:
        logger.error(f"错误：CNN 检查点未找到于 {cnn_checkpoint_path}")
        return
    except Exception as e:
        logger.error(f"加载 CNN 模型权重时出错: {e}")
        return

    # Set model to evaluation mode
    cnn_model.eval()

    # --- 创建数据变换 (使用带增强的版本) ---
    image_size = tuple(cfg['data'].get('image_size', [128, 128]))
    # Use is_train=False to get normalization, grayscale, resize ONLY
    # val_test_transforms = create_spatial_transforms(image_size=image_size, is_train=False)
    # 修改：应用训练时的增强变换
    train_transforms = create_spatial_transforms(image_size=image_size, is_train=True)
    logger.info(f"数据变换创建完成 (带增强). 图像大小: {image_size}")

    # --- 创建数据集实例 (序列模式) ---
    logger.info("创建数据集实例 (模式: sequence)...")
    # Ensure the dataset correctly handles 'sequence' mode and applies the transform to each frame
    try:
        full_dataset = MicroExpressionDataset(
            metadata_df=metadata_df,
            data_root=data_path_config, # Use the raw data path
            mode='sequence', # Important: Use sequence mode
            image_size=image_size, # Pass image_size for potential internal use
            # 修改：传递带增强的变换
            spatial_transform=train_transforms, # Apply basic transforms
            sequence_length=None # Let dataset return full sequence
        )
        logger.info(f"数据集创建完成. 找到 {len(full_dataset)} 个样本序列。")
    except Exception as e:
        logger.error(f"创建数据集时出错: {e}")
        return

    if len(full_dataset) == 0:
        logger.error("数据集为空，无法提取特征。")
        return

    # --- 创建 DataLoader ---
    # No weighted sampling or shuffling needed for feature extraction
    batch_size = cfg['environment'].get('batch_size', 16)
    num_workers = cfg['environment'].get('num_workers', 0) # Start with 0 for debugging
    
    # Define a collate function to handle variable length sequences if needed
    # For now, assume MicroExpressionDataset in 'sequence' mode returns padded sequences or uses some internal padding.
    # If it returns lists of tensors, a custom collate_fn will be needed. Let's assume default collate works for now.
    # TODO: Verify if MicroExpressionDataset pads sequences or if a custom collate is needed.
    # UPDATE: We know we need padding now.
    
    data_loader = DataLoader(full_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=(device == 'cuda'),
                             collate_fn=pad_collate_fn) # Use the custom collate function

    # --- 特征提取循环 ---
    logger.info("开始提取特征...")
    all_features = []
    all_labels = []
    all_sample_info = []

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(data_loader, desc="提取特征"):
            # Assuming batch structure: (sequences, labels, sample_info)
            # sequences shape: (B, T, C, H, W) - T might be variable if not padded
            # labels shape: (B,)
            # sample_info: List of dictionaries or similar
            if batch is None: continue # Handle potential errors? Dataset should filter bad samples.
            
            try:
                sequences, labels, sample_info = batch # Adjust based on actual dataset return format
                sequences = sequences.to(device)
                B, T, C, H, W = sequences.size()
                
                # Reshape for CNN: (B, T, C, H, W) -> (B*T, C, H, W)
                cnn_in = sequences.view(B * T, C, H, W)
                
                # Extract features using the modified CNN
                # Use return_features=True
                features_flat = cnn_model(cnn_in, return_features=True) # Shape: (B*T, feature_dim)
                
                # Reshape back to sequence: (B*T, feature_dim) -> (B, T, feature_dim)
                feature_dim = features_flat.size(-1)
                features_seq = features_flat.view(B, T, feature_dim)
                
                # Store results (move features to CPU to avoid accumulating GPU memory)
                all_features.extend([f.cpu() for f in features_seq]) # List of tensors (T, feature_dim)
                all_labels.extend(labels.cpu().tolist()) # List of integers
                all_sample_info.extend(sample_info) # List of dicts (e.g., {'subject': s, 'filename': f})
                
            except Exception as e:
                logger.error(f"处理批次时出错: {e}. 跳过此批次.")
                # Optionally log more details about the batch here
                continue

    logger.info(f"特征提取完成. 共处理 {len(all_features)} 个序列.")

    if not all_features:
        logger.error("未提取到任何特征，无法保存。")
        return

    # --- 保存特征 ---
    output_dir = cfg['output']['output_dir']
    output_file = cfg['output']['feature_file']
    os.makedirs(output_dir, exist_ok=True)

    # Save as a list of dictionaries or one large dictionary
    # List of dicts might be more flexible if sequences have variable length
    feature_data = []
    for i in range(len(all_features)):
        feature_data.append({
            'feature': all_features[i], # Tensor on CPU
            'label': all_labels[i],
            'subject': all_sample_info[i].get('subject', 'N/A'),
            'filename': all_sample_info[i].get('filename', 'N/A')
        })
        
    # Or save as one dictionary (if padding is handled and T is consistent)
    # feature_data_dict = {
    #     'features': torch.stack(all_features), # Requires all tensors to have the same shape (T, feature_dim)
    #     'labels': torch.tensor(all_labels),
    #     'subjects': [s['subject'] for s in all_sample_info],
    #     'filenames': [s['filename'] for s in all_sample_info]
    # }

    try:
        torch.save(feature_data, output_file)
        logger.info(f"特征已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存特征时出错: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Stage 1 CNN 模型提取序列特征")
    parser.add_argument("--config", type=str, required=True, help="指向特征提取 YAML 配置文件的路径")
    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到于 {args.config}")
        exit(1)
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        exit(1)

    # 设置日志
    log_dir = config['logging'].get('log_dir', '18_2/logs')
    exp_name = config['logging'].get('experiment_name', 'feature_extraction')
    setup_logging(log_dir, exp_name) # Make sure logger is configured before first use
    logger.info(f"配置加载自: {args.config}")
    logger.info(f"实验名称: {exp_name}")

    # 开始提取特征
    extract_features(config) 