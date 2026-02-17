import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import yaml
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shutil # Import shutil for file copying
# Import precision_recall_fscore_support and f1_score
from sklearn.metrics import precision_recall_fscore_support, f1_score

# --- 导入自定义模块 ---
# 从 .utils 导入原始的 7 类定义，但稍后会覆盖
from .utils import NUM_CLASSES as ORIGINAL_NUM_CLASSES, EMOTION_IDX_TO_NAME as ORIGINAL_EMOTION_IDX_TO_NAME, logger, setup_logging 
from .feature_dataset import FeatureDataset # Import the new dataset class
from .models import LSTMOnlyModel # Import the LSTM-only model

# --- Collate Function --- 
def simple_collate_fn(batch):
    """Collates batches, filtering out None values resulting from loading errors."""
    batch = [(data, target) for data, target in batch if data is not None and data[0] is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- 主训练函数 ---
def train_lstm_stage2(cfg, param_overrides=None):
    # --- Apply Parameter Overrides ---
    if param_overrides:
        logger.info(f"Applying parameter overrides: {param_overrides}")
        # Update relevant sections of the config dictionary
        if 'learning_rate' in param_overrides:
            cfg['training']['learning_rate'] = param_overrides['learning_rate']
        if 'weight_decay' in param_overrides:
             cfg['training']['weight_decay'] = param_overrides['weight_decay']
        if 'dropout_lstm' in param_overrides:
            cfg['model']['dropout_lstm'] = param_overrides['dropout_lstm']
        if 'dropout_fc' in param_overrides:
            cfg['model']['dropout_fc'] = param_overrides['dropout_fc']
        if 'lstm_hidden_size' in param_overrides:
             cfg['model']['lstm_hidden_size'] = param_overrides['lstm_hidden_size']
        # Add override for sadness repeats if provided
        if 'sadness_repeats' in param_overrides:
             # Ensure the balancing and repeats_per_label sections exist
             if 'balancing' not in cfg: cfg['balancing'] = {}
             if 'repeats_per_label' not in cfg['balancing']: cfg['balancing']['repeats_per_label'] = {}
             # Key 5 corresponds to the remapped sadness label
             cfg['balancing']['repeats_per_label'][4] = param_overrides['sadness_repeats'] # Use NEW index 4 for sadness
             logger.info(f"Overriding sadness repeats (label 4) to: {param_overrides['sadness_repeats']}")


    # --- 环境设置 ---
    if cfg['environment']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['environment']['device'])
    logger.info(f"使用设备: {device}")

    seed = cfg['data'].get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Create Unique Output Directory for this Run ---
    output_base_dir = cfg['logging'].get('output_dir', '18_2/stage2_lstm_output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = cfg.get('experiment_name', 'lstm_stage2')
    # Add trial ID to output path if provided (for HPO)
    trial_id = param_overrides.get('trial_id', None) if param_overrides else None
    run_dir_name = f"{exp_name}_{timestamp}" + (f"_trial_{trial_id}" if trial_id is not None else "")
    run_output_dir = os.path.join(output_base_dir, run_dir_name)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        logger.info(f"本次运行结果将保存在: {run_output_dir}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {run_output_dir} - {e}")
        # Fallback to base directory? Or exit? Let's fallback for now.
        logger.warning(f"将尝试保存结果到基础目录: {output_base_dir}")
        run_output_dir = output_base_dir 

    # --- Save Configuration File to Run Directory ---
    config_path = cfg.get('config_path') # Get path stored during loading
    if config_path and os.path.exists(config_path):
        try:
            shutil.copy2(config_path, os.path.join(run_output_dir, os.path.basename(config_path)))
            logger.info(f"配置文件已复制到: {run_output_dir}")
        except Exception as e:
            logger.error(f"复制配置文件失败 从 {config_path} 到 {run_output_dir}: {e}")
    else:
        logger.warning("在配置字典中未找到 'config_path' 或文件不存在，无法复制配置文件。")

    # It's better to save the *effective* config used in the trial
    effective_config_path = os.path.join(run_output_dir, f"effective_config_trial_{trial_id}.yaml" if trial_id else "effective_config.yaml")
    try:
        with open(effective_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Effective configuration saved to: {effective_config_path}")
    except Exception as e:
        logger.error(f"保存有效配置文件失败: {e}")

    # --- Load Feature Dataset AND Original Info for Splitting ---
    logger.info("加载预提取的特征及原始信息...")
    feature_file = cfg['data']['feature_file']
    target_seq_len = cfg['data']['target_sequence_length'] # Sub-sequence length
    repeats_cfg = cfg.get('balancing', {}).get('repeats_per_label', {}) # Use potentially overridden repeats
    logger.info(f"使用的 repeats_per_label 配置: {repeats_cfg}") # Log the effective repeats
    
    try:
        all_original_data_unfiltered = FeatureDataset._load_features_static(feature_file)
        if not all_original_data_unfiltered:
             raise ValueError("从特征文件中未加载任何原始样本数据。")
             
        # --- Filter out 'fear' (4) AND 'sadness' (5) ---
        logger.info("Filtering out 'fear' (original label 4) AND 'sadness' (original label 5)...")
        fear_label_index = 4 # Original index for fear
        sadness_label_index = 5 # Original index for sadness
        labels_to_exclude = {fear_label_index, sadness_label_index} # Set of labels to exclude
        
        all_original_data = []
        original_labels_unmapped = []
        original_subjects = []
        
        for item in all_original_data_unfiltered:
            original_label = item['label']
            if original_label not in labels_to_exclude: # Check if label is NOT in the exclude set
                all_original_data.append(item) # Keep the item
                original_labels_unmapped.append(original_label) # Keep original label for mapping
                original_subjects.append(item['subject'])

        num_filtered_samples = len(all_original_data)
        num_removed = len(all_original_data_unfiltered) - num_filtered_samples
        if num_removed > 0:
             logger.info(f"Filtered data: {num_filtered_samples} samples remaining (removed {num_removed} samples with labels in {labels_to_exclude}).")
        else:
             logger.warning(f"No samples with labels {labels_to_exclude} found to filter.")
        
        if num_filtered_samples == 0:
            logger.error("所有样本都被过滤掉了！无法继续。")
            return 0.0 # Return bad score for HPO

        # --- Remap remaining labels (0, 1, 2, 3, 6) to (0, 1, 2, 3, 4) ---
        logger.info("Remapping labels to contiguous range for 5 classes...")
        unique_original_labels = sorted(list(set(original_labels_unmapped))) # Should be [0, 1, 2, 3, 6]
        
        # Explicitly check if the unique labels are as expected
        expected_unique_labels = {0, 1, 2, 3, 6}
        if set(unique_original_labels) != expected_unique_labels:
             logger.warning(f"过滤后的唯一原始标签与预期 {expected_unique_labels} 不符: {unique_original_labels}。标签映射可能不正确。")

        # Create the mapping from old label -> new label
        label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_original_labels)}
        # Create the inverse mapping for names: new label -> old label
        inverse_label_mapping = {v: k for k, v in label_mapping.items()} 
        
        # Apply the mapping to get the final labels list for splitting/stratification
        original_labels = [label_mapping[label] for label in original_labels_unmapped]
        
        # --- Update Class Definitions for 5 Classes ---
        NUM_CLASSES = len(unique_original_labels) # Should be 5
        # Use a default name if the original index is somehow missing
        EMOTION_IDX_TO_NAME = {new_idx: ORIGINAL_EMOTION_IDX_TO_NAME.get(inverse_label_mapping[new_idx], f"Unknown_Orig_{inverse_label_mapping[new_idx]}") 
                               for new_idx in range(NUM_CLASSES)}

        logger.info(f"Labels remapped to {NUM_CLASSES} classes. New mapping: {EMOTION_IDX_TO_NAME}")
        
        # --- Update the labels within the data dictionaries themselves ---
        remapped_repeats_cfg = {} # Rebuild repeats config based on actual remaining labels and mapping
        for i in range(len(all_original_data)):
            original_label = all_original_data[i]['label'] # Still original label here
            if original_label in label_mapping: # Check if it's a label we are keeping
                new_label = label_mapping[original_label]
                all_original_data[i]['label'] = new_label # Update dict with NEW label
                
                # Check if the NEW label exists in the repeats config from cfg
                # The config should already be using new labels (0-4) based on the 5-class plan
                if new_label in repeats_cfg:
                     remapped_repeats_cfg[new_label] = repeats_cfg[new_label]
                # else: # Log if a mapped label is missing in the config's repeats? Should not happen if config is correct.
                     # logger.warning(f"New label {new_label} (from original {original_label}) not found in provided repeats_per_label config: {repeats_cfg}")
            # else: # Should not happen after filtering
                # logger.error(f"Unexpected original label {original_label} found after filtering in item {i}")
                
        # Log the effective repeats config based on remapping (should match input if config is correct)
        logger.info(f"Effective repeats_per_label config after remapping: {remapped_repeats_cfg}")
        # Use the remapped config derived from the data and mapping, ensuring consistency.
        # This overrides the potentially incorrect 'repeats_cfg' read directly from the file if the mapping changes.
        repeats_cfg_effective = remapped_repeats_cfg 

        # Log distribution using the NEW labels and NEW map
        FeatureDataset._log_original_distribution_static(original_labels, 
                                                       emotion_map=EMOTION_IDX_TO_NAME, 
                                                       num_classes_override=NUM_CLASSES)
        num_original_samples = len(original_labels) # Correct number after filtering

    except Exception as e:
        logger.error(f"无法加载、过滤或重映射数据集信息: {e}", exc_info=True)
        return 0.0 # Return bad score for HPO

    # --- 设置 K 折交叉验证 (using remapped labels) --- 
    k_folds = cfg['training'].get('k_folds', 5) 
    if k_folds <= 1:
        logger.warning(f"K-Fold 交叉验证需要 k_folds > 1，但配置为 {k_folds}。将执行简单的训练/验证划分。")
        k_folds = 1 
        skf = None
    else:
        logger.info(f"设置 Stratified Group K-Fold 交叉验证, K = {k_folds}")
        skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # --- K 折交叉验证循环 ---
    all_fold_results = [] 
    original_indices_np = np.arange(num_original_samples) # Use the filtered count
    
    # Determine the iterator (using remapped labels and original subjects)
    if skf:
        # StratifiedGroupKFold expects numpy arrays or lists
        fold_iterator = enumerate(skf.split(original_indices_np, np.array(original_labels), np.array(original_subjects)))
    else: 
        logger.info("执行单次训练/验证划分 (无交叉验证循环)。")
        train_ratio = cfg['data']['split_ratio'][0]
        val_ratio = cfg['data']['split_ratio'][1]
        split_test_size = val_ratio
        try:
            # Use remapped labels for stratification
            train_idx, val_idx, _, _ = train_test_split(
                original_indices_np, original_labels,
                test_size=split_test_size, 
                random_state=seed,
                stratify=original_labels 
            )
        except ValueError:
             logger.warning("单次划分时分层失败，进行随机划分...")
             train_idx, val_idx = train_test_split(original_indices_np, test_size=split_test_size, random_state=seed)
        fold_iterator = enumerate([(train_idx, val_idx)]) 

    # --- Get Early Stopping Config ---
    es_enabled = cfg['training'].get('early_stopping', {}).get('enabled', False)
    es_metric = cfg['training'].get('early_stopping', {}).get('metric', 'val_loss')
    es_patience = cfg['training'].get('early_stopping', {}).get('patience', 10)
    es_min_delta = cfg['training'].get('early_stopping', {}).get('min_delta', 0)
    es_mode = cfg['training'].get('early_stopping', {}).get('mode', 'min')
    if es_enabled:
        logger.info(f"Early Stopping Enabled: metric='{es_metric}', patience={es_patience}, min_delta={es_min_delta}, mode='{es_mode}'")
        if es_mode == 'min':
            best_score = float('inf')
            delta_comparison = lambda current, best: current < best - es_min_delta
        else: # mode == 'max'
            best_score = float('-inf')
            delta_comparison = lambda current, best: current > best + es_min_delta
        wait_count = 0
        stopped_epoch = -1 # Track if early stopping occurred

    # Lists to store metrics across epochs FOR THE LAST FOLD for plotting curves
    last_fold_train_losses = []
    last_fold_val_losses = []
    last_fold_train_accs = []
    last_fold_val_accs = []

    for fold, (train_idx, val_idx) in fold_iterator:
        logger.info(f"--- 开始 Fold {fold+1}/{k_folds} ---")
        
        # --- Reset Fold-specific variables ---
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        fold_val_accs = []
        best_val_loss_fold = float('inf') # Still track best loss for model saving
        best_model_path_fold = None

        # --- Reset Early Stopping state for the fold ---
        if es_enabled:
            if es_mode == 'min':
                best_score = float('inf')
            else:
                best_score = float('-inf')
            wait_count = 0
            stopped_epoch = -1 # Reset stopped epoch for this fold

        # --- 创建当前 Fold 的数据子集 ---
        # train_idx and val_idx are indices relative to the filtered & remapped all_original_data
        train_data_subset = [all_original_data[i] for i in train_idx]
        val_data_subset = [all_original_data[i] for i in val_idx]
        logger.info(f"Fold {fold+1}: 训练子集大小: {len(train_data_subset)}, 验证子集大小: {len(val_data_subset)}")

        # --- 创建当前 Fold 的数据集实例 (使用预加载的数据子集和5类repeats) ---
        train_dataset = FeatureDataset(target_sequence_length=target_seq_len,
                                       preloaded_data=train_data_subset,
                                       # Pass the EFFECTIVE repeats derived after filtering/remapping
                                       repeats_per_label=repeats_cfg_effective, 
                                       is_train=True,
                                       padding_value=cfg['data'].get('padding_value', 0.0))
        # Log distribution using the dataset instance (which now should internally use the 5-class map)
        logger.info("--- 训练数据集实例 (含重复/裁剪, 5类) ---")
        train_dataset._log_distribution(class_map=EMOTION_IDX_TO_NAME, num_classes=NUM_CLASSES) # Pass map for correct logging
        
        logger.info(f"Fold {fold+1}: 创建验证数据集实例 (仅应用裁剪/填充, 5类)...")
        val_dataset = FeatureDataset(target_sequence_length=target_seq_len,
                                     preloaded_data=val_data_subset, # Pass subset
                                     is_train=False,
                                     padding_value=cfg['data'].get('padding_value', 0.0))
        logger.info("--- 验证数据集实例 (仅裁剪/填充, 5类) ---")
        val_dataset._log_distribution(class_map=EMOTION_IDX_TO_NAME, num_classes=NUM_CLASSES) # Pass map for correct logging
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"Fold {fold+1}: 训练集或验证集为空，跳过此折。")
            continue

        # --- 处理数据集不平衡 (Sampler uses labels from train_dataset which are already remapped) ---
        sampler = None
        num_train_samples_expanded = len(train_dataset) # Get length before potential sampler
        if cfg['balancing'].get('use_weighted_sampler', True) and num_train_samples_expanded > 0:
            logger.info(f"Fold {fold+1}: 为扩展后的训练集创建 WeightedRandomSampler...")
            # Fix: Correctly unpack the two return values from get_sampler_weights
            returned_weights, returned_num_samples = train_dataset.get_sampler_weights() 
            
            # Check if weights were successfully returned (is a list)
            if isinstance(returned_weights, list) and returned_weights:
                # Use the returned number of samples for consistency
                num_train_samples_expanded = returned_num_samples 
                try:
                    # Pass the weights list directly
                    sampler = WeightedRandomSampler(returned_weights, num_samples=num_train_samples_expanded, replacement=True)
                    logger.info(f"WeightedRandomSampler 创建成功，样本权重数量: {len(returned_weights)}")
                except Exception as e:
                    logger.error(f"Fold {fold+1}: 创建 WeightedRandomSampler 时出错: {e}", exc_info=True)
                    sampler = None # Fallback to no sampler
            else:
                 logger.warning(f"Fold {fold+1}:未能从数据集中获取有效的样本权重 ({returned_weights})，不使用Sampler。")
            
        # --- 创建当前 Fold 的 DataLoader --- 
        batch_size = cfg['training']['batch_size']
        num_workers = cfg['environment'].get('num_workers', 0)
        pin_memory = cfg['environment'].get('pin_memory', True)
        
        logger.info(f"Fold {fold+1}: 创建 DataLoader... Batch Size: {batch_size}, Workers: {num_workers}")
        
        # Use shuffle=False if sampler is used, otherwise shuffle=True
        shuffle_train = sampler is None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=shuffle_train, 
                                  sampler=sampler, 
                                  num_workers=num_workers, 
                                  pin_memory=pin_memory,
                                  collate_fn=simple_collate_fn) # Use collate_fn
                                  
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory,
                                collate_fn=simple_collate_fn) # Use collate_fn

        # --- 模型初始化 (使用新的 NUM_CLASSES=5) ---
        logger.info(f"Fold {fold+1}: 初始化 LSTM 模型 for {NUM_CLASSES} classes...")
        model = LSTMOnlyModel( 
            input_feature_dim=cfg['model']['cnn_feature_dim'],
            lstm_hidden_size=cfg['model']['lstm_hidden_size'],
            lstm_num_layers=cfg['model']['lstm_num_layers'],
            num_classes=NUM_CLASSES, # Use updated class count (5)
            sequence_length=target_seq_len, 
            use_attention=cfg['model'].get('use_attention', False), 
            dropout_lstm=cfg['model']['dropout_lstm'],
            dropout_fc=cfg['model']['dropout_fc']
        ).to(device)
        logger.info(f"Fold {fold+1}: 模型 LSTMOnlyModel 初始化完成。参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # --- 损失函数和优化器 --- 
        criterion_name = cfg['training'].get('criterion', 'CrossEntropyLoss')
        if criterion_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss() 
        else:
            logger.error(f"不支持的损失函数: {criterion_name}. 使用 CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
        logger.info(f"Fold {fold+1}: 使用损失函数 {criterion_name}")
        
        optimizer_name = cfg['training'].get('optimizer', 'Adam')
        lr = cfg['training']['learning_rate']
        weight_decay = cfg['training']['weight_decay']
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
             logger.error(f"不支持的优化器: {optimizer_name}. 使用 AdamW.")
             optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"Fold {fold+1}: 使用优化器 {optimizer_name}, LR={lr}, Weight Decay={weight_decay}")

        # --- 学习率调度器 --- 
        scheduler_name = cfg['training'].get('scheduler', None)
        scheduler = None
        if scheduler_name:
            scheduler_params = cfg['training'].get('scheduler_params', {})
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **scheduler_params)
                logger.info(f"Fold {fold+1}: 使用 ReduceLROnPlateau 调度器，参数: {scheduler_params}")
            elif scheduler_name == 'StepLR':
                scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
                logger.info(f"Fold {fold+1}: 使用 StepLR 调度器，参数: {scheduler_params}")
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
                logger.info(f"Fold {fold+1}: 使用 CosineAnnealingLR 调度器，参数: {scheduler_params}")
            else:
                logger.warning(f"不支持的调度器: {scheduler_name}. 不使用调度器.")

        # --- 混合精度训练 --- 
        use_amp = cfg['training'].get('use_mixed_precision', False)
        # Fix TypeError: Remove unexpected keyword argument 'device_type'
        # Use the recommended import path torch.amp
        scaler = torch.amp.GradScaler(enabled=use_amp) 
        logger.info(f"Fold {fold+1}: 使用混合精度训练 (AMP): {use_amp}")

        # --- 获取训练配置 --- 
        max_epochs = cfg['training']['max_epochs'] 
        gradient_clip_val = cfg['training'].get('gradient_clip_val', None)
        # Output dir is now the run-specific directory
        # os.makedirs(run_output_dir, exist_ok=True) # Already created above

        # --- 当前 Fold 的训练循环 ---
        logger.info(f"Fold {fold+1}: === 开始训练, Max Epochs: {max_epochs} ===")
        for epoch in range(max_epochs):
            # --- Training Step ---
            model.train()
            train_loss = 0.0
            train_corrects = 0
            train_total = 0
            
            # TQDM progress bar
            train_pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
            for batch in train_pbar:
                # Handle potential None from collate_fn
                if batch is None: 
                    logger.warning(f"Fold {fold+1} Epoch {epoch+1}: Skipping empty/invalid batch in training.")
                    continue 
                inputs, targets = batch
                if inputs is None or targets is None:
                    logger.warning(f"Fold {fold+1} Epoch {epoch+1}: Skipping batch with None inputs/targets after unpacking.")
                    continue
                    
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Mixed Precision Forward Pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = model(inputs) # Shape: (batch_size, num_classes)
                    loss = criterion(outputs, targets)
                
                # Mixed Precision Backward Pass & Optimizer Step
                scaler.scale(loss).backward()
                
                # Gradient Clipping (apply before scaler.step)
                if gradient_clip_val:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                    
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_corrects += (predicted == targets).sum().item()
                
                # Update progress bar description
                train_pbar.set_postfix(loss=loss.item()) 

            train_epoch_loss = train_loss / train_total if train_total > 0 else 0.0
            train_epoch_acc = (100. * train_corrects / train_total) if train_total > 0 else 0.0
            
            # --- Validation Step ---
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_total = 0
            all_val_preds_epoch = []
            all_val_targets_epoch = []
            
            val_pbar = tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{max_epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    if batch is None: continue
                    inputs, targets = batch
                    if inputs is None or targets is None: continue
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Use AMP context for validation inference too for consistency & speed
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_corrects += (predicted == targets).sum().item()
                    all_val_preds_epoch.extend(predicted.cpu().numpy())
                    all_val_targets_epoch.extend(targets.cpu().numpy())
                    
                    val_pbar.set_postfix(loss=loss.item())
                    
            val_epoch_loss = val_loss / val_total if val_total > 0 else 0.0
            val_epoch_acc = (100. * val_corrects / val_total) if val_total > 0 else 0.0
            
            # --- Record Epoch Metrics --- 
            fold_train_losses.append(train_epoch_loss)
            fold_val_losses.append(val_epoch_loss)
            fold_train_accs.append(train_epoch_acc)
            fold_val_accs.append(val_epoch_acc)
            
            logger.info(f"Fold {fold+1} [Epoch {epoch+1}/{max_epochs}] Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

            # --- Learning Rate Scheduler Step --- 
            current_lr = optimizer.param_groups[0]['lr'] # Get current LR
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_epoch_loss)
                    logger.debug(f"Fold {fold+1} [Epoch {epoch+1}] ReduceLROnPlateau step with val_loss: {val_epoch_loss:.4f}")
                else:
                    scheduler.step()
                    logger.debug(f"Fold {fold+1} [Epoch {epoch+1}] Scheduler step executed.")
                # Log LR change
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    logger.info(f"Fold {fold+1} [Epoch {epoch+1}] Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")

            # --- Checkpoint Saving & Early Stopping --- 
            if val_epoch_loss < best_val_loss_fold:
                best_val_loss_fold = val_epoch_loss
                # Define the path using run_output_dir
                best_model_path_fold = os.path.join(run_output_dir, f"{exp_name}_fold_{fold+1}_best.pth") 
                try:
                    torch.save(model.state_dict(), best_model_path_fold)
                    logger.info(f"Fold {fold+1} [Epoch {epoch+1}] New best model saved to {best_model_path_fold} with Val Loss: {best_val_loss_fold:.4f}")
                except Exception as e:
                    logger.error(f"Fold {fold+1} [Epoch {epoch+1}]: Failed to save best model to {best_model_path_fold}: {e}", exc_info=True)

            # --- Early Stopping Check ---
            if es_enabled:
                current_score = val_epoch_loss if es_metric == 'val_loss' else val_epoch_acc
                
                # Check for improvement based on mode and min_delta
                improved = delta_comparison(current_score, best_score)

                if improved:
                    logger.debug(f"EarlyStopping: Metric improved ({es_metric}: {best_score:.6f} -> {current_score:.6f}). Resetting counter.")
                    best_score = current_score
                    wait_count = 0
                else:
                    wait_count += 1
                    logger.debug(f"EarlyStopping: Metric did not improve. Counter: {wait_count}/{es_patience}")

                if wait_count >= es_patience:
                    stopped_epoch = epoch + 1 # Record the epoch number we stopped at
                    logger.info(f"Fold {fold+1}: Early stopping triggered at epoch {stopped_epoch} because '{es_metric}' did not improve for {es_patience} epochs.")
                    break # Exit epoch loop for this fold

        # Logging after loop completion
        final_epoch_count = stopped_epoch if es_enabled and stopped_epoch > 0 else max_epochs
        logger.info(f"Fold {fold+1}: === Training Completed ({final_epoch_count} epochs) === Best Val Loss: {best_val_loss_fold:.4f}")

        # --- Plot Training Curves for this Fold --- 
        try:
            epochs_ran = len(fold_train_losses)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, epochs_ran + 1), fold_train_losses, label='Train Loss')
            plt.plot(range(1, epochs_ran + 1), fold_val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold+1} Loss Curve')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epochs_ran + 1), fold_train_accs, label='Train Accuracy')
            plt.plot(range(1, epochs_ran + 1), fold_val_accs, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Fold {fold+1} Accuracy Curve')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            curve_path = os.path.join(run_output_dir, f"{exp_name}_fold_{fold+1}_training_curves.png")
            plt.savefig(curve_path)
            plt.close()
            logger.info(f"Fold {fold+1} training curves saved to {curve_path}")
        except Exception as e:
            logger.error(f"Fold {fold+1}: Error plotting training curves: {e}", exc_info=True)

        # Store metrics of the last fold for potential overall curve plotting (optional)
        if fold == k_folds - 1: 
            last_fold_train_losses = fold_train_losses
            last_fold_val_losses = fold_val_losses
            last_fold_train_accs = fold_train_accs
            last_fold_val_accs = fold_val_accs

        # --- Final Evaluation for the Fold (Use new class defs) --- 
        logger.info(f"Fold {fold+1}: Loading best model for final evaluation...")
        all_val_preds_fold = [] 
        all_val_targets_fold = [] 
        # Modify the condition here to check for None as well
        if best_model_path_fold is not None and os.path.exists(best_model_path_fold):
            try: # Add try block for loading
                model.load_state_dict(torch.load(best_model_path_fold))
                logger.info(f"Fold {fold+1}: Loaded best model with validation loss: {best_val_loss_fold:.4f}")

                model.eval()
                fold_test_corrects = 0
                fold_test_total = 0
                logger.info(f"Fold {fold+1}: Evaluating best model on validation set ({NUM_CLASSES} classes)...")
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Fold {fold+1} [Final Eval]", leave=False):
                        if batch is None: continue
                        inputs, targets = batch
                        if inputs is None or targets is None: continue
                        
                        inputs, targets = inputs.to(device), targets.to(device)
                        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                            outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        fold_test_total += targets.size(0)
                        fold_test_corrects += (predicted == targets).sum().item()
                        all_val_preds_fold.extend(predicted.cpu().numpy())
                        all_val_targets_fold.extend(targets.cpu().numpy())
                
                fold_accuracy = (100. * fold_test_corrects / fold_test_total) if fold_test_total > 0 else 0.0
                logger.info(f"Fold {fold+1}: Final Validation Accuracy: {fold_accuracy:.2f}%")
                
                all_fold_results.append({
                    'fold': fold+1, 
                    'accuracy': fold_accuracy, 
                    'best_val_loss': best_val_loss_fold,
                    'predictions': all_val_preds_fold, 
                    'targets': all_val_targets_fold # These are already remapped labels
                })
                                   
                # --- Reporting using NEW class defs ---
                try: # Inner try for reporting
                    target_names = [EMOTION_IDX_TO_NAME.get(i, str(i)) for i in range(NUM_CLASSES)]
                    expected_labels = list(range(NUM_CLASSES)) # Labels are 0 to NUM_CLASSES-1
                    report = classification_report(all_val_targets_fold, all_val_preds_fold, 
                                           target_names=target_names, labels=expected_labels, digits=4, zero_division=0)
                    # Fix: Combine f-string into one line
                    logger.info(f"Fold {fold+1} Classification Report ({NUM_CLASSES} classes):\n{report}")
                    cm = confusion_matrix(all_val_targets_fold, all_val_preds_fold, labels=expected_labels)
                    # Consider saving cm data instead of plotting per fold
                except Exception as e:
                    logger.error(f"Fold {fold+1}: Error generating classification report or confusion matrix: {e}", exc_info=True)
            
            except Exception as load_err: # Add except block for model loading
                 logger.error(f"Fold {fold+1}: Error loading best model state dict from {best_model_path_fold}: {load_err}")
        else:
            logger.warning(f"Fold {fold+1}: Best model checkpoint not found at {best_model_path_fold}, skipping final evaluation for this fold.")

    # --- End of K-Fold Loop ---

    # --- Aggregate and Report K-Fold Results (Use new class defs) --- 
    overall_f1_macro = 0.0 # Default value if evaluation fails
    if k_folds > 1 and all_fold_results:
        # ... (Calculate avg accuracy/loss - no change) ...
        
        # Aggregate predictions and targets across all folds (no change)
        all_preds_overall = np.concatenate([res['predictions'] for res in all_fold_results])
        all_targets_overall = np.concatenate([res['targets'] for res in all_fold_results])
        
        logger.info(f"--- Overall Evaluation (Aggregated across {k_folds} Folds, {NUM_CLASSES} classes) ---")
        try:
            # Use the remapped names and count
            target_names = [EMOTION_IDX_TO_NAME.get(i, str(i)) for i in range(NUM_CLASSES)]
            expected_labels = list(range(NUM_CLASSES))
            
            # Generate text report first
            overall_report_text = classification_report(all_targets_overall, all_preds_overall, 
                                           target_names=target_names, labels=expected_labels, digits=4, zero_division=0)
            logger.info(f"Overall Classification Report:\n{overall_report_text}")
            
            # Save text report
            report_path = os.path.join(run_output_dir, f"{exp_name}_overall_classification_report.txt")
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"Overall Classification Report (Aggregated across {k_folds} Folds, {NUM_CLASSES} classes):\n")
                    f.write(overall_report_text)
                logger.info(f"Overall classification report saved to {report_path}")
            except Exception as e:
                logger.error(f"保存总体分类报告到文件失败 {report_path}: {e}", exc_info=True)

            # Calculate metrics for plotting
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets_overall, 
                all_preds_overall, 
                labels=expected_labels, 
                zero_division=0, 
                average=None # Get per-class scores
            )
            
            # --- Plot Overall Metrics Bar Chart --- 
            plt.style.use('seaborn-v0_8-whitegrid') # Use a style with grid
            fig, ax = plt.subplots(figsize=(12, 7))
            bar_width = 0.25
            index = np.arange(len(target_names))
            
            bar1 = ax.bar(index - bar_width, precision, bar_width, label='precision')
            bar2 = ax.bar(index, recall, bar_width, label='recall')
            bar3 = ax.bar(index + bar_width, f1, bar_width, label='f1-score')

            ax.set_xlabel('情绪类别 (Emotion Class)')
            ax.set_ylabel('分数 (Score)')
            ax.set_title('微表情识别 - 分类指标报告')
            ax.set_xticks(index)
            ax.set_xticklabels(target_names, rotation=45, ha="right")
            ax.legend(title='指标 (Metric)')
            ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
            ax.set_ylim(0, 1.05) # Set y-axis limit
            
            # Add value labels on top of bars (optional, can be cluttered)
            # ax.bar_label(bar1, padding=3, fmt='%.2f')
            # ax.bar_label(bar2, padding=3, fmt='%.2f')
            # ax.bar_label(bar3, padding=3, fmt='%.2f')

            plt.tight_layout()
            metrics_plot_path = os.path.join(run_output_dir, f"{exp_name}_overall_metrics_report.png")
            plt.savefig(metrics_plot_path)
            plt.close(fig)
            logger.info(f"Overall metrics report plot saved to {metrics_plot_path}")

            # --- Save Confusion Matrix Plot --- 
            overall_cm = confusion_matrix(all_targets_overall, all_preds_overall, labels=expected_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Overall Confusion Matrix ({NUM_CLASSES} classes, K={k_folds} Folds)') 
            # Save plot to the run-specific directory
            overall_cm_path = os.path.join(run_output_dir, f"{exp_name}_overall_confusion_matrix.png") 
            plt.tight_layout()
            plt.savefig(overall_cm_path)
            plt.close()
            logger.info(f"Overall confusion matrix saved to {overall_cm_path}")
            
            # Calculate overall F1 macro average
            overall_f1_macro = f1_score(all_targets_overall, all_preds_overall, labels=expected_labels, average='macro', zero_division=0)
            logger.info(f"Overall Macro F1-Score: {overall_f1_macro:.4f}")
            
        except Exception as e:
            logger.error(f"Error generating/saving overall classification report or plots: {e}", exc_info=True)
            overall_f1_macro = 0.0 # Return 0 if error occurs during reporting
            
    elif k_folds < 1:
         logger.error("K-Folds configured to < 1, cannot perform evaluation.")
         overall_f1_macro = 0.0
    else: # No folds were successfully evaluated
        logger.warning("没有成功的 Fold 结果可供聚合评估。")
        overall_f1_macro = 0.0

    # Return the metric to be optimized by Optuna
    return overall_f1_macro

# --- 主程序入口 (修改日志记录器名称) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 LSTM 模型 (阶段二) - 支持 K-Fold 和 6 类") # Update description
    parser.add_argument("--config", type=str, required=True, help="指向 Stage 2 YAML 配置文件的路径")
    args = parser.parse_args()

    # 加载配置 (更稳妥的方式)
    config = None
    try:
        # 修改：先以二进制读取，再用 UTF-8 解码
        with open(args.config, 'rb') as f_bytes:
            config_bytes = f_bytes.read()
        config_string = config_bytes.decode('utf-8')
        config = yaml.safe_load(config_string)
        config['config_path'] = args.config # Store config path
    except FileNotFoundError:
        print(f"错误: 配置文件未找到于 {args.config}")
        exit(1)
    except UnicodeDecodeError:
        print(f"错误: 配置文件 {args.config} 不是有效的 UTF-8 编码。请检查文件编码。")
        exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 配置文件 {args.config} 时出错: {e}")
        exit(1)
    except Exception as e:
        print(f"加载配置文件时发生未知错误: {e}")
        exit(1)
        
    if config is None:
        print("错误: 配置文件未能成功加载。")
        exit(1)

    # 设置日志
    log_dir = config['logging'].get('log_dir', '18_2/logs')
    # Use the experiment name from the config file
    exp_name = config.get('experiment_name', 'lstm_stage2_kfold_6class') 
    setup_logging(log_dir, exp_name)
    logger.info(f"配置加载自: {args.config}")
    logger.info(f"实验名称: {exp_name}")

    # 开始训练
    train_lstm_stage2(config)