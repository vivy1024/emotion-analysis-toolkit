import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
import yaml
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- 导入自定义模块 ---
from .utils import MICRO_EMOTION_MAP, NUM_CLASSES, EMOTION_IDX_TO_NAME, logger, setup_logging # Use setup_logging from utils
from .dataset import MicroExpressionDataset
from .transforms import create_spatial_transforms
from .models import CNNModel

# --- Wrapper Dataset to Apply Transforms to Subset ---
class TransformedSubsetDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]
        if data is None: # Handle loading error from underlying dataset
             return None, None
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)

# --- Collate Function --- 
def simple_collate_fn(batch):
    """Collates batches, filtering out None values resulting from loading errors."""
    batch = [(data, target) for data, target in batch if data is not None]
    if not batch: # If all samples in batch failed
        return None, None
    # Default collate handles stacking tensors and labels
    return torch.utils.data.dataloader.default_collate(batch)

# --- 主训练函数 ---
def train_cnn_stage1(cfg):
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
        # 可选: 设置确定性算法，可能影响性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- 数据加载与预处理 ---
    logger.info("加载元数据...")
    metadata_file_config = cfg['data']['metadata_file']
    data_path_config = cfg['data']['data_path']
    if os.path.isabs(metadata_file_config):
        metadata_path = metadata_file_config
    else:
        logger.warning(f"Metadata file path '{metadata_file_config}' is relative. Joining with data_path '{data_path_config}'. Consider using an absolute path in the config.")
        metadata_path = os.path.join(data_path_config, metadata_file_config)

    try:
        metadata_df = pd.read_excel(metadata_path)
        logger.info(f"元数据加载成功，包含 {len(metadata_df)} 条记录。")
    except FileNotFoundError:
        logger.error(f"错误：元数据文件未找到于 {metadata_path}")
        return
    except Exception as e:
        logger.error(f"加载元数据时出错: {e}")
        return

    # --- 创建数据变换 ---
    image_size = tuple(cfg['data'].get('image_size', [128, 128]))
    train_transforms = create_spatial_transforms(image_size=image_size, is_train=True)
    val_test_transforms = create_spatial_transforms(image_size=image_size, is_train=False)
    logger.info(f"数据变换创建完成. 训练时增强: True, 图像大小: {image_size}")

    # --- 创建数据集实例 (单帧模式) ---
    logger.info("创建数据集实例 (模式: single_frame)...")
    full_dataset = MicroExpressionDataset(
        metadata_df=metadata_df,
        data_root=cfg['data']['data_path'],
        mode='single_frame', 
        image_size=image_size,
        spatial_transform=None # Transforms applied in DataLoader loop
    )

    if len(full_dataset) == 0:
        logger.error("数据集为空，无法继续训练。请检查数据路径和元数据。")
        return

    # --- 数据集划分 ---
    logger.info("划分数据集 (基于帧)...")
    # 注意：这里的标签是每一帧的标签
    all_labels = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))
    
    # Stratify based on frame labels
    try:
        train_idx, temp_idx, _, _ = train_test_split(
            indices, all_labels, 
            test_size=(1.0 - cfg['data']['split_ratio'][0]), 
            random_state=seed, 
            stratify=all_labels
        )
        # Adjust validation ratio relative to the remaining data
        val_ratio_of_remaining = cfg['data']['split_ratio'][1] / (cfg['data']['split_ratio'][1] + cfg['data']['split_ratio'][2])
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, [all_labels[i] for i in temp_idx], 
            test_size=(1.0 - val_ratio_of_remaining), 
            random_state=seed, 
            stratify=[all_labels[i] for i in temp_idx]
        )
        logger.info(f"数据集划分完成: 训练帧 {len(train_idx)}, 验证帧 {len(val_idx)}, 测试帧 {len(test_idx)}")
    except ValueError as e:
         logger.error(f"数据集划分失败，可能是因为某些类别样本过少无法进行分层抽样: {e}")
         logger.warning("尝试不使用分层抽样进行划分...")
         train_idx, temp_idx = train_test_split(indices, test_size=(1.0 - cfg['data']['split_ratio'][0]), random_state=seed)
         val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - val_ratio_of_remaining), random_state=seed)
         logger.info(f"数据集划分完成 (无分层): 训练帧 {len(train_idx)}, 验证帧 {len(val_idx)}, 测试帧 {len(test_idx)}")

    # --- 获取训练集标签用于加权采样 ---
    train_labels = [all_labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    logger.info(f"训练集帧类别分布: { {EMOTION_IDX_TO_NAME.get(k, k): v for k, v in sorted(class_counts.items())} }")
    
    # --- 处理数据集不平衡 (WeightedRandomSampler) ---
    sampler = None
    if cfg['balancing'].get('use_weighted_sampler', True): # Default to True for stage 1
        logger.info("启用 WeightedRandomSampler...")
        num_samples = len(train_labels)
        # Weight calculation: weight = 1 / num_samples_in_class
        class_weights_calc = {cls: 1.0 / count if count > 0 else 0 for cls, count in class_counts.items()}
        
        # Optional: Boost minority classes (like 'fear')
        if cfg['balancing'].get('boost_minority_classes', True):
            # Define minority classes (e.g., 'fear', 'sadness', or based on percentile)
            fear_idx = MICRO_EMOTION_MAP.get('fear', -1)
            sadness_idx = MICRO_EMOTION_MAP.get('sadness', -1)
            minority_classes_idx = {idx for idx in [fear_idx, sadness_idx] if idx != -1 and idx in class_weights_calc}
            # Or use percentile: minority_threshold = np.percentile(list(class_counts.values()), cfg['balancing'].get('minority_percentile', 25))
            # for cls, count in class_counts.items(): if count <= minority_threshold: minority_classes_idx.add(cls)
            
            boost_factor = cfg['balancing'].get('oversample_factor', 2.0)
            for cls_idx in minority_classes_idx:
                 if cls_idx in class_weights_calc:
                     original_weight = class_weights_calc[cls_idx]
                     class_weights_calc[cls_idx] *= boost_factor
                     logger.info(f"提升少数类 '{EMOTION_IDX_TO_NAME.get(cls_idx, cls_idx)}' 权重: {original_weight:.4f} -> {class_weights_calc[cls_idx]:.4f} (因子: {boost_factor:.2f})")
                 
        # Assign weight to each sample in the training set
        sample_weights = [class_weights_calc[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("WeightedRandomSampler 创建完成。")

    # --- 创建 Subset 实例和 DataLoader ---
    # Pass transforms to apply in the loop, as they are applied per-frame
    train_dataset_subset = Subset(full_dataset, train_idx)
    val_dataset_subset = Subset(full_dataset, val_idx)
    test_dataset_subset = Subset(full_dataset, test_idx)
    
    # Wrapper to apply transforms within DataLoader, needed because Subset doesn't hold transforms
    train_dataset_transformed = TransformedSubsetDataset(train_dataset_subset, train_transforms)
    val_dataset_transformed = TransformedSubsetDataset(val_dataset_subset, val_test_transforms)
    test_dataset_transformed = TransformedSubsetDataset(test_dataset_subset, val_test_transforms)
    
    logger.info("创建数据加载器...")
    batch_size = cfg['training'].get('batch_size', 32)
    num_workers = cfg['environment'].get('num_workers', 4)
    pin_memory = cfg['environment'].get('pin_memory', True)

    train_loader = DataLoader(train_dataset_transformed,
                              batch_size=batch_size,
                              sampler=sampler,
                              shuffle=(sampler is None), # Shuffle must be False if sampler is used
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=simple_collate_fn)

    val_loader = DataLoader(val_dataset_transformed,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=simple_collate_fn)

    test_loader = DataLoader(test_dataset_transformed, 
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=simple_collate_fn)

    # --- 模型初始化 ---
    logger.info("初始化 CNN 模型...")
    model = CNNModel(num_classes=NUM_CLASSES, input_channels=1).to(device) # Input is grayscale (1 channel)
    logger.info(f"模型 CNNModel 初始化完成。参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 损失函数和优化器 ---
    logger.info("设置损失函数和优化器...")
    criterion = nn.CrossEntropyLoss() # Includes Softmax

    optimizer_name = cfg['training'].get('optimizer', 'AdamW').lower()
    lr = cfg['training'].get('learning_rate', 0.0001)
    wd = cfg['training'].get('weight_decay', 0.01)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'sgd':
        # Match original project's LSTM stage optimizer params loosely
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['training'].get('momentum', 0.9), weight_decay=wd)
    else:
        logger.warning(f"不支持的优化器: {optimizer_name}. 使用 AdamW 作为默认.")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    logger.info(f"优化器: {type(optimizer).__name__}, 学习率: {lr}, 权重衰减: {wd}")

    # --- 学习率调度器 ---
    scheduler = None
    scheduler_name = cfg['training'].get('scheduler')
    if scheduler_name:
        scheduler_params = cfg['training'].get('scheduler_params', {})
        try:
            if scheduler_name.lower() == 'reducelronplateau':
                # Monitor validation loss like original project
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                 mode='min', # Reduce LR when val_loss stops decreasing
                                                                 factor=scheduler_params.get('factor', 0.1), 
                                                                 patience=scheduler_params.get('patience', 3), 
                                                                 verbose=True,
                                                                 min_lr=scheduler_params.get('min_lr', 1e-7))
            elif scheduler_name.lower() == 'cosineannealinglr':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_params.get('T_max', cfg['training']['epochs']))
            elif scheduler_name.lower() == 'steplr':
                 scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                     step_size=scheduler_params.get('step_size', 10),
                                                     gamma=scheduler_params.get('gamma', 0.1))
            else:
                logger.warning(f"不支持的学习率调度器: {scheduler_name}")
            if scheduler:
                logger.info(f"使用学习率调度器: {scheduler_name} with params {scheduler_params}")
        except KeyError as e:
             logger.error(f"配置调度器 {scheduler_name} 时缺少参数: {e}. 将不使用调度器.")
             scheduler = None

    # --- 混合精度训练 (可选) ---
    scaler = None
    use_amp = cfg['training'].get('use_mixed_precision', False)
    if use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("启用混合精度训练 (AMP)")
    elif use_amp and not torch.cuda.is_available():
        logger.warning("请求混合精度训练，但 CUDA 不可用。将不使用 AMP.")

    # --- 训练循环 ---
    logger.info("开始 CNN 阶段训练...")
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    output_dir = cfg['logging'].get('output_dir', '18_2/stage1_cnn_output')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{cfg['experiment_name']}_cnn_best.pth")

    epochs = cfg['training'].get('epochs', 30)
    patience = cfg['training'].get('early_stopping_patience', 5)
    accumulate_grad_batches = cfg['training'].get('accumulate_grad_batches', 1)
    gradient_clip_val = cfg['training'].get('gradient_clip_val', 0) # 0 means no clipping

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        
        optimizer.zero_grad() # Reset gradients at the start of accumulation cycle

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, batch in enumerate(train_loader_tqdm):
            if batch is None or batch[0] is None: 
                # logger.debug(f"Skipping empty batch {i}")
                continue # Skip empty batches 
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulate_grad_batches # Scale loss for accumulation

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation and Clipping Step
            if (i + 1) % accumulate_grad_batches == 0:
                if gradient_clip_val > 0:
                    if scaler: scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle

            # Statistics
            running_loss += loss.item() * accumulate_grad_batches # Accumulate unscaled loss for logging
            _, predicted = torch.max(outputs.data, 1)
            running_total += targets.size(0)
            running_corrects += (predicted == targets).sum().item()
            
            # Update progress bar
            train_loader_tqdm.set_postfix(loss=f"{(running_loss / (i + 1)):.4f}", acc=f"{(100. * running_corrects / running_total):.2f}%")
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * running_corrects / running_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        logger.info(f"Epoch {epoch+1} Train Summary: Avg Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%")

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_loader_tqdm:
                if batch is None or batch[0] is None: continue
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_corrects += (predicted == targets).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
                val_loader_tqdm.set_postfix(loss=f"{(val_loss / val_total * batch_size if val_total > 0 else 0):.4f}", acc=f"{(100. * val_corrects / val_total if val_total > 0 else 0):.2f}%")

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = (100. * val_corrects / val_total) if val_total > 0 else 0.0
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        logger.info(f"Epoch {epoch+1} Validation Summary: Avg Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.2f}%")

        # --- 学习率调度和早停 ---
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} completed. Current LR: {current_lr:.8f}")
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Checkpointing and Early Stopping
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path) # Save only the state_dict
            logger.info(f"Validation accuracy improved to {best_val_accuracy:.2f}%. Saving model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation accuracy did not improve. Best: {best_val_accuracy:.2f}%. Early stopping counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"CNN Stage 1 training finished. Best validation accuracy: {best_val_accuracy:.2f}% achieved.")

    # --- 绘制训练曲线 ---
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.title('CNN Stage 1 Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')
        plt.title('CNN Stage 1 Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, f"{cfg['experiment_name']}_cnn_training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training curves saved to {plot_path}")
    except Exception as e:
         logger.error(f"Error plotting training curves: {e}")

    # --- 测试阶段 (使用最佳模型) ---
    logger.info("Loading best model for testing...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.2f}%")
        
        model.eval()
        test_corrects = 0
        test_total = 0
        all_test_preds = []
        all_test_targets = []

        test_loader_tqdm = tqdm(test_loader, desc="[Test]", leave=False)
        with torch.no_grad():
            for batch in test_loader_tqdm:
                if batch is None or batch[0] is None: continue
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_corrects += (predicted == targets).sum().item()
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_targets.extend(targets.cpu().numpy())
                test_loader_tqdm.set_postfix(acc=f"{(100. * test_corrects / test_total if test_total > 0 else 0):.2f}%")
        
        test_accuracy = (100. * test_corrects / test_total) if test_total > 0 else 0.0
        logger.info(f"--- Test Set Evaluation ---:")
        logger.info(f"        Accuracy: {test_accuracy:.2f}%")
        
        # Classification Report and Confusion Matrix
        try:
            target_names = [EMOTION_IDX_TO_NAME.get(i, str(i)) for i in range(NUM_CLASSES)]
            report = classification_report(all_test_targets, all_test_preds, target_names=target_names, digits=4)
            logger.info("Classification Report:\n" + report)
            
            cm = confusion_matrix(all_test_targets, all_test_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('CNN Stage 1 Confusion Matrix')
            cm_path = os.path.join(output_dir, f"{cfg['experiment_name']}_cnn_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()
            logger.info(f"Confusion matrix saved to {cm_path}")
            
        except Exception as e:
            logger.error(f"Error generating/saving classification report or confusion matrix: {e}")
    else:
        logger.warning("Best model checkpoint not found, skipping testing.")

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 CNN 模型 (阶段一)")
    parser.add_argument("--config", type=str, required=True, help="指向 Stage 1 YAML 配置文件的路径")
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
    exp_name = config.get('experiment_name', 'cnn_stage1')
    setup_logging(log_dir, exp_name)
    logger.info(f"配置加载自: {args.config}")
    logger.info(f"实验名称: {exp_name}")
    # logger.info(f"完整配置: {config}") # Optional: log full config

    # 开始训练
    train_cnn_stage1(config) 