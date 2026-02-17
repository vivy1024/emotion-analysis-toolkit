#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
整合多个FER2013数据集文件进行表情识别模型训练
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 添加当前目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 从emotion_model模块导入函数和常量
from macro_expression.emotion_model import get_emotion_model

# 情绪标签映射
EMOTIONS = {
    0: "生气",     # Angry
    1: "厌恶",     # Disgust
    2: "恐惧",     # Fear
    3: "开心",     # Happy
    4: "难过",     # Sad
    5: "惊讶",     # Surprise
    6: "中性"      # Neutral
}

# 设定随机种子以确保结果可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检查并创建目录
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# FER2013数据集类
class FER2013Dataset(Dataset):
    def __init__(self, pixels, emotions, transform=None, use_augmentation=False):
        """
        FER2013数据集加载器
        
        Args:
            pixels: 像素数据数组，形状为 (n_samples, 48*48)
            emotions: 情绪标签数组，形状为 (n_samples,)
            transform: PyTorch变换
            use_augmentation: 是否使用数据增强
        """
        self.pixels = pixels
        self.emotions = emotions
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # 数据增强变换
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    def __len__(self):
        return len(self.pixels)
    
    def __getitem__(self, idx):
        # 获取像素数据并转换为图像
        pixel_data = self.pixels[idx].reshape(48, 48).astype(np.float32)
        image = Image.fromarray(np.uint8(pixel_data * 255), 'L')  # 转换为PIL图像
        
        # 应用数据增强（如果启用）
        if self.use_augmentation and random.random() > 0.5:
            image = self.augmentation_transforms(image)
            
        # 应用基本变换
        if self.transform:
            image = self.transform(image)
            
        # 获取标签
        label = self.emotions[idx]
        
        return image, label

# 数据预处理函数
def preprocess_fer2013(csv_paths):
    """
    处理多个FER2013数据集文件，支持不同格式
    
    Args:
        csv_paths: FER2013 CSV文件路径列表
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: 训练集、验证集和测试集的数据和标签
    """
    all_pixels = []
    all_emotions = []
    
    # 处理所有数据文件
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"警告: 文件不存在 {csv_path}")
            continue
            
        print(f"处理数据文件: {csv_path}")
        try:
            # 针对icml_face_data.csv格式，预处理列名
            if "icml_face_data.csv" in csv_path:
                # 由于icml_face_data.csv的列名有特殊字符
                data = pd.read_csv(csv_path, skipinitialspace=True)
                # 检查并修正列名前后空格
                data.columns = [col.strip() for col in data.columns]
                
                if 'emotion' in data.columns and 'Usage' in data.columns and 'pixels' in data.columns:
                    print(f"从 {csv_path} 加载 {len(data)} 个样本 (ICML格式)")
                    pixels = data['pixels'].tolist()
                    emotions = data['emotion'].values
                    
            # 其他格式的数据文件
            else:
                data = pd.read_csv(csv_path)
                
                # 标准FER2013格式
                if 'pixels' in data.columns and 'emotion' in data.columns:
                    print(f"从 {csv_path} 加载 {len(data)} 个样本 (标准FER2013格式)")
                    pixels = data['pixels'].tolist()
                    emotions = data['emotion'].values
                
                # test.csv没有emotion列
                elif 'pixels' in data.columns and len(data.columns) == 1:
                    print(f"从 {csv_path} 加载 {len(data)} 个样本 (测试集格式-无标签)")
                    pixels = data['pixels'].tolist()
                    
                    # 对于测试数据，我们没有标签，但需要一个占位符
                    # 使用特殊值-1标记，后面会过滤掉
                    emotions = np.full(len(pixels), -1, dtype=np.int32)
                
                # 可能是example_submission.csv格式
                elif 'emotion' in data.columns and 'id' in data.columns and len(data.columns) == 2:
                    print(f"文件 {csv_path} 似乎是提交格式文件，无像素数据，跳过")
                    continue
                
                # 尝试处理其他可能的格式
                else:
                    # 尝试识别像素数据列
                    potential_pixel_columns = ['pixels', 'raw_pixels', 'pixel', 'Pixels', 'features']
                    pixel_column = None
                    
                    for col in potential_pixel_columns:
                        if col in data.columns:
                            pixel_column = col
                            break
                    
                    if not pixel_column:
                        print(f"文件 {csv_path} 找不到像素数据列，跳过")
                        continue
                    
                    # 寻找情绪标签列
                    potential_emotion_columns = ['emotion', 'label', 'class', 'Expression', 'expression']
                    emotion_column = None
                    
                    for col in potential_emotion_columns:
                        if col in data.columns:
                            emotion_column = col
                            break
                    
                    if emotion_column:
                        print(f"从 {csv_path} 加载 {len(data)} 个样本 (使用 {pixel_column} 作为像素, {emotion_column} 作为标签)")
                        pixels = data[pixel_column].tolist()
                        emotions = data[emotion_column].values
                    else:
                        print(f"文件 {csv_path} 找不到情绪标签列，使用-1作为占位符")
                        pixels = data[pixel_column].tolist()
                        emotions = np.full(len(pixels), -1, dtype=np.int32)
            
            # 将像素字符串转换为数组
            for i, pixel_str in enumerate(pixels):
                try:
                    # 尝试不同的分隔符
                    for sep in [' ', ',']:
                        try:
                            pixel_array = np.fromstring(str(pixel_str), dtype=float, sep=sep)
                            if len(pixel_array) == 2304:  # 48*48
                                all_pixels.append(pixel_array)
                                all_emotions.append(emotions[i])
                                break
                        except:
                            continue
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    
        except Exception as e:
            print(f"读取 {csv_path} 时出错: {e}")
    
    if not all_pixels:
        raise ValueError("没有加载到有效数据！")
        
    # 转换为numpy数组
    X = np.array(all_pixels) / 255.0  # 归一化到 [0, 1]
    y = np.array(all_emotions)
    
    print(f"总共加载 {len(X)} 个样本")
    
    # 过滤掉没有标签的样本
    valid_indices = y != -1
    if not np.all(valid_indices):
        print(f"过滤掉 {np.sum(~valid_indices)} 个没有标签的样本")
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"过滤后剩余 {len(X)} 个样本")
    
    # 数据集划分: 先划分训练集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 再从训练集中划分出验证集
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)
    
    print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本, 测试集: {len(X_test)} 样本")
    
    # 分析每个类别的样本数量
    for emotion_id in range(7):
        print(f"情绪 {EMOTIONS[emotion_id]}: 训练集 {np.sum(y_train == emotion_id)}, "
              f"验证集 {np.sum(y_val == emotion_id)}, 测试集 {np.sum(y_test == emotion_id)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=50, model_save_path='models'):
    # 确保模型保存目录存在
    ensure_dir(model_save_path)
    
    # 开始训练
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm创建进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({'loss': f'{train_loss/train_total:.4f}', 'acc': f'{train_acc:.2f}%'})
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({'loss': f'{val_loss/val_total:.4f}', 'acc': f'{val_acc:.2f}%'})
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印统计信息
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, f'{model_save_path}/best_model.pt')
            print(f'模型保存至 {model_save_path}/best_model.pt')
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, f'{model_save_path}/checkpoint_epoch{epoch+1}.pt')
    
    # 训练结束，保存最终模型
    torch.save(model.state_dict(), f'{model_save_path}/final_model.pt')
    print(f'最终模型保存至 {model_save_path}/final_model.pt')
    
    return history

# 评估模型
def evaluate_model(model, test_loader, device, save_dir):
    """
    在测试集上评估模型
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='测试'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    test_acc = 100 * test_correct / test_total
    print(f'测试准确率: {test_acc:.2f}%')
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))],
               yticklabels=[EMOTIONS[i] for i in range(len(EMOTIONS))])
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    
    # 打印分类报告
    report = classification_report(all_labels, all_preds, 
                                  target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))],
                                  digits=4)
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'测试准确率: {test_acc:.2f}%\n\n')
        f.write('分类报告:\n')
        f.write(report)
    
    return test_acc

# 绘制训练曲线
def plot_training_curves(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FER2013 表情识别模型训练 (多数据集版本)')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, nargs='+', 
                        default=[
                            "D:/fer2013/icml_face_data.csv",
                            "D:/fer2013/test.csv",
                            "D:/fer2013/train.csv",
                            "D:/fer2013/example_submission.csv",
                            "D:/fer2013/fer2013/fer2013.csv"
                        ],
                        help='数据集CSV文件路径列表')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='EmotionResNet',
                        choices=['EmotionCNN', 'DeepEmotionCNN', 'EmotionResNet', 'EmotionEfficientNet'],
                        help='模型名称')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 保存相关参数
    parser.add_argument('--save_dir', type=str, default='models/multi_dataset', help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保保存目录存在
    ensure_dir(args.save_dir)
    
    # 数据预处理
    print("预处理数据...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_fer2013(args.data_path)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集
    train_dataset = FER2013Dataset(X_train, y_train, transform=transform, use_augmentation=True)
    val_dataset = FER2013Dataset(X_val, y_val, transform=transform, use_augmentation=False)
    test_dataset = FER2013Dataset(X_test, y_test, transform=transform, use_augmentation=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_emotion_model(args.model_name)
    model = model.to(device)
    
    # 打印模型信息
    print(f"模型: {args.model_name}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    print(f"开始训练 {args.model_name}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        model_save_path=args.save_dir
    )
    
    # 绘制训练曲线
    plot_training_curves(history, args.save_dir)
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    # 评估模型
    print("评估最佳模型...")
    test_acc = evaluate_model(model, test_loader, device, args.save_dir)
    
    print(f"训练完成! 最终测试准确率: {test_acc:.2f}%")
    print(f"模型和评估结果保存在 {args.save_dir}")

if __name__ == "__main__":
    main() 