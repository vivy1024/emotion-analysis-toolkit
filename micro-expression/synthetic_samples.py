# 18_1/augmentation/synthetic_samples.py

import numpy as np
import torch
import cv2
import random
from collections import defaultdict

class SyntheticSampleGenerator:
    """产生合成样本以增强少数类的表示
    
    通过组合现有样本的特征来生成新的合成样本，用于提高少数类的样本量。
    """
    
    def __init__(self, dataset, target_class, min_samples=10, method='interpolation'):
        """
        Args:
            dataset: 原始数据集对象
            target_class: 需要生成合成样本的目标类别
            min_samples: 每个类别最少需要的样本数
            method: 合成方法，可选'interpolation', 'jitter', 'mask_mix'
        """
        self.dataset = dataset
        self.target_class = target_class
        self.min_samples = min_samples
        self.method = method
        
        # 按类别组织样本索引
        self.class_indices = defaultdict(list)
        for i in range(len(dataset)):
            # 使用数据集的get_label方法获取标签
            try:
                label = dataset.get_label(i)
                self.class_indices[label].append(i)
            except:
                # 如果是子集，使用原始数据集的索引
                if hasattr(dataset, 'indices'):
                    original_idx = dataset.indices[i]
                    label = dataset.dataset.get_label(original_idx)
                    self.class_indices[label].append(i)
    
    def generate_samples(self, num_samples=None):
        """为目标类别生成指定数量的合成样本
        
        Args:
            num_samples: 要生成的样本数量，默认为None（使用min_samples计算）
            
        Returns:
            样本和标签对的列表，格式为[(sample1, label1), (sample2, label2), ...]
        """
        target_indices = self.class_indices[self.target_class]
        
        # 如果类别中没有样本，无法生成合成样本
        if len(target_indices) == 0:
            print(f"无法为类别 {self.target_class} 生成合成样本：没有原始样本")
            return []
            
        # 确定需要生成的样本数量
        if num_samples is None:
            num_samples = max(0, self.min_samples - len(target_indices))
            
        if num_samples <= 0:
            return []
            
        print(f"为类别 {self.target_class} 生成 {num_samples} 个合成样本")
        
        synthetic_samples = []
        
        # 根据不同方法生成合成样本
        if self.method == 'interpolation':
            synthetic_samples = self._generate_by_interpolation(target_indices, num_samples)
        elif self.method == 'jitter':
            synthetic_samples = self._generate_by_jitter(target_indices, num_samples)
        elif self.method == 'mask_mix':
            synthetic_samples = self._generate_by_mask_mix(target_indices, num_samples)
        else:
            raise ValueError(f"不支持的合成方法: {self.method}")
            
        return synthetic_samples
    
    def _generate_by_interpolation(self, target_indices, num_samples):
        """通过线性插值生成合成样本"""
        synthetic_samples = []
        
        for _ in range(num_samples):
            # 随机选择两个样本
            if len(target_indices) == 1:
                # 如果只有一个样本，只能复制它
                idx1 = idx2 = target_indices[0]
            else:
                idx1, idx2 = random.sample(target_indices, 2)
                
            # 获取样本
            sample1, _ = self.dataset[idx1]
            sample2, _ = self.dataset[idx2]
            
            # 确保两个样本形状相同
            if sample1.shape != sample2.shape:
                continue
                
            # 生成随机插值权重
            alpha = random.uniform(0.3, 0.7)
            
            # 创建新的合成样本
            synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
            
            synthetic_samples.append((synthetic_sample, self.target_class))
            
        return synthetic_samples
    
    def _generate_by_jitter(self, target_indices, num_samples):
        """通过添加随机抖动生成合成样本"""
        synthetic_samples = []
        
        for _ in range(num_samples):
            # 随机选择一个样本
            idx = random.choice(target_indices)
            
            # 获取样本
            sample, _ = self.dataset[idx]
            
            # 添加随机抖动
            noise = torch.randn_like(sample) * 0.1
            synthetic_sample = sample + noise
            
            # 确保值域在合理范围内
            synthetic_sample = torch.clamp(synthetic_sample, 0, 1)
            
            synthetic_samples.append((synthetic_sample, self.target_class))
            
        return synthetic_samples
    
    def _generate_by_mask_mix(self, target_indices, num_samples):
        """通过掩码混合生成合成样本"""
        synthetic_samples = []
        
        for _ in range(num_samples):
            # 随机选择两个样本
            if len(target_indices) == 1:
                idx1 = idx2 = target_indices[0]
            else:
                idx1, idx2 = random.sample(target_indices, 2)
                
            # 获取样本
            sample1, _ = self.dataset[idx1]
            sample2, _ = self.dataset[idx2]
            
            # 确保两个样本形状相同
            if sample1.shape != sample2.shape:
                continue
                
            # 创建随机掩码
            mask = torch.rand_like(sample1[0]).unsqueeze(0)
            mask = (mask > 0.5).float()
            
            # 将掩码复制到与样本相同的时间维度
            T = sample1.shape[0]
            mask = mask.repeat(T, 1, 1, 1)
            
            # 混合样本
            synthetic_sample = sample1 * mask + sample2 * (1 - mask)
            
            synthetic_samples.append((synthetic_sample, self.target_class))
            
        return synthetic_samples

class MixedDataset(torch.utils.data.Dataset):
    """合并原始数据集与合成样本"""
    
    def __init__(self, original_dataset, synthetic_samples=None):
        """
        Args:
            original_dataset: 原始数据集
            synthetic_samples: 合成样本列表[(sample1, label1), ...]
        """
        self.original_dataset = original_dataset
        self.synthetic_samples = synthetic_samples or []
        
        # 统计原始数据集中各类别的样本数量
        self.original_class_counts = defaultdict(int)
        for i in range(len(original_dataset)):
            try:
                label = original_dataset.get_label(i)
                self.original_class_counts[label] += 1
            except:
                # 处理子集情况
                if hasattr(original_dataset, 'indices'):
                    original_idx = original_dataset.indices[i]
                    label = original_dataset.dataset.get_label(original_idx)
                    self.original_class_counts[label] += 1
        
        # 统计合成样本中各类别的样本数量
        self.synthetic_class_counts = defaultdict(int)
        for _, label in self.synthetic_samples:
            self.synthetic_class_counts[label] += 1
            
        # 汇总统计
        print("数据集统计:")
        print(f"原始样本总数: {len(original_dataset)}")
        print(f"合成样本总数: {len(self.synthetic_samples)}")
        print("类别分布:")
        
        for label in sorted(set(list(self.original_class_counts.keys()) + list(self.synthetic_class_counts.keys()))):
            orig_count = self.original_class_counts.get(label, 0)
            synth_count = self.synthetic_class_counts.get(label, 0)
            total = orig_count + synth_count
            print(f"  类别 {label}: {total} 样本 (原始: {orig_count}, 合成: {synth_count})")
    
    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_samples)
    
    def __getitem__(self, idx):
        # 如果索引超出原始数据集范围，返回合成样本
        if idx >= len(self.original_dataset):
            synth_idx = idx - len(self.original_dataset)
            sample, label = self.synthetic_samples[synth_idx]
            return sample, torch.tensor(label, dtype=torch.long)
        else:
            # 返回原始数据集中的样本
            return self.original_dataset[idx]
    
    def get_label(self, idx):
        """获取样本标签"""
        if idx >= len(self.original_dataset):
            synth_idx = idx - len(self.original_dataset)
            _, label = self.synthetic_samples[synth_idx]
            return label
        else:
            # 获取原始数据集中样本的标签
            try:
                return self.original_dataset.get_label(idx)
            except:
                # 处理子集情况
                if hasattr(self.original_dataset, 'indices'):
                    original_idx = self.original_dataset.indices[idx]
                    return self.original_dataset.dataset.get_label(original_idx)
                raise

# 使用示例:
"""
# 为少数类生成合成样本
generator = SyntheticSampleGenerator(
    dataset=train_dataset,
    target_class=4,  # 'fear' 类别
    min_samples=20,
    method='interpolation'
)

# 生成样本
synthetic_samples = generator.generate_samples()

# 创建混合数据集
mixed_dataset = MixedDataset(train_dataset, synthetic_samples)

# 使用混合数据集创建 DataLoader
train_loader = DataLoader(
    mixed_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
""" 