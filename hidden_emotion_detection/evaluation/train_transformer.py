#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Transformer 训练/评估脚本（纯 PyTorch，不依赖 cv2）

从预提取的 .npy 特征文件加载数据，在 CASME II 上进行 LOSO 交叉验证。

两阶段流程:
    1. 先运行 extract_roi_features.py 提取特征（需要 cv2 + dlib）
    2. 再运行本脚本训练（纯 PyTorch）

用法:
    python -m hidden_emotion_detection.evaluation.train_transformer \
        --features_dir hidden_emotion_detection/evaluation/features_casme2 \
        --epochs 30 --lr 0.001

该脚本不修改现有 micro_expression/ 下的训练架构，
而是作为独立的评估模块运行。
"""

import argparse
import logging
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TransformerTrain")


# ============================================================
# 数据集
# ============================================================

class CASME2ROIDataset(torch.utils.data.Dataset):
    """CASME II ROI 光流特征数据集"""

    # CASME II 情绪映射到 MEGC 4类
    EMOTION_MAP_4CLASS = {
        'happiness': 1,    # positive
        'surprise': 2,     # surprise
        'disgust': 3,      # negative
        'repression': 3,   # negative
        'fear': 3,         # negative
        'sadness': 3,      # negative
        'others': 0,       # others
        'contempt': 3,     # negative
        'tense': 3,        # negative
    }

    def __init__(self, features_list, labels_list, max_seq_len=64, num_rois=7):
        self.features = features_list  # list of (T, 14) arrays
        self.labels = labels_list      # list of int
        self.max_seq_len = max_seq_len
        self.num_rois = num_rois

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # (T, 14)
        label = self.labels[idx]

        # 截断或填充到 max_seq_len
        t = feat.shape[0]
        if t >= self.max_seq_len:
            feat = feat[:self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - t, feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


# ============================================================
# 训练与评估
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for features, labels in dataloader:
        features = features.to(device)
        logits = model(features)
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_uf1_uar(preds, labels, num_classes=4):
    """计算 UF1 和 UAR"""
    per_class_f1 = []
    per_class_recall = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1.append(f1)
        per_class_recall.append(recall)
    uf1 = np.mean(per_class_f1)
    uar = np.mean(per_class_recall)
    return uf1, uar


def run_loso_training(features_dir: str, config: dict) -> dict:
    """
    LOSO 训练 + 评估 ROITransformerModel

    从预提取的 .npy 特征文件加载数据，不依赖 cv2。
    不修改 micro_expression/ 下的任何训练代码。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from micro_expression.models import ROITransformerModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    # 从 manifest.json 加载样本清单
    manifest_path = os.path.join(features_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"找不到 manifest.json: {manifest_path}\n"
                                f"请先运行 extract_roi_features.py 提取特征")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    all_subjects = sorted(set(m['subject'] for m in manifest))
    if config.get('subjects'):
        all_subjects = [s for s in all_subjects if s in config['subjects']]

    # 按被试组织数据
    subject_data = defaultdict(lambda: {'features': [], 'labels': [], 'videos': []})

    logger.info(f"从 {features_dir} 加载预提取特征...")
    for m in manifest:
        subj = m['subject']
        if subj not in all_subjects:
            continue
        video = m['video']
        label = m['label']
        key = f"{subj}_{video}"
        if key in subject_data[subj]['videos']:
            continue

        npy_path = os.path.join(features_dir, m['npy_file'])
        if not os.path.exists(npy_path):
            logger.warning(f"跳过 {key}: npy 文件不存在")
            continue

        feat = np.load(npy_path)
        if len(feat) < 3:
            continue

        subject_data[subj]['features'].append(feat)
        subject_data[subj]['labels'].append(label)
        subject_data[subj]['videos'].append(key)

    logger.info(f"加载完成，共 {sum(len(v['labels']) for v in subject_data.values())} 个样本")

    # LOSO 训练
    num_classes = 4
    max_seq_len = config.get('max_seq_len', 64)
    all_preds = []
    all_labels = []
    per_subject_results = {}

    for test_subj in all_subjects:
        if not subject_data[test_subj]['features']:
            continue

        # 划分训练/测试
        train_feats, train_labels = [], []
        for subj in all_subjects:
            if subj == test_subj:
                continue
            train_feats.extend(subject_data[subj]['features'])
            train_labels.extend(subject_data[subj]['labels'])

        test_feats = subject_data[test_subj]['features']
        test_labels = subject_data[test_subj]['labels']

        if not train_feats or not test_feats:
            continue

        # 创建数据集
        train_ds = CASME2ROIDataset(train_feats, train_labels, max_seq_len)
        test_ds = CASME2ROIDataset(test_feats, test_labels, max_seq_len)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.get('batch_size', 32),
            shuffle=True, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=32, shuffle=False
        )

        # 创建模型（每个 fold 重新初始化）
        model = ROITransformerModel(
            num_rois=7,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
            dim=config.get('dim', 64),
            depth=config.get('depth', 3),
            heads=config.get('heads', 4),
            mlp_dim=config.get('mlp_dim', 128),
            dropout=config.get('dropout', 0.1),
        ).to(device)

        # 加权损失
        class_weights = torch.FloatTensor([1, 3, 3, 3]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('epochs', 30)
        )

        # 训练
        best_uf1 = 0
        for epoch in range(config.get('epochs', 30)):
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

        # 评估
        preds, labels = evaluate(model, test_loader, device)
        uf1, uar = compute_uf1_uar(preds, labels, num_classes)

        per_subject_results[test_subj] = {
            'n_samples': len(test_labels),
            'uf1': float(uf1),
            'uar': float(uar),
        }
        logger.info(f"sub{test_subj}: {len(test_labels)} 样本, UF1={uf1:.4f}, UAR={uar:.4f}")

        all_preds.extend(preds)
        all_labels.extend(labels)

    # 全局指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    global_uf1, global_uar = compute_uf1_uar(all_preds, all_labels, num_classes)

    logger.info(f"全局: UF1={global_uf1:.4f}, UAR={global_uar:.4f}")

    return {
        'global': {'uf1': float(global_uf1), 'uar': float(global_uar),
                    'n_samples': len(all_labels)},
        'per_subject': per_subject_results,
        'config': config,
    }


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ROI Transformer CASME II 训练')
    parser.add_argument('--features_dir', type=str,
                        default=r'hidden_emotion_detection\evaluation\features_casme2',
                        help='预提取特征目录（含 manifest.json）')
    parser.add_argument('--subjects', nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--output_json', type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = project_root / features_dir

    config = {
        'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
        'dim': args.dim, 'depth': args.depth, 'heads': args.heads,
        'mlp_dim': args.dim * 2, 'max_seq_len': args.max_seq_len,
        'dropout': 0.1, 'weight_decay': 0.01,
        'subjects': args.subjects,
    }

    results = run_loso_training(str(features_dir), config)

    # 输出
    print("\n" + "=" * 50)
    print("ROI Transformer LOSO 评估结果")
    print("=" * 50)
    g = results['global']
    print(f"样本数: {g['n_samples']}")
    print(f"UF1: {g['uf1']:.4f}")
    print(f"UAR: {g['uar']:.4f}")
    print("=" * 50)

    if args.output_json:
        json_path = Path(args.output_json)
        if not json_path.is_absolute():
            json_path = project_root / json_path
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存: {json_path}")


if __name__ == '__main__':
    main()
