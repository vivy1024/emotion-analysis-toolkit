#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI 光流特征提取 + Transformer 训练/评估脚本
在 CASME II 上进行 LOSO 交叉验证，计算完整 STRS 指标。

用法:
    python -m hidden_emotion_detection.evaluation.train_transformer \
        --data_root "旧有文件/18/data/raw/CASME II" \
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
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TransformerTrain")


# ============================================================
# ROI 光流特征提取器
# ============================================================

class ROIFlowFeatureExtractor:
    """
    从 CASME II 图片序列中提取 7-ROI 光流特征

    输出: (T-1, 14) — T-1帧的光流，每帧 7 ROI × 2 (dx, dy)
    """

    ROI_DEFINITIONS = {
        'left_eyebrow': {'landmarks': list(range(17, 22)), 'percent': 0.3},
        'right_eyebrow': {'landmarks': list(range(22, 27)), 'percent': 0.3},
        'left_eye': {'landmarks': list(range(36, 42)), 'percent': 0.2},
        'right_eye': {'landmarks': list(range(42, 48)), 'percent': 0.2},
        'nose': {'landmarks': list(range(27, 36)), 'percent': 0.2},
        'mouth_outer': {'landmarks': list(range(48, 60)), 'percent': 0.3},
        'mouth_inner': {'landmarks': list(range(60, 68)), 'percent': 0.3},
    }
    GLOBAL_REF = [29, 30, 31]
    ROI_NAMES = list(ROI_DEFINITIONS.keys())

    def __init__(self, predictor_path: str):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        # 处理中文路径
        try:
            predictor_path.encode('ascii')
            self.predictor = dlib.shape_predictor(predictor_path)
        except (UnicodeEncodeError, UnicodeDecodeError):
            import tempfile, shutil
            tmp = os.path.join(tempfile.gettempdir(), 'shape_predictor_68.dat')
            if not os.path.exists(tmp):
                shutil.copy2(predictor_path, tmp)
            self.predictor = dlib.shape_predictor(tmp)

        self.flow_params = dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

    def _detect_landmarks(self, gray):
        import dlib
        faces = self.detector(gray, 0)
        if len(faces) == 0:
            h, w = gray.shape
            faces = [dlib.rectangle(0, 0, w, h)]
        shape = self.predictor(gray, faces[0])
        return np.array([[shape.part(i).x, shape.part(i).y]
                         for i in range(68)], dtype=np.float32)

    def _imread_unicode(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        try:
            with open(path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

    def _get_roi_flow_vec(self, flow, landmarks, indices, percent, expand=5):
        """提取单个 ROI 的光流向量 (dx, dy)"""
        h, w = flow.shape[:2]
        pts = landmarks[indices].astype(int)
        x_min = max(0, pts[:, 0].min() - expand)
        x_max = min(w, pts[:, 0].max() + expand)
        y_min = max(0, pts[:, 1].min() - expand)
        y_max = min(h, pts[:, 1].max() + expand)
        if x_max <= x_min or y_max <= y_min:
            return 0.0, 0.0
        roi = flow[y_min:y_max, x_min:x_max]
        fx, fy = roi[:, :, 0].flatten(), roi[:, :, 1].flatten()
        mag = np.sqrt(fx ** 2 + fy ** 2)
        if len(mag) == 0:
            return 0.0, 0.0
        threshold = np.percentile(mag, (1 - percent) * 100)
        mask = mag >= threshold
        if mask.sum() == 0:
            return float(fx.mean()), float(fy.mean())
        return float(fx[mask].mean()), float(fy[mask].mean())

    def extract_features(self, data_root: str, subject: str, video: str) -> Optional[np.ndarray]:
        """
        提取一个视频的 ROI 光流特征

        Returns:
            np.ndarray: (T-1, 14) 或 None
        """
        video_dir = os.path.join(data_root, f'sub{subject}', video)
        if not os.path.isdir(video_dir):
            return None

        # 加载帧
        frames = []
        for fname in os.listdir(video_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.bmp')):
                continue
            try:
                fid = int(''.join(filter(str.isdigit, os.path.splitext(fname)[0])))
            except ValueError:
                continue
            img = self._imread_unicode(os.path.join(video_dir, fname))
            if img is not None:
                frames.append((fid, img))
        frames.sort(key=lambda x: x[0])

        if len(frames) < 5:
            return None

        # 检测关键点
        landmarks_list = []
        grays = []
        for _, gray in frames:
            lm = self._detect_landmarks(gray)
            landmarks_list.append(lm)
            grays.append(gray)

        # 帧对齐
        ref_lm = landmarks_list[0]
        src_pts = np.float32([ref_lm[39], ref_lm[42], ref_lm[33]])
        h, w = grays[0].shape[:2]
        aligned = [grays[0]]
        for i in range(1, len(grays)):
            lm = landmarks_list[i]
            dst_pts = np.float32([lm[39], lm[42], lm[33]])
            try:
                M = cv2.getAffineTransform(dst_pts, src_pts)
                warped = cv2.warpAffine(grays[i], M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                aligned.append(warped)
            except cv2.error:
                aligned.append(grays[i])

        # 计算光流特征
        n = len(aligned)
        features = np.zeros((n - 1, len(self.ROI_NAMES) * 2), dtype=np.float32)

        for i in range(n - 1):
            flow = cv2.calcOpticalFlowFarneback(
                aligned[i], aligned[i + 1], None, **self.flow_params
            )
            lm = landmarks_list[min(i, len(landmarks_list) - 1)]

            # 全局运动补偿
            gx, gy = self._get_roi_flow_vec(flow, lm, self.GLOBAL_REF, 0.5, 10)

            for ri, (name, roi_def) in enumerate(self.ROI_DEFINITIONS.items()):
                dx, dy = self._get_roi_flow_vec(
                    flow, lm, roi_def['landmarks'], roi_def['percent'], 5
                )
                features[i, ri * 2] = dx - gx
                features[i, ri * 2 + 1] = dy - gy

        return features


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
    from collections import Counter
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


def run_loso_training(data_root: str, predictor_path: str,
                      annotations: list, config: dict) -> dict:
    """
    LOSO 训练 + 评估 ROITransformerModel

    不修改 micro_expression/ 下的任何训练代码。
    """
    # 延迟导入，避免循环依赖
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from micro_expression.models import ROITransformerModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    # 提取所有特征
    extractor = ROIFlowFeatureExtractor(predictor_path)
    all_subjects = sorted(set(a['subject'] for a in annotations))

    if config.get('subjects'):
        all_subjects = [s for s in all_subjects if s in config['subjects']]

    # 按被试组织数据
    subject_data = defaultdict(lambda: {'features': [], 'labels': [], 'videos': []})

    logger.info(f"提取 {len(all_subjects)} 个被试的 ROI 光流特征...")
    for ann in annotations:
        subj = ann['subject']
        if subj not in all_subjects:
            continue
        video = ann['video']
        emotion = ann.get('emotion', 'others').lower()
        label = CASME2ROIDataset.EMOTION_MAP_4CLASS.get(emotion, 0)

        # 检查是否已提取
        key = f"{subj}_{video}"
        if key in subject_data[subj]['videos']:
            continue

        feat = extractor.extract_features(data_root, subj, video)
        if feat is None or len(feat) < 3:
            continue

        # 只取 onset-offset 区间的特征（如果标注有效）
        onset_idx = max(0, ann['onset'] - 1)  # 帧号转索引
        offset_idx = min(len(feat), ann['offset'])
        if offset_idx > onset_idx and (offset_idx - onset_idx) >= 3:
            segment_feat = feat[onset_idx:offset_idx]
        else:
            segment_feat = feat  # fallback: 用整个视频

        subject_data[subj]['features'].append(segment_feat)
        subject_data[subj]['labels'].append(label)
        subject_data[subj]['videos'].append(key)

    logger.info(f"特征提取完成，共 {sum(len(v['labels']) for v in subject_data.values())} 个样本")

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
    parser.add_argument('--data_root', type=str,
                        default=r'旧有文件\18\data\raw\CASME II')
    parser.add_argument('--predictor', type=str,
                        default=r'hidden_emotion_detection\models\shape_predictor_68_face_landmarks.dat')
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
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    predictor_path = Path(args.predictor)
    if not predictor_path.is_absolute():
        predictor_path = project_root / predictor_path

    # 加载标注
    from .evaluate_casme2 import load_casme2_annotations
    annotations = load_casme2_annotations(str(data_root))

    config = {
        'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
        'dim': args.dim, 'depth': args.depth, 'heads': args.heads,
        'mlp_dim': args.dim * 2, 'max_seq_len': args.max_seq_len,
        'dropout': 0.1, 'weight_decay': 0.01,
        'subjects': args.subjects,
    }

    results = run_loso_training(str(data_root), str(predictor_path), annotations, config)

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
