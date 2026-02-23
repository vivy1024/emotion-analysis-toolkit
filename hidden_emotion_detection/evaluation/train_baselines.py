#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
传统分类器 Baseline（SVM / MLP / RandomForest）

从预提取的 (T, 14) ROI 光流特征中提取统计特征，用传统分类器做 LOSO 评估。
用于对比 Transformer 的时序建模是否有价值。

用法:
    python -m hidden_emotion_detection.evaluation.train_baselines \
        --features_dir hidden_emotion_detection/evaluation/features_casme2
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Baselines")


def extract_stat_features(seq):
    """从 (T, 14) 序列中提取统计特征 → 70 维"""
    # 每个通道: mean, std, max, min, range
    feats = []
    for ch in range(seq.shape[1]):
        col = seq[:, ch]
        feats.extend([col.mean(), col.std(), col.max(), col.min(), col.max() - col.min()])
    return np.array(feats, dtype=np.float32)


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
    return np.mean(per_class_f1), np.mean(per_class_recall)


def run_loso_baseline(features_dir, classifier_name='svm_rbf'):
    """LOSO 评估传统分类器"""
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    manifest_path = os.path.join(features_dir, 'manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 按被试组织数据
    subject_data = defaultdict(lambda: {'features': [], 'labels': []})
    for m in manifest:
        subj = m['subject']
        npy_path = os.path.join(features_dir, m['npy_file'])
        if not os.path.exists(npy_path):
            continue
        seq = np.load(npy_path)
        if len(seq) < 3:
            continue
        stat_feat = extract_stat_features(seq)
        subject_data[subj]['features'].append(stat_feat)
        subject_data[subj]['labels'].append(m['label'])

    all_subjects = sorted(subject_data.keys())
    all_preds = []
    all_labels = []

    for test_subj in all_subjects:
        if not subject_data[test_subj]['features']:
            continue

        # 划分训练/测试
        train_X, train_y = [], []
        for subj in all_subjects:
            if subj == test_subj:
                continue
            train_X.extend(subject_data[subj]['features'])
            train_y.extend(subject_data[subj]['labels'])

        test_X = np.array(subject_data[test_subj]['features'])
        test_y = np.array(subject_data[test_subj]['labels'])
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        # 标准化
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # 动态 class_weight
        if classifier_name == 'svm_rbf':
            clf = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
        elif classifier_name == 'svm_linear':
            clf = SVC(kernel='linear', class_weight='balanced', C=1.0)
        elif classifier_name == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                early_stopping=True, random_state=42)
        elif classifier_name == 'rf':
            clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         random_state=42)
        else:
            raise ValueError(f"未知分类器: {classifier_name}")

        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        all_preds.extend(preds)
        all_labels.extend(test_y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    uf1, uar = compute_uf1_uar(all_preds, all_labels, num_classes=4)
    return uf1, uar, len(all_labels)


def main():
    parser = argparse.ArgumentParser(description='传统分类器 Baseline LOSO 评估')
    parser.add_argument('--features_dir', type=str,
                        default=r'hidden_emotion_detection\evaluation\features_casme2')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = project_root / features_dir

    classifiers = ['svm_rbf', 'svm_linear', 'mlp', 'rf']
    results = {}

    print("\n" + "=" * 60)
    print("传统分类器 Baseline LOSO 评估")
    print("=" * 60)
    print(f"{'分类器':<15} {'UF1':>8} {'UAR':>8} {'样本数':>8}")
    print("-" * 45)

    for clf_name in classifiers:
        uf1, uar, n = run_loso_baseline(str(features_dir), clf_name)
        results[clf_name] = {'uf1': float(uf1), 'uar': float(uar), 'n_samples': n}
        print(f"{clf_name:<15} {uf1:>8.4f} {uar:>8.4f} {n:>8}")

    print("-" * 45)
    print(f"{'Transformer':<15} {'0.4712':>8} {'0.4991':>8} {'251':>8}  (参考)")
    print("=" * 60)

    # 保存结果
    output_path = project_root / '_baselines.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
