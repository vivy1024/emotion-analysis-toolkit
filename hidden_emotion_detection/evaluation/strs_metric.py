#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STRS 评估指标
参考 MEGC 官方评估脚本 (STRS-Metric)，适配 pandas 2.x（去除 append）

STRS = F1_spotting × F1_analysis
- Spotting: IoU ≥ 0.5 匹配，计算 F1
- Analysis: 基于 Spotting TP，按情绪类别计算 F1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set


class STRSEvaluator:
    """STRS 评估器"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def _calculate_iou(pred_onset: int, pred_offset: int,
                       gt_onset: int, gt_offset: int) -> float:
        """计算两个时间段的 IoU（帧级别）"""
        pred_range = set(range(pred_onset, pred_offset + 1))
        gt_range = set(range(gt_onset, gt_offset + 1))
        intersection = len(pred_range & gt_range)
        union = len(pred_range | gt_range)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _calc_f1(tp: int, fp: int, fn: int) -> tuple:
        """计算 precision, recall, f1"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def evaluate_spotting(self, predictions: List[dict],
                          ground_truths: List[dict]) -> dict:
        """
        评估 Spotting 任务

        Args:
            predictions: [{'subject': str, 'video': str, 'onset': int, 'offset': int, 'emotion': str}, ...]
            ground_truths: 同上格式

        Returns:
            {'tp': int, 'fp': int, 'fn': int, 'precision': float, 'recall': float, 'f1': float,
             'tp_pairs': [(pred, gt), ...]}
        """
        # 按 subject + video 分组
        gt_by_video = {}
        for gt in ground_truths:
            key = (gt['subject'], gt['video'])
            gt_by_video.setdefault(key, []).append(gt)

        pred_by_video = {}
        for pred in predictions:
            key = (pred['subject'], pred['video'])
            pred_by_video.setdefault(key, []).append(pred)

        total_tp, total_fp, total_fn = 0, 0, 0
        tp_pairs = []

        # 遍历所有有 GT 的视频
        all_keys = set(gt_by_video.keys()) | set(pred_by_video.keys())
        for key in all_keys:
            gts = gt_by_video.get(key, [])
            preds = pred_by_video.get(key, [])
            matched_gt = set()

            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                for gi, gt in enumerate(gts):
                    if gi in matched_gt:
                        continue
                    iou = self._calculate_iou(
                        pred['onset'], pred['offset'],
                        gt['onset'], gt['offset']
                    )
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    tp_pairs.append((pred, gts[best_gt_idx]))
                else:
                    total_fp += 1

            total_fn += len(gts) - len(matched_gt)

        precision, recall, f1 = self._calc_f1(total_tp, total_fp, total_fn)
        return {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp_pairs': tp_pairs
        }

    def evaluate_analysis(self, tp_pairs: List[tuple],
                          emotion_classes: Optional[Set[str]] = None) -> dict:
        """
        评估 Analysis 任务（基于 Spotting TP）

        Args:
            tp_pairs: [(pred_dict, gt_dict), ...]
            emotion_classes: 情绪类别集合

        Returns:
            {'f1': float, 'uf1': float, 'uar': float, 'per_class': {emotion: {tp, fp, fn, f1}}}
        """
        if not tp_pairs:
            return {'f1': 0.0, 'uf1': 0.0, 'uar': 0.0, 'per_class': {}}

        # 收集所有情绪类别
        if emotion_classes is None:
            emotion_classes = set()
            for pred, gt in tp_pairs:
                emotion_classes.add(gt.get('emotion', 'unknown'))

        per_class = {}
        for emotion in emotion_classes:
            tp = sum(1 for p, g in tp_pairs
                     if p.get('emotion') == emotion and g.get('emotion') == emotion)
            fp = sum(1 for p, g in tp_pairs
                     if p.get('emotion') == emotion and g.get('emotion') != emotion)
            fn = sum(1 for p, g in tp_pairs
                     if p.get('emotion') != emotion and g.get('emotion') == emotion)
            _, _, f1 = self._calc_f1(tp, fp, fn)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class[emotion] = {'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1, 'recall': recall}

        f1_list = [v['f1'] for v in per_class.values()]
        recall_list = [v['recall'] for v in per_class.values()]
        uf1 = np.mean(f1_list) if f1_list else 0.0
        uar = np.mean(recall_list) if recall_list else 0.0

        # 宏平均 F1
        avg_p = np.mean([v['tp'] / (v['tp'] + v['fp']) if (v['tp'] + v['fp']) > 0 else 0
                         for v in per_class.values()])
        avg_r = np.mean([v['recall'] for v in per_class.values()])
        f1_analysis = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

        return {'f1': f1_analysis, 'uf1': float(uf1), 'uar': float(uar), 'per_class': per_class}

    def evaluate_strs(self, predictions: List[dict],
                      ground_truths: List[dict],
                      emotion_classes: Optional[Set[str]] = None) -> dict:
        """
        完整 STRS 评估

        Returns:
            {'strs': float, 'spotting': {...}, 'analysis': {...}}
        """
        spot_result = self.evaluate_spotting(predictions, ground_truths)
        analysis_result = self.evaluate_analysis(spot_result['tp_pairs'], emotion_classes)

        strs = spot_result['f1'] * analysis_result['f1']

        return {
            'strs': strs,
            'spotting': {
                'tp': spot_result['tp'],
                'fp': spot_result['fp'],
                'fn': spot_result['fn'],
                'precision': spot_result['precision'],
                'recall': spot_result['recall'],
                'f1': spot_result['f1'],
            },
            'analysis': analysis_result
        }
