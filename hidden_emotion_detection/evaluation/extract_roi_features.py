#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI 光流特征提取脚本（独立于训练，需要 cv2 + dlib）

将 CASME II 每个标注样本的 onset-offset 区间 ROI 光流特征
提取并保存为 .npy 文件，供 train_transformer.py 直接加载。

用法（在 PyCharm 终端或有 cv2 的环境中运行）:
    python -m hidden_emotion_detection.evaluation.extract_roi_features \
        --data_root "旧有文件/18/data/raw/CASME II" \
        --output_dir hidden_emotion_detection/evaluation/features_casme2

输出目录结构:
    features_casme2/
        sub01_EP02_01f.npy      # (T, 14) float32
        sub01_EP02_01f.json     # 元信息 {subject, video, emotion, label, onset, offset}
        ...
        manifest.json           # 所有样本清单
"""

import argparse
import gc
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ExtractROI")


def _imread_unicode(path):
    """处理中文路径的图片读取"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


class ROIFlowFeatureExtractor:
    """从 CASME II 图片序列中提取 7-ROI 光流特征"""

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

    def _detect_landmarks(self, gray, face_rect=None):
        """检测68个关键点。如果提供 face_rect 则跳过人脸检测直接预测。"""
        import dlib
        if face_rect is None:
            faces = self.detector(gray, 0)
            if len(faces) == 0:
                h, w = gray.shape
                face_rect = dlib.rectangle(0, 0, w, h)
            else:
                face_rect = faces[0]
        shape = self.predictor(gray, face_rect)
        return np.array([[shape.part(i).x, shape.part(i).y]
                         for i in range(68)], dtype=np.float32), face_rect

    def _get_roi_flow_vec(self, flow, landmarks, indices, percent, expand=5):
        import cv2
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

    def extract_segment(self, data_root: str, subject: str, video: str,
                        onset: int, offset: int):
        """
        提取一个视频 onset-offset 区间的 ROI 光流特征

        Returns:
            np.ndarray: (T-1, 14) 或 None
        """
        import cv2
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
            frames.append((fid, os.path.join(video_dir, fname)))
        frames.sort(key=lambda x: x[0])

        if len(frames) < 5:
            return None

        # 加载图片
        grays = []
        for fid, fpath in frames:
            img = _imread_unicode(fpath)
            if img is not None:
                grays.append((fid, img))

        if len(grays) < 5:
            return None

        # 检测关键点（只在第一帧做完整人脸检测，后续帧复用 bbox）
        landmarks_list = []
        gray_imgs = []
        face_rect = None
        for _, gray in grays:
            lm, face_rect = self._detect_landmarks(gray, face_rect)
            landmarks_list.append(lm)
            gray_imgs.append(gray)

        # 帧对齐
        ref_lm = landmarks_list[0]
        src_pts = np.float32([ref_lm[39], ref_lm[42], ref_lm[33]])
        h, w = gray_imgs[0].shape[:2]
        aligned = [gray_imgs[0]]
        for i in range(1, len(gray_imgs)):
            lm = landmarks_list[i]
            dst_pts = np.float32([lm[39], lm[42], lm[33]])
            try:
                M = cv2.getAffineTransform(dst_pts, src_pts)
                warped = cv2.warpAffine(gray_imgs[i], M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                aligned.append(warped)
            except cv2.error:
                aligned.append(gray_imgs[i])

        # 计算光流特征（逐帧计算，及时释放内存）
        n = len(aligned)
        features = np.zeros((n - 1, len(self.ROI_NAMES) * 2), dtype=np.float32)

        for i in range(n - 1):
            flow = cv2.calcOpticalFlowFarneback(
                aligned[i], aligned[i + 1], None, **self.flow_params
            )
            lm = landmarks_list[min(i, len(landmarks_list) - 1)]
            gx, gy = self._get_roi_flow_vec(flow, lm, self.GLOBAL_REF, 0.3, 5)

            for ri, (name, roi_def) in enumerate(self.ROI_DEFINITIONS.items()):
                dx, dy = self._get_roi_flow_vec(
                    flow, lm, roi_def['landmarks'], roi_def['percent'], 5
                )
                features[i, ri * 2] = dx - gx
                features[i, ri * 2 + 1] = dy - gy
            del flow  # 及时释放光流矩阵

        # 截取 onset-offset 区间（先提取 frame_ids 再释放大数组）
        frame_ids = [fid for fid, _ in grays]
        del aligned, gray_imgs, landmarks_list, grays

        onset_idx = None
        offset_idx = None
        for idx, fid in enumerate(frame_ids[:-1]):  # features 比 frames 少1
            if onset_idx is None and fid >= onset:
                onset_idx = idx
            if fid >= offset:
                offset_idx = idx
                break

        # 如果 onset 未匹配到，尝试用最接近的帧
        if onset_idx is None:
            onset_idx = 0
            logger.debug(f"  onset={onset} 未匹配到帧号，使用首帧 fid={frame_ids[0]}")
        if offset_idx is None:
            offset_idx = len(features)
            logger.debug(f"  offset={offset} 未匹配到帧号，使用末帧 fid={frame_ids[-1]}")

        segment = features[onset_idx:offset_idx]

        # 如果截取后太短，记录警告，取 onset 附近上下文窗口而非整个视频
        if len(segment) < 3:
            logger.warning(
                f"  segment 过短({len(segment)}帧), onset={onset}, offset={offset}, "
                f"帧范围=[{frame_ids[0]}..{frame_ids[-1]}], 总帧数={len(features)}"
            )
            center = onset_idx if onset_idx is not None else len(features) // 2
            start = max(0, center - 8)
            end = min(len(features), center + 8)
            segment = features[start:end]

        return segment


# CASME II 情绪映射到 MEGC 4类
EMOTION_MAP_4CLASS = {
    'happiness': 1,
    'surprise': 2,
    'disgust': 3,
    'repression': 3,
    'fear': 3,
    'sadness': 3,
    'others': 0,
    'contempt': 3,
    'tense': 3,
}


def main():
    parser = argparse.ArgumentParser(description='CASME II ROI 光流特征提取')
    parser.add_argument('--data_root', type=str,
                        default=r'旧有文件\18\data\raw\CASME II')
    parser.add_argument('--predictor', type=str,
                        default=r'hidden_emotion_detection\models\shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--output_dir', type=str,
                        default=r'hidden_emotion_detection\evaluation\features_casme2')
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='指定被试编号，如 09 15，不指定则提取全部')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    predictor_path = Path(args.predictor)
    if not predictor_path.is_absolute():
        predictor_path = project_root / predictor_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注
    sys.path.insert(0, str(project_root))
    from hidden_emotion_detection.evaluation.evaluate_casme2 import load_casme2_annotations
    annotations = load_casme2_annotations(str(data_root))

    if args.subjects:
        annotations = [a for a in annotations if a['subject'] in args.subjects]

    logger.info(f"共 {len(annotations)} 个标注样本")

    extractor = ROIFlowFeatureExtractor(str(predictor_path))
    manifest = []
    extracted = 0
    skipped = 0

    for i, ann in enumerate(annotations):
        subj = ann['subject']
        video = ann['video']
        emotion = ann.get('emotion', 'others').lower()
        label = EMOTION_MAP_4CLASS.get(emotion, 0)
        onset = ann.get('onset', 0)
        offset = ann.get('offset', 0)

        sample_id = f"sub{subj}_{video}"
        npy_path = output_dir / f"{sample_id}.npy"

        # 跳过已提取的
        if npy_path.exists():
            logger.info(f"[{i+1}/{len(annotations)}] {sample_id} 已存在，跳过")
            manifest.append({
                'sample_id': sample_id,
                'subject': subj,
                'video': video,
                'emotion': emotion,
                'label': label,
                'onset': onset,
                'offset': offset,
                'npy_file': f"{sample_id}.npy",
            })
            extracted += 1
            continue

        logger.info(f"[{i+1}/{len(annotations)}] 提取 {sample_id} ...")
        try:
            feat = extractor.extract_segment(str(data_root), subj, video, onset, offset)
        except (MemoryError, Exception) as e:
            logger.warning(f"  跳过 {sample_id}: {type(e).__name__}: {e}")
            gc.collect()
            skipped += 1
            continue

        if feat is None or len(feat) < 3:
            logger.warning(f"  跳过 {sample_id}: 特征不足")
            skipped += 1
            continue

        np.save(str(npy_path), feat)

        # 保存元信息
        meta = {
            'sample_id': sample_id,
            'subject': subj,
            'video': video,
            'emotion': emotion,
            'label': label,
            'onset': onset,
            'offset': offset,
            'npy_file': f"{sample_id}.npy",
            'shape': list(feat.shape),
        }
        with open(output_dir / f"{sample_id}.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        manifest.append(meta)
        extracted += 1
        logger.info(f"  保存: {feat.shape}")
        del feat
        gc.collect()

    # 保存清单
    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n提取完成: {extracted} 个样本, 跳过 {skipped} 个")
    logger.info(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
