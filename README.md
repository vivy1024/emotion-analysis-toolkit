# Emotion Analysis Toolkit

实时视频隐藏情绪检测系统 + 微表情/宏表情模型训练工具集。

## 项目结构

```
emotion-analysis-toolkit/
├── hidden_emotion_detection/   # 实时视频隐藏情绪检测系统（PyQt5 GUI）
│   └── evaluation/             #   离线评估框架（CASME II、STRS）
├── micro_expression/           # 微表情识别模型训练（CNN+LSTM 两阶段）
├── macro_expression/           # 宏表情识别模型训练（FER2013 等）
├── docs/                       # 项目文档
└── references/                 # 参考实现（MEGC 竞赛代码）
```

## 技术栈

PyTorch · OpenCV · PyQt5 · dlib · MediaPipe · scikit-learn · Python 3.7+

## 系统架构

```
摄像头 → FaceDetectionEngine → PoseEstimator
                ↓
     ┌──────────┼──────────┐
     ↓          ↓          ↓
  Macro      Micro       AU
  Engine     Engine     Engine
     ↓   OpticalFlow    ↓
     ↓     Spotting     ↓
     └──────────┼──────────┘
                ↓
      EmotionIntegrator
                ↓
    HiddenEmotionEngine（宏微冲突检测）
                ↓
        6-Panel PyQt5 UI
```

核心思路：当宏表情（外显）与微表情（真实）不一致时，判定为隐藏情绪。

## 快速开始

```bash
# 安装依赖
pip install -r hidden_emotion_detection/requirements.txt

# 启动实时检测 GUI
python -m hidden_emotion_detection.main

# 微表情训练（两阶段）
python -m micro_expression.train_cnn --config micro_expression/config_cnn_stage1.yaml
python -m micro_expression.extract_features --config micro_expression/config_feature_extraction.yaml
python -m micro_expression.train_lstm --config micro_expression/config_lstm_stage2.yaml

# 宏表情训练
python -m macro_expression.multi_dataset_train --dataset fer2013

# 离线评估（CASME II）
python -m hidden_emotion_detection.evaluation.extract_roi_features
python -m hidden_emotion_detection.evaluation.train_transformer
```

## 子项目说明

### 实时隐藏情绪检测系统

多引擎并行的实时视频分析系统，基于事件驱动架构（EventBus），包含人脸检测、头部姿态估计、宏表情识别、微表情检测、AU 检测、光流 Spotting 等引擎，通过 EmotionIntegrator 聚合结果，由 HiddenEmotionEngine 进行宏微冲突检测。

### 微表情训练工具集

CNN+LSTM 两阶段训练流程：Stage 1 训练 CNN 特征提取器，Stage 2 用 LSTM 建模时序关系。支持 StratifiedGroupKFold 和 LOSO 交叉验证。

### 宏表情训练工具集

基于 ResNet18 的宏表情分类器，支持 FER2013 等公开数据集，通过 `get_emotion_model` 工厂函数创建模型。

### 离线评估框架

基于 MEGC2024 CCS 冠军方案的 7-ROI Farneback 光流特征提取 + ROI Transformer 分类器，采用 LOSO 协议在 CASME II 上评估，指标为 UF1/UAR。

## 文档

| 文档 | 说明 |
|------|------|
| [系统架构](docs/02-核心架构/01-系统架构.md) | 分层架构、数据流、设计模式 |
| [微表情训练](docs/03-训练指南/01-微表情训练.md) | CNN+LSTM 两阶段训练指南 |
| [宏表情训练](docs/03-训练指南/02-宏表情训练.md) | 多数据集训练指南 |
| [评估框架](docs/03-训练指南/04-评估框架使用指南.md) | CASME II 离线评估使用方法 |
| [领域参考索引](docs/03-训练指南/03-微表情领域参考索引.md) | 论文、竞赛、方法演进 |
| [目录与存储规范](docs/04-开发指南/04-目录结构与存储规范.md) | 命名规则、存储规范 |

## 作者

薛小川 · [GitHub](https://github.com/vivy1024) · 1336495069@qq.com
