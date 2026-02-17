# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

情绪分析工具集（Emotion Analysis Toolkit），包含三个子项目：
- **hidden_emotion_detection/** — 实时视频隐藏情绪检测系统（PyQt5 GUI，6面板布局）
- **micro_expression/** — 微表情识别模型训练（CNN+LSTM两阶段）
- **macro_expression/** — 宏表情识别模型训练（支持FER2013等公开数据集）

Python 3.7+，核心依赖：PyTorch、OpenCV、PyQt5、dlib、MediaPipe。

## 常用命令

```bash
# 启动实时检测GUI
python -m hidden_emotion_detection.main
# 可选参数: --debug, --log-level DEBUG/INFO/WARNING

# 微表情 - CNN第一阶段训练
python -m micro_expression.train_cnn --config micro_expression/config_cnn_stage1.yaml

# 微表情 - 特征提取
python -m micro_expression.extract_features --config micro_expression/config_feature_extraction.yaml

# 微表情 - LSTM第二阶段训练
python -m micro_expression.train_lstm --config micro_expression/config_lstm_stage2.yaml

# 微表情 - 超参数搜索
python -m micro_expression.hpo_train --config micro_expression/config_lstm_stage2.yaml

# 宏表情训练
python -m macro_expression.multi_dataset_train --dataset fer2013

# 安装依赖（各子项目有独立的requirements.txt）
pip install -r hidden_emotion_detection/requirements.txt
pip install -r micro_expression/requirements.txt
pip install -r macro_expression/requirements.txt
```

无正式测试框架（pytest/tox），无CI/CD流水线。

## 架构

### hidden_emotion_detection — 实时检测系统

数据流：
```
摄像头输入 → FaceDetectionEngine → PoseEstimator
                    ↓
     ┌──────────────┼──────────────┐
     ↓              ↓              ↓
MacroEmotionEngine MicroEmotionEngine AUEngine
     ↓              ↓              ↓
     └──────────────┼──────────────┘
                    ↓
          EmotionIntegrator
                    ↓
        HiddenEmotionEngine（冲突检测）
                    ↓
          6-Panel PyQt5 UI
```

关键分层：
- **core/** — 基础设施：`EventBus`（单例，异步发布/订阅）、`data_types.py`（EmotionType枚举、FaceDetection、EmotionResult、AUResult等数据结构）、`pipeline.py`（通用管道阶段）
- **engines/** — 各分析引擎：人脸检测(MediaPipe/dlib)、头部姿态估计、宏表情(7类)、微表情(CNN+LSTM集成)、AU检测(SVM分类器)、隐藏情绪(宏微冲突检测)、结果聚合
- **ui/** — PyQt5面板：`LayoutManager`编排6列布局，各面板(video/face/au_intensity/macro_emotion/micro_emotion/hidden_emotion)继承`BasePanel`
- **config/** — `ConfigManager`集中管理，配置文件为`config/config.json`

核心设计模式：事件驱动（EventBus松耦合）、多线程（线程池并行处理各引擎）、单例配置管理。

### micro_expression — 两阶段训练

1. **Stage 1 (CNN)**: `train_cnn.py` → 训练CNN特征提取器，输出 `.pth` 权重
2. **特征提取**: `extract_features.py` → 用训练好的CNN提取序列特征
3. **Stage 2 (LSTM)**: `train_lstm.py` → 在提取的特征上训练LSTM时序模型，使用StratifiedGroupKFold交叉验证

关键模块：`models.py`（CNNModel、LSTMOnlyModel等架构定义）、`dataset.py`（图像数据加载）、`feature_dataset.py`（特征序列数据加载）、`transforms.py`（数据增强）、`utils.py`（情绪映射常量 MICRO_EMOTION_MAP、NUM_CLASSES）

训练配置通过YAML文件控制（`config_cnn_stage1.yaml`、`config_lstm_stage2.yaml`、`config_feature_extraction.yaml`）。

### 模型文件路径

配置在 `hidden_emotion_detection/config/config.json` 中，默认指向本地绝对路径 `D:/pycharm2/PythonProject2/enhance_hidden/models/`。部署到新环境时需修改这些路径。

## 代码约定

- 项目语言为中文，注释和文档使用中文
- 各子项目作为Python包使用相对导入（`from .utils import ...`），需以 `-m` 方式运行
- 微表情训练输出保存在 `stage2_lstm_output/` 和 `stage1_cnn_output/`
- AU检测子模块位于 `engines/au_detection/`，使用预训练SVM分类器
- `enhance_hidden/` 目录存放增强版开发代码和模型文件
