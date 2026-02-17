# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**版本**: v2.0.0 | **更新**: 2026-02-18 | **语言**: 始终使用中文回复

**项目**: 情绪分析工具集（Emotion Analysis Toolkit）| **开发者**: 薛小川

---

## 项目概览

实时视频隐藏情绪检测系统 + 微表情/宏表情模型训练工具集。

**技术栈**: PyTorch + OpenCV + PyQt5 + dlib + MediaPipe + scikit-learn
**Python**: 3.7+

| 子项目 | 目录 | 说明 |
|--------|------|------|
| 实时检测系统 | `hidden_emotion_detection/` | PyQt5 GUI，6面板布局，多引擎并行 |
| 微表情训练 | `micro_expression/` | CNN+LSTM 两阶段训练 |
| 宏表情训练 | `macro_expression/` | 支持 FER2013 等公开数据集 |

### 联系方式

| 开发者 | 邮箱 | GitHub |
|--------|------|--------|
| 薛小川 | 1336495069@qq.com | vivy1024 |

---

## 常用命令

```bash
# 启动实时检测 GUI
python -m hidden_emotion_detection.main
# 可选: --debug, --log-level DEBUG/INFO/WARNING

# 微表情 - CNN 第一阶段
python -m micro_expression.train_cnn --config micro_expression/config_cnn_stage1.yaml

# 微表情 - 特征提取
python -m micro_expression.extract_features --config micro_expression/config_feature_extraction.yaml

# 微表情 - LSTM 第二阶段
python -m micro_expression.train_lstm --config micro_expression/config_lstm_stage2.yaml

# 微表情 - 超参数搜索
python -m micro_expression.hpo_train --config micro_expression/config_lstm_stage2.yaml

# 宏表情训练
python -m macro_expression.multi_dataset_train --dataset fer2013

# 安装依赖
pip install -r hidden_emotion_detection/requirements.txt
pip install -r micro_expression/requirements.txt
pip install -r macro_expression/requirements.txt
```

无测试框架，无 CI/CD。

---

## 架构

### 实时检测系统数据流

```
摄像头 → FaceDetectionEngine → PoseEstimator
                ↓
     ┌──────────┼──────────┐
     ↓          ↓          ↓
  Macro      Micro       AU
  Engine     Engine     Engine
     ↓          ↓          ↓
     └──────────┼──────────┘
                ↓
      EmotionIntegrator
                ↓
    HiddenEmotionEngine（冲突检测）
                ↓
        6-Panel PyQt5 UI
```

### 分层

| 层 | 目录 | 职责 |
|----|------|------|
| 基础设施 | `core/` | EventBus（单例异步发布/订阅）、数据结构、管道抽象 |
| 引擎 | `engines/` | 人脸检测、姿态估计、宏/微表情、AU检测、隐藏情绪、聚合 |
| 界面 | `ui/` | LayoutManager 6列布局，各面板继承 BasePanel |
| 配置 | `config/` | ConfigManager 集中管理，`config.json` |

核心模式：事件驱动（EventBus）、多线程（线程池并行引擎）、单例配置。

### 微表情两阶段

1. `train_cnn.py` → CNN 特征提取器 → `.pth`
2. `extract_features.py` → 提取序列特征
3. `train_lstm.py` → LSTM 时序模型（StratifiedGroupKFold 交叉验证）

---

## 核心规则

### 规则1：代码约定
- 注释和文档使用中文，代码标识符英文
- 各子项目使用相对导入，必须 `-m` 方式运行
- 模型权重（`.pth`/`.pt`）、训练输出不纳入 git

### 规则2：模型路径
配置在 `hidden_emotion_detection/config/config.json`，使用绝对路径指向 `hidden_emotion_detection/models/`。部署新环境时需修改。

### 规则3：文档管理
文档目录结构：`docs/01-快速开始/`、`02-核心架构/`、`03-训练指南/`、`04-开发指南/`
文档编号不重复，按时间顺序。修改代码后同步更新相关文档。

### 规则4：Git 仓库

| 本地目录 | GitHub 仓库 |
|---------|------------|
| 当前项目 | `vivy1024/emotion-analysis-toolkit` |

---

## 按需加载参考

| 场景 | 参考文件 |
|------|---------|
| 系统架构 | `docs/02-核心架构/01-系统架构.md` |
| 微表情训练 | `docs/03-训练指南/01-微表情训练.md` |
| 宏表情训练 | `docs/03-训练指南/02-宏表情训练.md` |
| 开发约定 | `docs/04-开发指南/01-开发约定.md` |
| 实时检测配置 | `hidden_emotion_detection/config/config.json` |

---

**维护者**: 薛小川 | **Always respond in Chinese**
