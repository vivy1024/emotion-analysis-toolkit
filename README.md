# emotion-analysis-toolkit
情绪分析工具集：包含微表情、宏表情训练与隐藏情绪检测系统

---

## 快速开始 | Quick Start

### 1. 克隆仓库 | Clone the repository

```bash
git clone https://github.com/vivy1024/emotion-analysis-toolkit.git
cd emotion-analysis-toolkit
```

### 2. 进入子项目 | Enter a subproject

以微表情为例（其它子项目同理）：

```bash
cd micro-expression
# 查看README.md，按说明安装依赖并运行
```

---

## 子项目简介 | Subproject Introduction

### micro-expression

- 微表情识别的特征提取、模型训练与评估
- 支持多种深度学习模型结构
- 详见 [micro-expression/README.md](./micro-expression/README.md)

### macro-expression

- 宏观表情识别模型的训练与评估
- 支持FER2013等公开数据集
- 详见 [macro-expression/README.md](./macro-expression/README.md)

### hidden-emotion-detection

- 实时视频情绪识别系统，集成了人脸检测、头部姿态估计、宏观/微表情与隐藏情绪分析
- 详见 [hidden-emotion-detection/README.md](./hidden-emotion-detection/README.md)

---

## 依赖环境 | Requirements

- Python 3.7+
- 推荐使用虚拟环境（venv、conda等）
- 各子项目有独立 requirements.txt

---

## 贡献指南 | Contribution Guide

欢迎提交PR和Issue，建议先在Issue区讨论。  
See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

---

## 开源协议 | License

MIT License

---

## 联系方式 | Contact

- Email: 1336495069@qq.com

---

## 致谢 | Acknowledgement

感谢所有开源社区项目和数据集的支持！

---
