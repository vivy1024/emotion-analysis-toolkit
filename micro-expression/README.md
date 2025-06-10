# 微表情模型训练代码（18_2）

## 项目简介
本项目用于微表情识别的特征提取、模型训练与评估，支持多种深度学习模型结构，适用于科研与实际应用场景。

## 主要模块说明
- `train_cnn.py`：用于训练CNN模型，支持多种网络结构和参数配置。
- `train_lstm.py`：用于训练LSTM模型，适合时序特征建模。
- `extract_features.py`：实现特征提取流程，将原始数据转为模型可用特征。
- `hpo_train.py`：自动化超参数优化训练。
- `feature_dataset.py`、`dataset.py`：数据集加载、预处理与增强。
- `models.py`：定义各类深度学习模型结构。
- `utils.py`、`transforms.py`：工具函数与数据增强方法。

## 安装方法
```bash
git clone https://github.com/yourname/18_2.git
cd 18_2
pip install -r requirements.txt
```

## 使用方法
- 训练CNN模型：
  ```bash
  python train_cnn.py --config config_cnn_stage1.yaml
  ```
- 训练LSTM模型：
  ```bash
  python train_lstm.py --config config_lstm_stage2.yaml
  ```
- 特征提取：
  ```bash
  python extract_features.py --config config_feature_extraction.yaml
  ```
- 超参数优化：
  ```bash
  python hpo_train.py
  ```

## 数据集说明
- 数据集需按指定格式放置，具体格式请参考`feature_dataset.py`和`dataset.py`中的说明。
- 支持自定义特征和标签。

## 贡献指南
欢迎提交PR和Issue，建议先在Issue区讨论。

## 许可证
MIT License

## 联系方式
1336495069@qq.com 