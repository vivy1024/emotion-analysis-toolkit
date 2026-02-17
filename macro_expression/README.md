# 宏表情训练代码（fer2013model）

## 项目简介
本项目用于宏观表情识别模型的训练与评估，支持FER2013等公开数据集，适用于情感计算、表情识别等研究与应用。

## 主要模块说明
- `multi_dataset_train.py`：支持多数据集的模型训练主脚本，包含训练流程、数据加载、评估等。
- `emotion_model.py`：模型推理与单张图片表情识别脚本。
- `FacialExpressionRecognition.spec`：PyInstaller打包配置文件，便于生成可执行程序。

## 安装方法
```bash
git clone https://github.com/yourname/fer2013model.git
cd fer2013model
pip install -r requirements.txt
```

## 使用方法
- 训练模型：
  ```bash
  python multi_dataset_train.py --dataset fer2013
  ```
- 推理与评估：
  ```bash
  python emotion_model.py --input your_image.jpg
  ```
- 具体参数说明请参考脚本内注释。

## 数据集说明
- 推荐使用FER2013等公开表情数据集，需自行下载并按要求放置。
- 数据集格式及预处理方式请参考`multi_dataset_train.py`。

## 贡献指南
欢迎提交PR和Issue，建议先在Issue区讨论。

## 许可证
MIT License

## 联系方式
1336495069@qq.com 