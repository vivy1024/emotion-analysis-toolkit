# 增强版隐藏情绪检测系统（enhance_hidden）

## 项目简介
本项目为一套基于深度学习的实时视频情绪识别系统，集成了人脸检测、头部姿态估计、宏观表情、微表情与隐藏情绪分析等多功能模块，适用于科研、教育和实际应用场景。

## 主要模块说明
- `main.py`：主程序入口，负责界面调度与系统初始化。
- `app.py`：核心业务逻辑，管理各功能模块的调用与数据流转。
- `ui/`：图形界面相关代码，基于PyQt5实现多面板布局。
- `models/`：深度学习模型定义与加载。
- `engines/`：各类分析引擎（如人脸检测、表情识别、姿态估计等）。
- `core/`：核心算法与数据结构。
- `utils/`：工具函数与通用组件。
- `config/`：配置文件与参数管理。

## 安装方法
```bash
git clone https://github.com/yourname/enhance_hidden.git
cd enhance_hidden
pip install -r requirements.txt
```

## 使用方法
- 启动主程序：
  ```bash
  python main.py
  ```
- 主要参数说明请参考`main.py`和`app.py`内注释。
- 支持摄像头实时检测和视频文件分析。

## 依赖说明
- 依赖详见 requirements.txt，常用依赖包括：
  - numpy, pandas, opencv-python, torch, torchvision, PyQt5, dlib, matplotlib, tqdm, pyyaml 等。

## 界面说明
- 六列式布局，分别展示：
  1. 实时视频流
  2. 人脸与头部姿态
  3. 关键啁激活状态与强度
  4. 宏观表情分析
  5. 微表情分析
  6. 隐藏情绪分析
- 支持多分辨率与多平台。

## 贡献指南
欢迎提交PR和Issue，建议先在Issue区讨论。

## 许可证
MIT License

## 联系方式
1336495069@qq.com