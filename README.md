# 增强版隐藏表情检测系统

## 系统简介

本系统旨在检测和分析人脸表情中的微妙变化和潜在的隐藏情绪。系统集成了多种表情分析引擎，包括:

- 宏观表情分析 (基于FER2013数据集)
- 微表情分析 (基于CAS(ME)²数据集)
- 动作单元(AU)分析 (基于OpenFace预训练模型)
- 姿态估计 (基于OpenFace技术)
- 人脸检测 (基于DLIB模型)
- 隐藏情绪分析 (基于学术研究论文)

## 系统架构

系统由以下主要模块组成:

1. **核心模块 (core)**: 包含基础数据结构、事件总线和处理管道
2. **配置模块 (config)**: 管理系统配置、模型路径和UI设置
3. **引擎模块 (engines)**: 提供各种分析算法实现
4. **用户界面 (ui)**: 提供可视化界面展示分析结果

## 运行环境要求

- Python 3.7+
- OpenCV
- NumPy
- Dlib
- PyQt5/PySide2 (UI界面)
- 其他依赖见requirements.txt

## 快速开始

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 运行系统:
   ```bash
   python run.py
   ```
   
   或者:
   ```bash
   python -m enhance_hidden.main
   ```

## 命令行参数

系统支持以下命令行参数:

- `--debug`: 启用调试模式，输出详细日志信息
- `--config <文件路径>`: 指定自定义配置文件路径
- `--video <源路径>`: 指定视频源，可以是摄像头索引或视频文件路径

示例:
```bash
python run.py --debug --video test_video.mp4
```

## 系统配置

默认配置文件位于 `enhance_hidden/config/config.json`，可通过命令行参数 `--config` 指定自定义配置。

## 模型文件

系统使用的预训练模型路径在 `enhance_hidden/config/models.py` 中配置，请确保这些模型文件存在于指定位置

## 主要改进

### 1. 增强了微表情检测模块
- 重构了`preprocess_face`

# OpenFace Python封装库

这个项目提供了OpenFace的Python封装，使您能够直接在Python代码中使用OpenFace的面部动作单元(AU)分析功能，而不需要通过命令行调用。

## 优势

- 无需临时文件或进程间通信
- 处理速度更快
- 易于集成到现有Python项目中
- 直接在内存中处理图像数据

## 文件说明

- `openface_wrapper.cpp`: C++封装代码，将OpenFace功能暴露为C接口
- `CMakeLists_wrapper.txt`: 用于编译共享库的CMake文件
- `openface_python.py`: Python接口，使用ctypes加载共享库
- `test_openface_wrapper.py`: 测试脚本，展示如何使用此封装库

## 编译指南

### 先决条件

确保已安装以下依赖项：

- OpenFace 2.2.0
- OpenCV 4.x
- CMake 3.x
- C++编译器 (MSVC on Windows, GCC/Clang on Linux/macOS)
- Python 3.6+，含NumPy和OpenCV-Python

### Windows编译步骤

1. 确保OpenFace已正确编译且包含所有必要的模型
2. 在Visual Studio命令提示符中执行：

```batch
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=D:/pycharm2/PythonProject2/OpenFace-OpenFace_2.2.0 ..
cmake --build . --config Release
```

3. 将生成的DLL文件复制到Python代码所在的目录

### Linux/macOS编译步骤

1. 确保OpenFace已正确编译且包含所有必要的模型
2. 执行以下命令：

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/OpenFace-OpenFace_2.2.0 ..
make -j4
```

3. 将生成的.so文件复制到Python代码所在的目录

## 使用方法

### 基本用法

```python
import cv2
from openface_python import OpenFaceWrapper

# 初始化OpenFace
openface = OpenFaceWrapper(
    lib_path="path/to/openface_wrapper.dll",  # 可选，默认在当前目录查找
    model_path="path/to/model",               # 可选，默认在预设位置查找
    au_model_path="path/to/AU_predictors"     # 可选，默认在预设位置查找
)

# 处理单张图像
image = cv2.imread("face.jpg")
success = openface.process_image(image)

if success:
    # 获取AU结果
    au_intensities, au_present = openface.get_action_units()
    
    # 打印结果
    for au_name in sorted(au_intensities.keys()):
        intensity = au_intensities[au_name]
        present = au_present[au_name]
        print(f"{au_name}: {intensity:.3f} {'√' if present else '×'}")
```

### 在enhance_hidden项目中集成

1. 将`openface_wrapper.dll`、`openface_python.py`复制到您的项目目录
2. 在`au_engine.py`中添加以下代码：

```python
def _initialize_direct_au_predictor(self):
    """初始化直接AU预测器"""
    try:
        from openface_python import OpenFaceWrapper
        
        # 初始化OpenFace封装
        self.openface_wrapper = OpenFaceWrapper(
            model_path=os.path.join(self.openface_dir, "model"),
            au_model_path=os.path.join(self.openface_dir, "AU_predictors")
        )
        logger.info("OpenFace AU预测器初始化成功")
        self.use_direct_model = True
    except Exception as e:
        logger.error(f"OpenFace AU预测器初始化失败: {e}")
        self.use_direct_model = False

def _analyze_au_sequence_direct(self, frames, frame_ids, face_ids):
    """使用直接AU预测进行分析"""
    if not hasattr(self, 'use_direct_model') or not self.use_direct_model:
        return self._run_openface_and_parse_sequence(frames)
    
    try:
        # 使用序列中最后一帧进行预测
        frame = frames[-1]
        
        # 处理图像
        success = self.openface_wrapper.process_image(frame)
        
        if success:
            # 获取AU结果
            au_intensities, au_present = self.openface_wrapper.get_action_units()
            
            # 转换为AUResult对象
            result = AUResult(
                au_intensities=au_intensities,
                au_present=au_present,
                au_intensities_raw=au_intensities.copy()
            )
            
            # 添加序列标记
            setattr(result, 'from_sequence', True)
            setattr(result, 'sequence_length', len(frames))
            
            return result
        else:
            logger.warning("AU预测失败，可能没有检测到人脸")
            return None
    except Exception as e:
        logger.error(f"直接AU预测出错: {e}")
        # 失败时回退到原有方法
        return self._run_openface_and_parse_sequence(frames)
```

3. 在原有方法中调用此新方法：

```python
def _analyze_au_sequence(self, frames, frame_ids, face_ids):
    """序列AU分析"""
    try:
        # 优先使用直接方法
        if hasattr(self, 'use_direct_model') and self.use_direct_model:
            return self._analyze_au_sequence_direct(frames, frame_ids, face_ids)
        
        # 否则使用原有方法
        latest_frame_id = frame_ids[-1] if frame_ids else 0
        latest_face_id = face_ids[-1] if face_ids else 0
        
        au_result = self._run_openface_and_parse_sequence(frames)
        # ... 原有代码 ...
        
    except Exception as e:
        logger.error(f"序列AU分析线程错误 (OpenFace): {e}")
        # ... 原有代码 ...
```

## 故障排除

### 常见问题

1. **找不到共享库**: 确保`openface_wrapper.dll`或`.so`文件在Python能够找到的位置
2. **找不到OpenFace模型**: 检查模型路径是否正确
3. **内存访问错误**: 这可能是因为Python和C++之间数据传递的问题，检查图像格式和数组连续性

### 调试技巧

- 检查共享库是否成功加载：`print(ctypes.CDLL("openface_wrapper.dll"))`
- 启用详细日志记录：修改代码中的print语句为更详细的日志输出
- 使用调试器跟踪C++代码的执行
- 检查图像格式：OpenCV图像通常是BGR格式，确保正确传递

## 贡献与维护

如果您发现任何问题或有改进建议，请提交问题报告或拉取请求。