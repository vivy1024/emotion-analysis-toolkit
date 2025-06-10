# enhance_hidden.config 模块

## 模块概述

`config` 模块负责管理 `enhance_hidden` 应用程序的所有配置信息。它提供了一种结构化的方式来加载、访问和验证应用程序设置，包括模型路径、UI 参数、处理引擎配置等。

## 主要组件

*   `config_manager.py`: 核心配置管理器，负责加载、合并和提供对配置数据的访问。支持从 JSON 文件加载配置。
*   `models.py`: 定义了与模型配置相关的 Pydantic 模型，用于数据验证和类型提示。
*   `ui_settings.py`: 定义了与 UI 界面配置相关的 Pydantic 模型。
*   `config.json`: 默认的配置文件（或模板），包含应用程序的各项设置。
*   `__init__.py`: 初始化模块，可能包含一些配置加载的快捷方式或全局配置实例。

## 功能

*   从 JSON 文件加载配置。
*   使用 Pydantic 模型验证配置结构和数据类型。
*   提供中心化的配置访问点。
*   支持不同配置源的合并（例如，默认配置和用户指定配置）。

## 使用示例

```python
from enhance_hidden.config import ConfigManager

# 加载配置 (通常在应用启动时完成)
config_manager = ConfigManager()
config_manager.load_config('path/to/your/config.json') # 可选，否则加载默认

# 访问配置
app_config = config_manager.get_app_config()
model_config = config_manager.get_model_config('face_detection')

print(f"窗口宽度: {app_config.window_width}")
print(f"人脸检测模型路径: {model_config.model_path}")

```

## 设计原则

*   **中心化管理**: 所有配置集中管理，方便查找和修改。
*   **结构化与验证**: 使用 Pydantic 模型确保配置的正确性和可预测性。
*   **灵活性**: 支持从外部文件加载配置，方便用户自定义。
*   **可扩展性**: 易于添加新的配置项和配置模型。 