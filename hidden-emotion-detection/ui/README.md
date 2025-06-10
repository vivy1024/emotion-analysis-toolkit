# enhance_hidden.ui 模块

## 模块概述

`ui` 模块负责构建和管理 `enhance_hidden` 应用程序的用户界面。它使用图形库（如 OpenCV 的 `highgui` 或其他 GUI 框架如 PyQt/Tkinter - 具体需看实现）来展示视频流、绘制分析结果（如人脸框、情绪标签、AU 强度条）并可能提供用户交互。

## 主要组件

*   `layout_manager.py`: 管理 UI 中各个面板（Panel）的布局和排列方式。
*   `base_panel.py`: 提供 UI 面板的基类，定义通用接口和功能，如更新、绘制等。
*   `video_panel.py`: 显示原始或处理后的视频流。
*   `face_panel.py`: 在视频流上绘制人脸检测框、关键点、姿态等信息。
*   `au_intensity_panel.py`: 以图表（如条形图）形式展示检测到的面部动作单元（AU）强度。
*   `macro_emotion_panel.py`: 显示识别出的宏表情结果，来源于通用模型分析或基于面部动作单元（AU）的分析。
*   `micro_emotion_panel.py`: 显示检测到的微表情信息，来源于通用模型分析或基于面部动作单元（AU）的分析。
*   `hidden_emotion_panel.py`: 展示分析出的隐藏情绪状态。
*   `__init__.py`: 初始化模块，可能导出主要的布局管理器或面板类。

## 功能

*   将处理引擎的分析结果可视化。
*   提供用户友好的界面来观察情绪分析过程。
*   管理窗口布局和各个显示区域。
*   通过 `core.event_bus` 订阅事件（如 `FRAME_PROCESSED`）来获取最新数据并更新显示。

## 工作流程

1.  UI 组件（如 `MacroEmotionPanel`）在初始化时订阅感兴趣的事件（如 `FRAME_PROCESSED`）。
2.  当 `core.pipeline` 处理完一帧并通过 `event_bus` 发布 `FRAME_PROCESSED` 事件时，事件数据（`FrameData`）被传递给订阅者。
3.  对应的 UI 面板接收 `FrameData`，提取所需信息（如宏表情结果）。
4.  面板根据新数据更新其内部状态，并在下一轮绘制循环中将结果渲染到屏幕上。
5.  `LayoutManager` 负责协调各个面板在窗口中的位置和大小。

## 设计原则

*   **关注点分离**: UI 逻辑与核心处理逻辑分离，UI 只负责展示数据。
*   **事件驱动**: 通过事件总线与核心模块解耦，响应数据变化。
*   **模块化面板**: 每个面板负责展示特定类型的信息，易于组合和扩展。
*   **响应式**: 界面应能及时反映底层数据的变化。

## 依赖

*   `enhance_hidden.core`: 依赖核心数据结构 (`data_types`) 和事件总线 (`event_bus`)。
*   图形库: 需要一个图形库来绘制界面（如 OpenCV, PyQt, Tkinter等）。 