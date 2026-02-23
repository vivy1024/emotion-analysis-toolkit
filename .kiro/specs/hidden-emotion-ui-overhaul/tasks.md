# Tasks: UI 界面全面升级（Phase 2）

## Batch 1: 基础设施 — 主题 + 图表基类 [REQ-3]

### Task 1: 统一主题系统
- [x] 新建 `include/ui/theme.h` — 配色常量（背景/强调/情绪/文字）、字体常量、尺寸常量 (2026-02-23)
- [x] 新建 `resources/styles/theme.qss` — Qt 全局样式表（QDockWidget 标题栏、QToolBar、QMenuBar、QStatusBar） (2026-02-23)
- [x] main.cpp: 加载 theme.qss 替代内联 setStyleSheet (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 2: 自绘图表基类
- [x] 新建 `include/ui/chart_widgets.h` — 声明 BarChart, RadarChart, LineChart (2026-02-23)
- [x] 新建 `src/ui/chart_widgets.cpp`: (2026-02-23)
  - BarChart: QPainter 绘制柱状图 + QTimer 动画过渡
  - RadarChart: QPainter 绘制蛛网图（5层同心多边形 + 数据点连线 + 半透明填充）
  - LineChart: QPainter 绘制折线图（滑动窗口 + 多曲线 + 图例 + 十字线）
- [x] CMakeLists.txt: 添加 chart_widgets.cpp (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

## Batch 2: QDockWidget 面板系统 [REQ-1]

### Task 3: MainWindow 重构为 QDockWidget 架构
- [x] main_window.h: 将所有面板指针改为 QDockWidget* 包裹 (2026-02-23)
- [x] main_window.cpp setupUi(): (2026-02-23)
  - 视频面板设为 centralWidget
  - 其余面板各自包裹在 QDockWidget 中
  - 设置默认停靠位置（右侧：人脸/宏/微/AU/隐藏/意图，底部：时序/灵敏度）
- [x] 菜单栏"视图"菜单: 为每个 QDockWidget 添加 toggleViewAction() (2026-02-23)
- [x] 实现 saveLayout/restoreLayout（QSettings 持久化） (2026-02-23)
- [x] 预设布局：默认/精简/分析（菜单栏"视图→布局"子菜单） (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 4: 工具栏 + 状态栏
- [x] QToolBar: 添加按钮（开始/暂停/停止/截图） (2026-02-23)
- [x] 工具栏图标：使用 Unicode 符号（▶⏸⏹📷） (2026-02-23)
- [x] QStatusBar: 显示 FPS + 检测人脸数 + 当前意图模式 + 耗时详情 (2026-02-23)
- [x] 快捷键绑定：Space（开始/暂停）、S（截图） (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

## Batch 3: 面板内容升级 [REQ-2]

### Task 5: 情绪面板升级（宏表情 + 微表情）
- [x] emotion_panel.h/cpp: 替换 QProgressBar 为 BarChart 自绘柱状图 (2026-02-23)
- [x] 每个柱子使用情绪对应配色（theme.h 中定义） (2026-02-23)
- [x] 顶部显示当前情绪类型 + 概率（大字体，情绪配色） (2026-02-23)
- [x] 动画过渡：新数据到达时柱子高度平滑变化（QTimer 插值） (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 6: AU 雷达图面板
- [x] 新建 `include/ui/au_radar_panel.h` + `src/ui/au_radar_panel.cpp` (2026-02-23)
- [x] 使用 RadarChart 显示 AU 激活强度 (2026-02-23)
- [x] 核心AU 用红色标记，辅助AU 用蓝色标记 (2026-02-23)
- [x] 显示 AU 编号标签（17个AU） (2026-02-23)
- [x] CMakeLists.txt: 添加源文件 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 7: 情绪时序折线图面板
- [x] 新建 `include/ui/timeline_panel.h` + `src/ui/timeline_panel.cpp` (2026-02-23)
- [x] 使用 LineChart 显示最近60秒的情绪概率变化 (2026-02-23)
- [x] 8条曲线对应8种情绪（含Confusion），各自配色 (2026-02-23)
- [x] 支持图例显示 + 十字线 (2026-02-23)
- [x] CMakeLists.txt: 添加源文件 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 8: 意图分析面板 + 隐藏情绪面板升级
- [x] 新建 `include/ui/intention_panel.h` + `src/ui/intention_panel.cpp` (2026-02-23)
  - 显示当前意图模式（大图标 + 文字）
  - 意图置信度百分比
  - 意图历史时间线（QPainter 色块标记，120帧滑动窗口）
- [x] 重构 HiddenEmotionPanel: 使用 theme.h 统一配色 (2026-02-23)
- [x] CMakeLists.txt: 添加源文件 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

## Batch 4: 交互功能 [REQ-4, REQ-5, REQ-6]

### Task 9: 灵敏度控制面板
- [x] 新建 `include/ui/sensitivity_panel.h` + `src/ui/sensitivity_panel.cpp` (2026-02-23)
- [x] QSlider 控件：检测阈值 / 冲突阈值 / 时序窗口大小 (2026-02-23)
- [x] 实时生效：滑块值变化时通过信号通知 (2026-02-23)
- [x] 显示当前自适应阈值 τ 的实时计算值 (2026-02-23)
- [x] CMakeLists.txt: 添加源文件 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 10: 视频面板交互增强
- [x] 截图功能：QFileDialog 选择路径 + cv::imwrite (2026-02-23)
- [x] 截图快捷键 S (2026-02-23)
- [ ] 右键菜单：切换摄像头（枚举可用设备）
- [ ] 右键菜单：调整分辨率（640×480 / 1280×720 / 1920×1080）

### Task 11: 录制与回放（P2，可延后）
- [ ] 定义 .hed 文件格式（二进制头 + 帧序列：timestamp + jpeg_data + analysis_json）
- [ ] 录制：按帧写入 .hed 文件
- [ ] 回放：读取 .hed 文件，按原始帧率播放
- [ ] CSV 导出：时间戳, 宏表情, 微表情, 意图, 各项置信度

### Task 12: 最终集成 + 编译验证
- [x] 完整编译：`cmake --build . --config Release` (2026-02-23)
- [ ] 运行程序，验证所有面板正常显示
- [ ] 拖拽面板测试：停靠/浮动/隐藏/恢复
- [ ] 布局保存/恢复测试
- [ ] 帧率验证：≥ 10fps（图表刷新不拖慢主循环）
