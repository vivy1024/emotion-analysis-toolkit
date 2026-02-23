# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**版本**: v2.3.0 | **更新**: 2026-02-23 | **语言**: 始终使用中文回复

**项目**: 情绪分析工具集（Emotion Analysis Toolkit）| **开发者**: 薛小川

---

## 项目概览

实时视频隐藏情绪检测系统 + 微表情/宏表情模型训练工具集 + C++ 高性能重写版。

**Python 技术栈**: PyTorch + OpenCV + PyQt5 + dlib + MediaPipe + scikit-learn | Python 3.7+
**C++ 技术栈**: C++20 + Qt6 + OpenCV 4.12 + ONNX Runtime GPU + dlib + spdlog | CMake + MSVC

| 子项目 | 目录 | 说明 | 语言 |
|--------|------|------|------|
| 实时检测系统（Python） | `hidden_emotion_detection/` | PyQt5 GUI，6面板布局，多引擎并行 | Python |
| 实时检测系统（C++） | `hidden_emotion_cpp/` | Qt6 QDockWidget 面板 + ONNX GPU 加速，独立 git 仓库 | C++ |
| 微表情训练 | `micro_expression/` | CNN+LSTM 两阶段训练 | Python |
| 宏表情训练 | `macro_expression/` | 支持 FER2013 等公开数据集 | Python |
| 评估工具 | `hidden_emotion_detection/evaluation/` | CASME2 基线训练 + STRS 指标 + ROI 特征提取 | Python |

### 联系方式

| 开发者 | 邮箱 | GitHub |
|--------|------|--------|
| 薛小川 | 1336495069@qq.com | vivy1024 |

---

## 常用命令

```bash
# === Python 实时检测 ===
# 启动实时检测 GUI
python -m hidden_emotion_detection.main
# 可选: --debug, --log-level DEBUG/INFO/WARNING

# === 微表情训练 ===
# CNN 第一阶段
python -m micro_expression.train_cnn --config micro_expression/config_cnn_stage1.yaml

# 特征提取
python -m micro_expression.extract_features --config micro_expression/config_feature_extraction.yaml

# LSTM 第二阶段
python -m micro_expression.train_lstm --config micro_expression/config_lstm_stage2.yaml

# 超参数搜索
python -m micro_expression.hpo_train --config micro_expression/config_lstm_stage2.yaml

# === 宏表情训练 ===
python -m macro_expression.multi_dataset_train --dataset fer2013

# === 评估工具 ===
python -m hidden_emotion_detection.evaluation.train_baselines
python -m hidden_emotion_detection.evaluation.evaluate_casme2

# === C++ 版构建（独立 git 仓库） ===
cd hidden_emotion_cpp
cmake --preset default
cmake --build . --config Release
Release/HiddenEmotionDetector.exe

# === 安装依赖 ===
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
- 模型权重（`.pth`/`.pt`/`.pkl`/`.dat`/`.tflite`）、训练输出不纳入 git

### 规则2：模型路径
配置在 `hidden_emotion_detection/config/config.json`，使用绝对路径指向 `hidden_emotion_detection/models/`。部署新环境时需修改。

### 规则3：文档管理
文档目录结构：`docs/01-快速开始/`、`02-核心架构/`、`03-训练指南/`、`04-开发指南/`
文档编号不重复，按时间顺序。修改代码后同步更新相关文档。

### 规则4：Git 仓库

| 本地目录 | GitHub 仓库 | 说明 |
|---------|------------|------|
| 当前项目（根目录） | `vivy1024/emotion-analysis-toolkit` | Python 工具集 |
| `hidden_emotion_cpp/` | `vivy1024/hidden-emotion-cpp` | C++ 版，独立 git 仓库（已在 .gitignore 中排除） |

⚠️ `hidden_emotion_cpp/` 是独立 git 仓库，不要在主仓库中 `git add` 它

### 规则5：复杂任务必须使用 Agent Teams

触发条件：跨项目修改 | ≥4小时工作量 | 架构迁移 | 全量测试+部署

```
1. 读取 .claude/team-prompt.md
2. TeamCreate 创建正式团队（必须！）
3. Task spawn agent，指定 team_name
4. 汇总 → 测试 → 文档 → Git提交
5. TeamDelete 清理
```

❌ 禁止直接 Task spawn 不先 TeamCreate（监控工具无法追踪、agent间无法通信）
❌ 禁止团队完成后不 TeamDelete 清理

### 规则6：使用 aivectormemory 持久化记忆

每次对话必须使用 `aivectormemory` MCP 工具：
- **对话开始**：`recall` 查询相关记忆，恢复上下文
- **重要决策/发现**：`remember` 及时存储（不要等对话结束）
- **对话结束前**：`auto_save` 保存本次决策、修改、踩坑、待办
- **用户偏好**：存入 `user` 作用域（跨项目生效）
- **项目状态**：存入 `project` 作用域

❌ 禁止只在对话结束时才存记忆（中途崩溃/interrupted 会丢失）
❌ 禁止忽略 recall 结果（上次对话的记忆是重要上下文）

### 规则7：等待用户时必须暂停输出

当需要等待用户操作（环境变量更新等）时：
- ✅ 说明等待什么，然后**停止输出**，不要持续发送消息
- ✅ 收到旧的 `task-notification` 时，**完全忽略，不回复任何内容**
- ✅ 如果有多个旧通知堆积，一条都不要回复
- ❌ 禁止对每个旧后台任务通知都回复"旧通知，忽略"等消息
- ❌ 禁止在等待期间循环发送消息（会导致上下文膨胀、用户无法暂停、compact失败）

原因：VSCode Claude Code插件中，持续输出会阻塞用户交互，暂停按钮可能需要多次点击才能生效，
上下文过长会导致compact失败，对话彻底终结。

---

### 规则8：Windows Bash后台命令防风暴

Windows环境下bash命令可能全部以后台模式运行，导致无法直接获取输出。

**正确做法**：
```bash
# 1. 命令输出重定向到固定文件（复用同一个文件名，不要每次创建新文件）
docker exec fitness_daml_rag python -m pytest tests/ > /f/build_body/_output.txt 2>&1
# 2. 用Read工具读取文件（不用bash cat）
# 3. 读取完毕后及时删除临时文件，避免根目录污染
```

**禁止**：
- ❌ 使用 `sleep N && cat file` 链式命令等待输出（会形成后台任务雪崩）
- ❌ 反复spawn新bash命令读取上一个命令的输出
- ❌ Agent内部超过3次bash重试后仍继续（应改用Read工具或报告失败）
- ❌ 每个命令创建不同的临时文件名（如`_grep1.txt`、`_grep2.txt`），应复用`_output.txt`

---

## 🚫 严格禁止

- ❌ 修改代码后不更新文档/CHANGELOG/不 Git 提交
- ❌ 使用 curl 命令（Windows 兼容问题，用 Chrome DevTools）

---

## Git 工作流

### 提交格式

`type(scope): description`

常用类型：feat(新功能)、fix(修复)、docs(文档)、refactor(重构)、perf(性能)

示例：
```bash
git commit -m "feat(micro): 添加LOSO评估协议"
git commit -m "fix(au): 修复SVM模型加载路径"
git commit -m "docs(changelog): 记录架构重构变更"
```

### 分支命名（可选）

个人开发通常在 main 分支，需要时创建：
```bash
git checkout -b feature/loso-evaluation  # 大功能
git checkout -b fix/config-path          # Bug修复
```

---

## 按需加载参考

| 场景 | 参考文件 |
|------|---------|
| 系统架构 | `docs/02-核心架构/01-系统架构.md` |
| 微表情训练 | `docs/03-训练指南/01-微表情训练.md` |
| 宏表情训练 | `docs/03-训练指南/02-宏表情训练.md` |
| 微表情领域参考 | `docs/03-训练指南/03-微表情领域参考索引.md` |
| CASME2 实验计划 | `docs/03-训练指南/05-CASME2实验计划.md` |
| 开发约定 | `docs/04-开发指南/01-开发约定.md` |
| 数据集申请 | `docs/04-开发指南/数据集申请/` |
| 技术迭代路线 | `docs/04-开发指南/03-技术迭代路线.md` |
| 实时检测配置 | `hidden_emotion_detection/config/config.json` |
| C++ 版规则 | `hidden_emotion_cpp/CLAUDE.md` |
| C++ 版升级 spec | `hidden_emotion_cpp/.kiro/specs/` |

---

**维护者**: 薛小川 | v2.3.0 | **Always respond in Chinese**
