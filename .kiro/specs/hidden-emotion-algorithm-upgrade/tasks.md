# Tasks: 隐藏情绪检测算法升级（Phase 1）

## Batch 1: 基础设施 — 数据结构 + D-S 证据理论 [REQ-1, REQ-2]

### Task 1: 扩展数据类型 + 七元掩饰矩阵
- [x] data_types.h: 新增 `IntentionType` 枚举（Congruence/Suppression/Masking/Amplification）(2026-02-23)
- [x] data_types.h: 扩展 `HiddenEmotionResult` 新增字段（masking_coefficient, ds_belief, ds_plausibility, ds_conflict_k, intention, intention_confidence, intention_description, stability_score, is_anomalous_transition, adaptive_threshold）(2026-02-23)
- [x] hidden_emotion_engine.h: 替换 `ConflictMap kEmotionConflicts` 为 `constexpr std::array<std::array<float,8>,8> kMaskingMatrix`（8×8含Confusion）(2026-02-23)
- [x] hidden_emotion_engine.cpp: 删除旧的 kEmotionConflicts 定义，实现 kMaskingMatrix 查表函数 `getMaskingCoefficient(EmotionType macro, EmotionType micro)` (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 2: D-S 证据理论模块
- [x] 新建 `include/engines/ds_evidence.h` — BPA 类型定义 + DSEvidence 类声明 (2026-02-23)
- [x] 新建 `src/engines/ds_evidence.cpp` — 实现 Dempster 合成规则 `combine()` (2026-02-23)
- [x] 实现 Yager 修正规则 `combineYager()`（K > 0.9 时将冲突质量分配给全集 Θ）(2026-02-23)
- [x] 实现 `combineSmart()` 自动选择合成策略 (2026-02-23)
- [x] 实现 `emotionToBPA()` — EmotionResult 概率分布 → BPA 转换 (2026-02-23)
- [x] 实现 `auToBPA()` — AU 验证结果 → BPA 转换（基于 AU 匹配度生成质量分配）(2026-02-23)
- [x] CMakeLists.txt: 添加 ds_evidence.cpp 到源文件列表 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 3: AU 分层验证机制
- [x] hidden_emotion_engine.h: 新增 `AuHierarchy` 结构体（core_aus + auxiliary_aus）(2026-02-23)
- [x] hidden_emotion_engine.h: 新增 `kAuHierarchy` 静态常量表（7种情绪的核心/辅助AU映射，含Confusion）(2026-02-23)
- [x] hidden_emotion_engine.cpp: 实现 `validateAuHierarchy()` — 核心AU否决门 + 辅助AU加权 (2026-02-23)
- [x] hidden_emotion_engine.cpp: 实现 AU 强度 Z-score 标准化（维护 per-face 的 AU 基线均值/标准差，前30帧）(2026-02-23)
- [x] 编译验证通过 (2026-02-23)

## Batch 2: 核心算法集成 [REQ-2, REQ-4, REQ-6]

### Task 4: 重构 analyze() 主函数 — 集成 D-S + 掩饰矩阵 + AU验证
- [x] 重写 `analyze()` 函数流程：(2026-02-23)
  1. 检查缓存有效性（保留现有逻辑）
  2. 查询七元掩饰矩阵获取 conflict_coeff
  3. AU 分层验证（核心AU否决门）
  4. 构建三个 BPA：emotionToBPA(macro), emotionToBPA(micro), auToBPA(au)
  5. D-S 证据合成：PM⊕Pm → intermediate⊕PAU → final
  6. 自适应阈值计算
  7. 基于 Bel(hidden_emotion) > τ 做最终判定
- [x] 删除旧的 `calculateEmotionDiffScore()` 和 `calculateAuEvidence()`（被 D-S 替代）(2026-02-23)
- [x] 更新 `HiddenEmotionResult` 填充逻辑（新增字段赋值）(2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 5: 自适应阈值
- [x] hidden_emotion_engine.h: 新增 `calculateAdaptiveThreshold()` 方法 (2026-02-23)
- [x] hidden_emotion_engine.cpp: 实现 τ = α + β·(entropy(PM) + variance(Pm)) (2026-02-23)
- [x] 实现 `entropy()` 辅助函数（信息熵计算）(2026-02-23)
- [x] 实现 `variance()` 辅助函数（概率分布方差）(2026-02-23)
- [x] 在 analyze() 中用自适应阈值替代固定 detection_threshold_ (2026-02-23)
- [x] config.json: 新增 `hidden.adaptive_alpha`(0.5) 和 `hidden.adaptive_beta`(0.15) 配置项 (2026-02-23)

### Task 6: 意图推理引擎
- [x] 新建 `include/engines/intention_engine.h` — IntentionEngine 类声明 (2026-02-23)
- [x] 新建 `src/engines/intention_engine.cpp` — 实现 4 种意图模式判定逻辑 (2026-02-23)
  - Suppression: micro.probability < threshold && macro.type == Neutral
  - Masking: masking_coeff > 0.7 && macro.type != micro.type
  - Amplification: macro.type == micro.type && macro.probability > micro.probability × 1.5
  - Congruence: macro.type == micro.type && |macro.prob - micro.prob| < 0.2
- [x] 每种模式生成中文描述字符串（如"检测到情绪掩饰：表面快乐，实际悲伤"）(2026-02-23)
- [x] CMakeLists.txt: 添加 intention_engine.cpp (2026-02-23)
- [x] 在 hidden_emotion_engine.cpp 的 analyze() 末尾调用意图推理 (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

## Batch 3: 时序上下文 + 集成测试 [REQ-5]

### Task 7: 时序上下文管理器
- [x] 新建 `include/engines/temporal_context.h` — TemporalContext 类声明 (2026-02-23)
- [x] 新建 `src/engines/temporal_context.cpp` — 实现：(2026-02-23)
  - `push()`: 滑动窗口入队，更新转移计数矩阵
  - `dominantEmotion()`: 窗口内出现频率最高的情绪
  - `stabilityScore()`: 主导情绪占比
  - `isAnomalousTransition()`: 转移概率 < anomaly_threshold
  - `transitionProbability()`: 基于转移计数矩阵计算
- [x] CMakeLists.txt: 添加 temporal_context.cpp (2026-02-23)
- [x] hidden_emotion_engine.h: 新增 `TemporalContext` 成员（per-face，用 unordered_map<int, unique_ptr<TemporalContext>>）(2026-02-23)
- [x] analyze() 中集成时序上下文：push 当前情绪 → 检查异常转移 → 填充 stability_score (2026-02-23)
- [x] 编译验证通过 (2026-02-23)

### Task 8: 日志增强 + 实时调试
- [x] analyze() 中添加详细 spdlog::debug 输出：(2026-02-23)
  - 掩饰矩阵系数
  - AU 分层验证结果（核心AU通过/否决、辅助AU匹配数）
  - D-S 合成中间结果（K值、Bel、Pl）
  - 自适应阈值 τ 值
  - 意图判定结果
  - 时序稳定性评分
- [x] 添加 spdlog::info 级别的关键事件日志（隐藏情绪检测到、意图模式变化）(2026-02-23)

### Task 9: 编译 + 运行验证
- [x] 完整编译：`cmake --build . --config Release` (2026-02-23)
- [ ] 运行程序，摄像头实时测试
- [ ] 验证日志输出包含 D-S 合成结果、意图判定、时序信息
- [ ] 验证无崩溃，帧率保持 ≥ 10fps
- [ ] 确认 HiddenEmotionResult 新字段正确填充（通过日志验证）

## Batch 4: 面板更新 + 文档 [依赖 Phase 2 UI 重构]

### Task 10: 更新 HiddenEmotionPanel 显示
- [x] emotion_panel.h/cpp: HiddenEmotionPanel 新增显示：(2026-02-23)
  - 意图模式（抑制/掩饰/放大/一致）+ 置信度
  - D-S 信度/似然值/冲突因子
  - 时序稳定性评分 + 异常转移标记
  - 自适应阈值当前值 + 掩饰系数
- [x] 编译验证通过 (2026-02-23)

### Task 11: 配置文件 + 文档更新
- [x] config.json: 整理所有新增配置项（adaptive_alpha, adaptive_beta, suppression_threshold, masking_threshold, amplification_ratio, congruence_diff, anomaly_threshold, temporal_window_size）(2026-02-23)
- [x] README.md: 全面更新 — 技术栈版本、CMakePresets构建方式、项目结构（含★新增文件）、架构图（含v2管线）、算法说明（掩饰矩阵/D-S/AU验证/意图/时序/自适应阈值）、配置参数表 (2026-02-23)
- [x] design.md: 同步更新掩饰矩阵为8×8（含Confusion）、文件结构标记✅已实现 (2026-02-23)
