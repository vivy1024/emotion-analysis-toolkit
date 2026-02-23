# Tasks: 模型训练升级（Phase 3）

## Batch 1: 训练基础设施 [REQ-5]

### Task 1: 训练框架搭建
- [ ] 创建 `training/` 目录结构（configs/, datasets/, models/, trainers/, export/, augmentations/）
- [ ] 编写 `requirements.txt`（torch, torchvision, timm, onnx, onnxruntime, wandb, albumentations, einops）
- [ ] 实现 `trainers/base_trainer.py` — 通用训练器：
  - AMP 混合精度训练
  - W&B 实验追踪
  - Checkpoint 保存/恢复
  - 学习率调度（CosineAnnealing + Warmup）
  - EarlyStopping
- [ ] 实现 `augmentations/` — 数据增强管线（spatial + color + Mixup/CutMix）
- [ ] 实现 `export/to_onnx.py` — PyTorch → ONNX 导出 + 简化 + 验证

### Task 2: 数据集准备
- [ ] 实现 `datasets/affectnet.py` — AffectNet 数据加载器（8类→7类映射，去除 Contempt）
- [ ] 实现 `datasets/rafdb.py` — RAF-DB 数据加载器
- [ ] 实现 `datasets/casme2.py` — CASME II 数据加载器（视频片段→帧序列）
- [ ] 实现 `datasets/smic.py` — SMIC 数据加载器（3类：positive/negative/surprise）
- [ ] 实现 `datasets/samm.py` — SAMM 数据加载器（待数据集获取后完善）
- [ ] 实现 `datasets/casme3.py` — CAS(ME)³ 数据加载器（含 spotting 标注）
- [ ] 实现 `datasets/bp4d.py` — BP4D AU 数据加载器
- [ ] 实现 `datasets/disfa.py` — DISFA AU 数据加载器
- [ ] 实现 `datasets/composite.py` — 联合数据集（多数据集混合 + 类别映射统一）

## Batch 2: 宏表情模型训练 [REQ-1]

### Task 3: POSTER++ 模型实现
- [ ] 实现 `models/poster_v2.py` — POSTER++ 模型定义（或从官方仓库适配）
- [ ] 实现 `trainers/macro_trainer.py` — 宏表情训练器
- [ ] 编写 `configs/macro_poster.yaml` — 训练超参数配置
- [ ] AffectNet 预训练 → RAF-DB 微调
- [ ] 验证: AffectNet-7 val accuracy ≥ 63%, RAF-DB ≥ 90%

### Task 4: 宏表情模型导出
- [ ] PyTorch → ONNX 导出（输入 [1,3,224,224]，输出 [1,7]）
- [ ] ONNX 图优化（onnx-simplifier）
- [ ] ONNX Runtime 精度验证（与 PyTorch 输出对比 < 1e-5）
- [ ] 替换 C++ 端 macro.onnx，验证推理正常

## Batch 3: 微表情模型训练 [REQ-2]

### Task 5: VideoMAE V2 + Temporal Adapter 实现
- [ ] 实现 `models/videomae_v2.py` — VideoMAE V2 + Temporal Adapter
  - 加载预训练权重（Kinetics-400）
  - 冻结 ViT 主干，仅训练 Temporal Adapter + 分类头
- [ ] 实现 `trainers/micro_trainer.py` — 微表情训练器
  - LOSO 交叉验证循环
  - 联合数据集训练（CASME II + SMIC + SAMM）
  - 类别映射统一（3类/5类→统一类别体系）
- [ ] 编写 `configs/micro_videomae.yaml`

### Task 6: 微表情模型训练 + 导出
- [ ] LOSO 交叉验证训练（每个 subject 轮流做测试集）
- [ ] 验证: 联合数据集 UF1 ≥ 0.75, UAR ≥ 0.70
- [ ] 选择最佳 fold 模型导出 ONNX（输入 [1,16,3,224,224]，输出 [1,N]）
- [ ] 或导出 ensemble 模型（多 fold 平均）

## Batch 4: AU 模型训练 [REQ-3]

### Task 7: OpenGraphAU 实现
- [ ] 实现 `models/opengraphau.py` — OpenGraphAU 模型定义
  - 图神经网络建模 AU 关系
  - 双输出头：AU 激活概率 + AU 强度
- [ ] 实现 `trainers/au_trainer.py` — AU 训练器
  - 多标签分类（BCE Loss）
  - 强度回归（MSE Loss）
  - BP4D + DISFA 联合训练
- [ ] 编写 `configs/au_opengraphau.yaml`

### Task 8: AU 模型训练 + 导出
- [ ] BP4D 3-fold 交叉验证训练
- [ ] 验证: 12个 AU 平均 F1 ≥ 0.65
- [ ] ONNX 导出（输入 [1,3,256,256]，输出 au_prob [1,17] + au_intensity [1,17]）
- [ ] 替换 C++ 端 au.onnx

## Batch 5: C++ 推理端适配 [REQ-6]

### Task 9: 微表情引擎适配视频片段输入
- [ ] micro_emotion_engine.h: 新增 FrameBuffer 类（16帧滑动窗口）
- [ ] micro_emotion_engine.cpp: 修改推理逻辑
  - 每帧 push 到 FrameBuffer
  - FrameBuffer 满时触发推理
  - 输入张量从 [1,3,224,224] 改为 [1,16,3,224,224]
- [ ] 删除旧的 CNN+LSTM 5-fold ensemble 逻辑
- [ ] 编译验证通过

### Task 10: AU 引擎适配双输出头
- [ ] au_engine.h: 修改 AuResult 填充逻辑
  - 从单输出（au_present 二值）改为双输出（概率 + 强度）
  - 概率 > 0.5 → au_present = true
  - 强度直接从模型输出读取
- [ ] au_engine.cpp: 更新 ONNX 推理代码（两个输出节点）
- [ ] 编译验证通过

### Task 11: 模型热加载 + 端到端验证
- [ ] engine_base.h/cpp: 实现 `reloadModel()` — 原子替换 OnnxSession
- [ ] 配置文件支持模型路径配置（不再硬编码）
- [ ] 端到端测试：新模型 vs 旧模型 A/B 对比
- [ ] 性能基准：确认新模型推理延迟可接受（< 10ms/帧 总计）

## Batch 6: Spotting 模型（P2，可延后）[REQ-4]

### Task 12: Spotting 模型训练
- [ ] 实现 `models/spotting_net.py` — 基于学习的 Spotting 网络
- [ ] 实现 `trainers/spotting_trainer.py`
- [ ] CAS(ME)³ 数据集训练
- [ ] 验证: F1-score (IoU ≥ 0.5) ≥ 0.30

### Task 13: Spotting 引擎适配
- [ ] optical_flow_engine.h/cpp: 替换阈值法为学习模型
- [ ] 或新建 spotting_engine.h/cpp 独立引擎
- [ ] 编译 + 端到端验证
