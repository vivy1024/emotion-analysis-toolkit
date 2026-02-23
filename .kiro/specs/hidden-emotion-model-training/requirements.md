# Requirements: 模型训练升级（Phase 3）

## Introduction

当前系统使用毕设阶段训练的模型：ResNet-18 宏表情（FER2013, ~65%准确率）、CNN-LSTM 微表情（CASME II, ~45%准确率）、简单 AU 检测器。用户计划使用其兄长的双 NVIDIA RTX 4090 服务器进行专业级模型训练，并已申请 SMIC 数据集（Oulu大学已批准），SAMM 需要教授签名。

**目标**: 训练 SOTA 级别的情绪/微表情/AU 模型，显著提升系统准确率。

**优先级**: P2（依赖硬件资源和数据集获取）

**训练环境**: 双 RTX 4090 (48GB VRAM total) + Linux 服务器

## Requirements

### REQ-1: 宏表情模型升级 [P0]
- 当前: ResNet-18, FER2013, ~65% accuracy
- 目标: POSTER++ 或 DAN (Distract-your-Attention Network)
- 训练数据: AffectNet (440K张, 8类) + RAF-DB (30K张, 7类) + FER2013 (35K张, 7类)
- 目标准确率: AffectNet ≥ 63%, RAF-DB ≥ 90%（SOTA 水平）
- 输出: ONNX 格式，输入 224×224×3，输出 7 维概率向量
- 推理速度: < 5ms/帧 (RTX 3060)

### REQ-2: 微表情模型升级 [P0]
- 当前: CNN(ResNet-18) + LSTM 5-fold ensemble, CASME II only, ~45% UF1
- 目标: VideoMAE V2 + Temporal Adapter（参考 MEGC2024 STR 冠军方案）
- 训练数据: CASME II + SMIC + SAMM + CAS(ME)³ 联合训练
- 评估: LOSO (Leave-One-Subject-Out) 交叉验证
- 目标准确率: 联合数据集 UF1 ≥ 0.75, UAR ≥ 0.70
- 输出: ONNX 格式，输入 16帧×224×224×3 视频片段，输出 3/5/7 维概率向量
- 注意: 微表情类别数因数据集而异（CASME II=5类, SMIC=3类, SAMM=5类）

### REQ-3: AU 检测模型升级 [P1]
- 当前: 简单 CNN AU 检测器，精度有限
- 目标: OpenGraphAU 或 ME-GraphAU（基于图神经网络的 AU 检测）
- 训练数据: BP4D (41人, 328视频) + DISFA (27人, 54视频)
- 目标: 12个 AU 的平均 F1 ≥ 0.65
- 输出: ONNX 格式，输入 256×256×3，输出 AU 激活概率 + 强度
- 需要输出的 AU: AU1,2,4,5,6,7,9,10,12,15,17,20,23,24,25,26,27

### REQ-4: 微表情 Spotting 模型 [P2]
- 当前: 光流阈值法（optical_flow_engine.cpp），误报率高
- 目标: 基于学习的 Spotting 方法（参考 MEGC2024 Spotting 赛道方案）
- 训练数据: CAS(ME)³ (含 onset/apex/offset 标注)
- 评估指标: F1-score (IoU ≥ 0.5)
- 输出: ONNX 格式，输入连续帧序列，输出 onset/apex/offset 时间点

### REQ-5: 训练基础设施 [P0]
- PyTorch 训练框架搭建（统一的 trainer/dataloader/augmentation）
- 混合精度训练 (AMP) 充分利用 4090 的 FP16/BF16 性能
- Weights & Biases 实验追踪
- 模型导出管线: PyTorch → ONNX → ONNX Runtime 验证
- 数据增强: 随机裁剪、水平翻转、颜色抖动、Mixup、CutMix

### REQ-6: C++ 推理端适配 [P1]
- 更新 ONNX Runtime 推理代码适配新模型的输入/输出维度
- 微表情模型从单帧推理改为视频片段推理（16帧缓冲区）
- AU 模型输出格式适配（概率 + 强度双输出头）
- 模型热加载：支持运行时切换模型文件（无需重启程序）

## Out of Scope
- 自建数据集标注（使用公开数据集）
- 模型压缩/量化（先追求精度，后续优化推理速度）
- 边缘设备部署（仅桌面 GPU）

## Dependencies
- 双 RTX 4090 服务器可用
- SMIC 数据集已获取（Oulu 大学已批准）
- SAMM 数据集获取（需教授签名，进行中）
- CAS(ME)³ 数据集获取（需申请）
- AffectNet 数据集获取（需申请）
- Phase 1 算法升级完成（AU 分层验证需要新 AU 模型的精确输出）
