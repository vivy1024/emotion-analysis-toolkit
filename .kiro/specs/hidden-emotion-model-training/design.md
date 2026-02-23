# Design: 模型训练升级（Phase 3）

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    训练服务器 (双 RTX 4090)                    │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 宏表情训练  │  │ 微表情训练  │  │ AU 训练     │          │
│  │ POSTER++    │  │ VideoMAE V2 │  │ OpenGraphAU │          │
│  │ AffectNet   │  │ CASME+SMIC  │  │ BP4D+DISFA  │          │
│  │ +RAF-DB     │  │ +SAMM       │  │             │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                   │
│         ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────┐         │
│  │              ONNX 导出管线                       │         │
│  │  PyTorch → TorchScript → ONNX → Simplify       │         │
│  │  → ONNX Runtime 验证 → 精度对比                 │         │
│  └──────────────────────┬──────────────────────────┘         │
└─────────────────────────┼────────────────────────────────────┘
                          │ .onnx 文件
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    C++ 推理端 (RTX 3060)                      │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ macro.onnx  │  │ micro_v2.   │  │ au_v2.onnx  │          │
│  │ POSTER++    │  │ onnx        │  │ OpenGraphAU │          │
│  │ 224×224     │  │ VideoMAE V2 │  │ 256×256     │          │
│  │ → 7-dim     │  │ 16×224×224  │  │ → AU prob   │          │
│  │             │  │ → 3/5/7-dim │  │ + intensity │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 训练代码结构

```
hidden_emotion_cpp/training/          # 新建训练代码目录
├── configs/
│   ├── macro_poster.yaml             # POSTER++ 训练配置
│   ├── macro_dan.yaml                # DAN 训练配置
│   ├── micro_videomae.yaml           # VideoMAE V2 微表情配置
│   ├── au_opengraphau.yaml           # OpenGraphAU 配置
│   └── spotting.yaml                 # Spotting 模型配置
├── datasets/
│   ├── affectnet.py                  # AffectNet 数据加载器
│   ├── rafdb.py                      # RAF-DB 数据加载器
│   ├── casme2.py                     # CASME II 数据加载器
│   ├── smic.py                       # SMIC 数据加载器
│   ├── samm.py                       # SAMM 数据加载器
│   ├── casme3.py                     # CAS(ME)³ 数据加载器
│   ├── bp4d.py                       # BP4D 数据加载器
│   ├── disfa.py                      # DISFA 数据加载器
│   └── composite.py                  # 联合数据集（多数据集混合）
├── models/
│   ├── poster_v2.py                  # POSTER++ 模型定义
│   ├── dan.py                        # DAN 模型定义
│   ├── videomae_v2.py                # VideoMAE V2 + Temporal Adapter
│   ├── opengraphau.py                # OpenGraphAU 模型定义
│   └── spotting_net.py               # Spotting 网络
├── trainers/
│   ├── base_trainer.py               # 通用训练器（AMP, W&B, checkpoint）
│   ├── macro_trainer.py              # 宏表情训练器
│   ├── micro_trainer.py              # 微表情训练器（LOSO 交叉验证）
│   ├── au_trainer.py                 # AU 训练器
│   └── spotting_trainer.py           # Spotting 训练器
├── export/
│   ├── to_onnx.py                    # PyTorch → ONNX 导出
│   ├── onnx_simplify.py              # ONNX 图优化
│   └── validate_onnx.py             # ONNX Runtime 精度验证
├── augmentations/
│   ├── spatial.py                    # 空间增强（裁剪/翻转/旋转）
│   ├── color.py                      # 颜色增强（亮度/对比度/饱和度）
│   └── advanced.py                   # Mixup, CutMix, RandAugment
├── requirements.txt                  # Python 依赖
└── README.md                         # 训练指南
```

## 模型选型依据

### 宏表情: POSTER++ vs DAN
| 指标 | POSTER++ | DAN |
|------|----------|-----|
| AffectNet-7 | 67.49% | 65.69% |
| RAF-DB | 92.21% | 89.70% |
| 参数量 | 43.6M | 44.3M |
| 推理速度 | ~4ms | ~3ms |
| 特点 | 跨注意力融合 landmark+image | 多头注意力+注意力擦除 |

**选择 POSTER++**: 精度更高，且利用 landmark 信息与我们的 dlib 68点检测互补。

### 微表情: VideoMAE V2 + Temporal Adapter
- MEGC2024 STR 赛道冠军方案
- 预训练: VideoMAE V2 (ViT-Base, Kinetics-400)
- 微调: Temporal Adapter 冻结主干，仅训练适配器层
- 优势: 少样本学习能力强（微表情数据集普遍很小）

### AU: OpenGraphAU
- 基于图神经网络建模 AU 之间的关系
- BP4D 平均 F1: 64.7%（SOTA）
- 输出 AU 概率 + 强度双头

## C++ 推理端适配设计

### 微表情视频片段缓冲区
```cpp
// micro_emotion_engine.h 新增
class FrameBuffer {
    std::deque<cv::Mat> frames_;
    size_t capacity_ = 16;  // VideoMAE V2 需要 16 帧输入
public:
    void push(const cv::Mat& frame);
    bool isFull() const;
    std::vector<cv::Mat> getSequence() const;
    // 转换为 ONNX 输入张量: [1, 16, 3, 224, 224]
    std::vector<float> toTensor() const;
};
```

### AU 模型双输出头
```cpp
// au_engine.h 适配
struct AuModelOutput {
    std::vector<float> probabilities;  // [num_aus] 激活概率
    std::vector<float> intensities;    // [num_aus] 强度 0~5
};
// ONNX 模型有两个输出节点: "au_prob" 和 "au_intensity"
```

### 模型热加载
```cpp
// engine_base.h 新增
class EngineBase {
    // ...
    bool reloadModel(const std::string& new_model_path);
    // 原子替换 OnnxSession，旧 session 延迟释放
};
```

## 训练计划时间线

| 阶段 | 内容 | 预计时间 | GPU 占用 |
|------|------|---------|---------|
| 1 | 数据集准备 + 预处理 | 1周 | 无 |
| 2 | 宏表情 POSTER++ 训练 | 2-3天 | 单卡 |
| 3 | 微表情 VideoMAE V2 训练 (LOSO) | 5-7天 | 双卡 |
| 4 | AU OpenGraphAU 训练 | 2-3天 | 单卡 |
| 5 | ONNX 导出 + C++ 集成 | 2-3天 | 无 |
| 6 | 端到端验证 + 调参 | 3-5天 | 单卡 |

## Testing Strategy

- 训练端: W&B 追踪 loss/accuracy/F1 曲线
- 导出验证: PyTorch vs ONNX Runtime 输出差异 < 1e-5
- C++ 集成: 新旧模型 A/B 对比（同一视频流，对比检测结果）
- 性能基准: 新模型推理延迟 vs 旧模型，确保实时性
