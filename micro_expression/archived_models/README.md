# 归档模型说明

本目录包含开发过程中尝试过但因显存限制（8GB）未能完成训练的模型架构。
现已整理归档，待租赁服务器后重新训练对比。

## 模型清单

| 文件 | 模型 | 说明 | 显存需求 |
|------|------|------|---------|
| `dmca_net.py` | DMCANet | 双流多通道注意力网络（通道/空间/时序注意力 + 金字塔池化） | >16GB |
| `dmca_net_enhanced.py` | DMCANetEnhanced | 增强版（+ Temporal Shift Module + Non-Local Block） | >24GB |
| `dmca_net_plus.py` | DMCANetEnhanced+ | 精简增强版 | >12GB |
| `advanced_models.py` | Transformer | CNN + Transformer（8头注意力，4层） | >12GB |
| `hybrid_models.py` | 3D-CNN+LSTM | 混合架构（3D卷积 + LSTM + 注意力 + 残差） | >12GB |
| `cnn_models.py` | CNN基线 | 纯CNN空间特征提取（最终采用的Stage 1） | 4GB |
| `lstm_models.py` | LSTM基线 | 纯LSTM时序建模（最终采用的Stage 2） | 2GB |
| `cnn_lstm_models.py` | CNN-LSTM | 端到端CNN-LSTM（最终采用的轻量版） | 6GB |
| `model_factory.py` | 工厂类 | 统一模型创建接口 | - |
| `bench_dmca_net.py` | 基准测试 | DMCANet性能和显存消耗评测脚本 | - |

## 原始位置

这些文件来自 `旧有文件/18/src/models/` 和 `旧有文件/18/src_1/models/`。

## 后续计划

1. 租赁 GPU 服务器（建议 A100 40GB 或 RTX 4090 24GB）
2. 在 CASME II 上用 LOSO 协议重新训练所有模型
3. 跨数据集（SMIC、SAMM、CAS(ME)³）验证
4. 完整的模型对比实验报告
