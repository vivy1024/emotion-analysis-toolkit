#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMCA-Net模型性能评测脚本

比较不同配置下的DMCA-Net模型性能和资源消耗
"""

import os
import sys
import time
import torch
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# 设置项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入模型
from models.dmca_net import DMCANet

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_inference_time(model, input_tensor, num_iterations=20, warmup=5):
    """测量模型推理时间
    
    Args:
        model: 待测模型
        input_tensor: 输入张量
        num_iterations: 测量迭代次数
        warmup: 预热迭代次数
    
    Returns:
        平均推理时间(ms)
    """
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # 测量时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    # 计算平均推理时间(ms)
    avg_time = (end_time - start_time) * 1000 / num_iterations
    
    return avg_time

def measure_memory_usage(model, input_tensor):
    """测量模型内存使用
    
    Args:
        model: 待测模型
        input_tensor: 输入张量
    
    Returns:
        (参数数量, 峰值内存使用量(MB))
    """
    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 参数数量
    param_count = sum(p.numel() for p in model.parameters())
    
    # 运行前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 内存使用
    if torch.cuda.is_available():
        memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        memory_usage = 0  # CPU模式无法直接获取内存使用量
    
    return param_count, memory_usage

def benchmark_model_configs():
    """评测不同配置的模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 基本配置
    input_shape = (20, 128, 128, 4)
    output_shape = 7
    batch_size = 4
    
    # 创建随机输入
    x = torch.randn(batch_size, 4, input_shape[0], input_shape[1], input_shape[2]).to(device)
    
    # 模型配置变体
    configs = [
        {
            'name': 'DMCA-Net 基线',
            'params': {
                'use_channel_attention': True,
                'use_spatial_attention': True,
                'use_temporal_attention': True,
                'use_dynamic_weights': False,
                'use_diagonal_attention': False,
                'use_adversarial': False,
                'adv_weight': 0.1
            }
        },
        {
            'name': 'DMCA-Net++ (动态权重)',
            'params': {
                'use_channel_attention': True,
                'use_spatial_attention': True,
                'use_temporal_attention': True,
                'use_dynamic_weights': True,
                'use_diagonal_attention': False,
                'use_adversarial': False,
                'adv_weight': 0.1
            }
        },
        {
            'name': 'DMCA-Net++ (微注意力)',
            'params': {
                'use_channel_attention': True,
                'use_spatial_attention': True,
                'use_temporal_attention': True,
                'use_dynamic_weights': False,
                'use_diagonal_attention': True,
                'use_adversarial': False,
                'adv_weight': 0.1
            }
        },
        {
            'name': 'DMCA-Net++ (完整)',
            'params': {
                'use_channel_attention': True,
                'use_spatial_attention': True,
                'use_temporal_attention': True,
                'use_dynamic_weights': True,
                'use_diagonal_attention': True,
                'use_adversarial': False,
                'adv_weight': 0.1
            }
        },
        {
            'name': 'DMCA-Net+++ (对抗训练)',
            'params': {
                'use_channel_attention': True,
                'use_spatial_attention': True,
                'use_temporal_attention': True,
                'use_dynamic_weights': True,
                'use_diagonal_attention': True,
                'use_adversarial': True,
                'adv_weight': 0.1
            }
        }
    ]
    
    # 结果存储
    results = []
    
    # 测试各个配置
    for config in configs:
        logger.info(f"测试配置: {config['name']}")
        
        try:
            # 创建模型
            model = DMCANet(
                input_shape=input_shape,
                output_shape=output_shape,
                device=device,
                **config['params']
            )
            
            # 测量参数数量和内存使用
            params, memory = measure_memory_usage(model, x)
            
            # 测量推理时间
            inference_time = measure_inference_time(model, x)
            
            results.append({
                'name': config['name'],
                'params': params,
                'memory': memory,
                'time': inference_time
            })
            
            logger.info(f"结果: 参数量={params:,}, 内存={memory:.2f}MB, 推理时间={inference_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
            results.append({
                'name': config['name'],
                'params': 'ERROR',
                'memory': 'ERROR',
                'time': 'ERROR'
            })
    
    # 转换为DataFrame并显示
    df = pd.DataFrame(results)
    
    # 使用tabulate库美化显示
    table = tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False)
    logger.info(f"性能评测结果:\n{table}")
    
    return results

def main():
    """主函数"""
    logger.info("开始DMCA-Net++性能评测")
    benchmark_model_configs()
    logger.info("性能评测完成")

if __name__ == "__main__":
    main() 