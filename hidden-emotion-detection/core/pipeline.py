#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理管道模块
提供数据处理流程的抽象和实现，支持组合式管道构建
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional, TypeVar, Generic, Tuple
from queue import Queue

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pipeline")

# 泛型类型定义
T = TypeVar('T')  # 输入类型
R = TypeVar('R')  # 输出类型

class PipelineStage(Generic[T, R], ABC):
    """管道阶段抽象基类"""
    
    def __init__(self, name: str = None):
        """
        初始化管道阶段
        
        Args:
            name: 阶段名称
        """
        self.name = name or self.__class__.__name__
        self.is_active = True
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }
    
    @abstractmethod
    def process(self, data: T) -> R:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        pass
    
    def __call__(self, data: T) -> R:
        """
        使阶段可调用
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        if not self.is_active:
            # 如果阶段未激活，直接返回输入数据
            return data
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 处理数据
            result = self.process(data)
            
            # 更新统计信息
            self.stats["processed_count"] += 1
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["processed_count"]
            )
            
            return result
            
        except Exception as e:
            # 记录错误
            logger.error(f"管道阶段 {self.name} 处理失败: {e}")
            self.stats["error_count"] += 1
            # 重新抛出异常或返回None
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def activate(self) -> None:
        """激活阶段"""
        self.is_active = True
    
    def deactivate(self) -> None:
        """停用阶段"""
        self.is_active = False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }


class Pipeline(Generic[T, R]):
    """数据处理管道，由多个处理阶段组成"""
    
    def __init__(self, name: str = "Pipeline"):
        """
        初始化管道
        
        Args:
            name: 管道名称
        """
        self.name = name
        self.stages: List[PipelineStage] = []
        self.is_active = True
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "stage_stats": {}
        }
    
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """
        添加处理阶段
        
        Args:
            stage: 处理阶段
            
        Returns:
            管道本身，支持链式调用
        """
        self.stages.append(stage)
        return self
    
    def process(self, data: T) -> R:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        if not self.is_active:
            # 如果管道未激活，直接返回输入数据
            return data
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 依次通过每个阶段
            result = data
            for stage in self.stages:
                if stage.is_active:
                    result = stage(result)
            
            # 更新统计信息
            self.stats["processed_count"] += 1
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["processed_count"]
            )
            
            # 更新阶段统计信息
            for stage in self.stages:
                self.stats["stage_stats"][stage.name] = stage.get_stats()
            
            return result
            
        except Exception as e:
            # 记录错误
            logger.error(f"管道 {self.name} 处理失败: {e}")
            self.stats["error_count"] += 1
            # 重新抛出异常
            raise
    
    def __call__(self, data: T) -> R:
        """
        使管道可调用
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        return self.process(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def activate(self) -> None:
        """激活管道"""
        self.is_active = True
    
    def deactivate(self) -> None:
        """停用管道"""
        self.is_active = False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "stage_stats": {}
        }
        
        # 重置所有阶段的统计信息
        for stage in self.stages:
            stage.reset_stats()


class AsyncPipeline(Generic[T, R]):
    """异步数据处理管道，支持并行处理"""
    
    def __init__(self, pipeline: Pipeline[T, R], max_queue_size: int = 100, num_workers: int = 1):
        """
        初始化异步管道
        
        Args:
            pipeline: 同步处理管道
            max_queue_size: 最大队列大小
            num_workers: 工作线程数量
        """
        self.pipeline = pipeline
        self.input_queue = Queue(maxsize=max_queue_size)
        self.result_callbacks: Dict[int, Callable[[R], None]] = {}
        self.error_callbacks: Dict[int, Callable[[Exception], None]] = {}
        self.task_counter = 0
        self.num_workers = num_workers
        self.workers = []
        self.is_running = False
        self.lock = threading.Lock()
    
    def start(self) -> None:
        """启动异步处理"""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.workers = []
            
            # 创建工作线程
            for i in range(self.num_workers):
                worker = threading.Thread(
                    target=self._worker_loop, 
                    name=f"{self.pipeline.name}-Worker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"异步管道 {self.pipeline.name} 已启动，工作线程数: {self.num_workers}")
    
    def stop(self) -> None:
        """停止异步处理"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # 等待所有线程结束
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
            
            logger.info(f"异步管道 {self.pipeline.name} 已停止")
    
    def submit(self, data: T, callback: Callable[[R], None] = None, 
              error_callback: Callable[[Exception], None] = None) -> int:
        """
        提交数据进行处理
        
        Args:
            data: 输入数据
            callback: 结果回调函数
            error_callback: 错误回调函数
            
        Returns:
            任务ID
        """
        if not self.is_running:
            raise RuntimeError("异步管道未启动")
        
        with self.lock:
            # 生成任务ID
            task_id = self.task_counter
            self.task_counter += 1
            
            # 注册回调
            if callback:
                self.result_callbacks[task_id] = callback
            if error_callback:
                self.error_callbacks[task_id] = error_callback
            
            # 包装任务
            task = (task_id, data)
            
            # 放入队列
            self.input_queue.put(task)
            
            return task_id
    
    def _worker_loop(self) -> None:
        """工作线程循环"""
        while self.is_running:
            try:
                # 获取任务
                try:
                    task_id, data = self.input_queue.get(timeout=0.1)
                except Exception:
                    # 队列为空，继续循环
                    continue
                
                try:
                    # 处理数据
                    result = self.pipeline(data)
                    
                    # 调用回调函数
                    with self.lock:
                        callback = self.result_callbacks.pop(task_id, None)
                    
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"结果回调错误: {e}")
                
                except Exception as e:
                    # 处理错误
                    logger.error(f"处理任务 {task_id} 失败: {e}")
                    
                    # 调用错误回调
                    with self.lock:
                        error_callback = self.error_callbacks.pop(task_id, None)
                    
                    if error_callback:
                        try:
                            error_callback(e)
                        except Exception as e:
                            logger.error(f"错误回调错误: {e}")
                
                finally:
                    # 标记任务完成
                    self.input_queue.task_done()
                    
                    # 清理回调
                    with self.lock:
                        self.result_callbacks.pop(task_id, None)
                        self.error_callbacks.pop(task_id, None)
            
            except Exception as e:
                logger.error(f"工作线程异常: {e}")
                if not self.is_running:
                    break
    
    def wait_empty(self, timeout: float = None) -> bool:
        """
        等待队列清空
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否清空
        """
        try:
            self.input_queue.join(timeout=timeout)
            return True
        except Exception:
            return False


class FunctionStage(PipelineStage[T, R]):
    """函数包装阶段，将函数包装为管道阶段"""
    
    def __init__(self, func: Callable[[T], R], name: str = None):
        """
        初始化函数包装阶段
        
        Args:
            func: 处理函数
            name: 阶段名称
        """
        super().__init__(name or func.__name__)
        self.func = func
    
    def process(self, data: T) -> R:
        """处理数据"""
        return self.func(data)


class FilterStage(PipelineStage[T, Optional[T]]):
    """过滤阶段，根据条件过滤数据"""
    
    def __init__(self, predicate: Callable[[T], bool], name: str = None):
        """
        初始化过滤阶段
        
        Args:
            predicate: 过滤条件函数，返回True保留数据，返回False过滤掉
            name: 阶段名称
        """
        super().__init__(name or "Filter")
        self.predicate = predicate
    
    def process(self, data: T) -> Optional[T]:
        """处理数据"""
        if self.predicate(data):
            return data
        return None


class BranchPipeline(PipelineStage[T, R]):
    """分支管道，根据条件选择不同处理路径"""
    
    def __init__(self, condition: Callable[[T], bool], 
                true_branch: PipelineStage[T, R], 
                false_branch: PipelineStage[T, R],
                name: str = None):
        """
        初始化分支管道
        
        Args:
            condition: 条件函数，返回True选择true_branch，返回False选择false_branch
            true_branch: 条件为True时的处理阶段
            false_branch: 条件为False时的处理阶段
            name: 阶段名称
        """
        super().__init__(name or "Branch")
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
    
    def process(self, data: T) -> R:
        """处理数据"""
        if self.condition(data):
            return self.true_branch(data)
        else:
            return self.false_branch(data)


# 管道构建辅助函数
def pipeline(stages: List[PipelineStage], name: str = "Pipeline") -> Pipeline:
    """
    创建管道
    
    Args:
        stages: 处理阶段列表
        name: 管道名称
        
    Returns:
        管道对象
    """
    pipe = Pipeline(name)
    for stage in stages:
        pipe.add_stage(stage)
    return pipe


def function_stage(func: Callable) -> FunctionStage:
    """
    将函数转换为管道阶段
    
    Args:
        func: 处理函数
        
    Returns:
        函数包装阶段
    """
    return FunctionStage(func) 