"""
函数执行时间统计装饰器模块

提供 timer 装饰器用于统计函数执行时间，支持普通函数和类方法。
"""

import time
import functools
import logging
from typing import Callable, Optional, Any

from common.custom_logger import CustomLogger

# 全局 logger 实例
_timer_logger = CustomLogger(name="timer_decorator", level=logging.DEBUG)


def timer(
    level: int = logging.DEBUG,
    precision: int = 4,
    threshold_ms: Optional[float] = None
) -> Callable:
    """
    统计函数执行时间的装饰器
    
    Args:
        level: 日志级别，默认 DEBUG
        precision: 时间精度（小数点位数），默认 4 位
        threshold_ms: 仅当执行时间超过此阈值（毫秒）时才记录日志，默认 None 表示始终记录
    
    Returns:
        装饰器函数
    
    Example:
        # 基本用法
        @timer()
        def my_function():
            pass
        
        # 指定日志级别
        @timer(level=logging.INFO)
        def my_function():
            pass
        
        # 仅记录执行时间超过 100ms 的调用
        @timer(threshold_ms=100)
        def my_function():
            pass
    
    数学原理:
        执行时间 \\( T = t_{end} - t_{start} \\)
        其中 \\( t_{start} \\) 和 \\( t_{end} \\) 分别是函数调用前后的高精度时间戳（perf_counter）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 获取函数完整名称（处理类方法的情况）
            if args and hasattr(args[0], '__class__'):
                # 可能是类方法，尝试获取类名
                cls_name = args[0].__class__.__name__
                func_name = f"{cls_name}.{func.__name__}"
            else:
                func_name = func.__qualname__
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            try:
                # 执行原函数
                result = func(*args, **kwargs)
                return result
            finally:
                # 计算执行时间
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                elapsed_ms = elapsed_time * 1000
                
                # 检查是否需要记录（基于阈值）
                should_log = threshold_ms is None or elapsed_ms >= threshold_ms
                
                if should_log:
                    # 格式化输出
                    if elapsed_ms < 1:
                        time_str = f"{elapsed_time * 1e6:.{precision}f} us"
                    elif elapsed_ms < 1000:
                        time_str = f"{elapsed_ms:.{precision}f} ms"
                    else:
                        time_str = f"{elapsed_time:.{precision}f} s"
                    
                    msg = f"[TIMER] {func_name}() executed in {time_str}"
                    
                    # 使用 stacklevel 确保日志显示正确的调用位置
                    _timer_logger._log(level, msg, stacklevel=1)
        
        return wrapper
    return decorator


def timer_sync(
    level: int = logging.DEBUG,
    precision: int = 4,
    sync_cuda: bool = True
) -> Callable:
    """
    带 CUDA 同步的计时装饰器（用于 GPU 操作的精确计时）
    
    由于 CUDA 操作是异步的，直接测量时间可能不准确。
    此装饰器在测量前后添加 torch.cuda.synchronize() 确保准确性。
    
    Args:
        level: 日志级别
        precision: 时间精度
        sync_cuda: 是否同步 CUDA，默认 True
    
    数学原理:
        GPU 异步执行模型下，CPU 侧测量的时间 \\( T_{cpu} \\) 可能小于实际 GPU 执行时间 \\( T_{gpu} \\)
        通过在测量点调用 synchronize()，强制等待 GPU 完成所有操作，使 \\( T_{measured} = T_{gpu} \\)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 获取函数名
            if args and hasattr(args[0], '__class__'):
                cls_name = args[0].__class__.__name__
                func_name = f"{cls_name}.{func.__name__}"
            else:
                func_name = func.__qualname__
            
            # CUDA 同步（如果可用且启用）
            try:
                import torch
                cuda_available = sync_cuda and torch.cuda.is_available()
            except ImportError:
                cuda_available = False
            
            if cuda_available:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cuda_available:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                
                if elapsed_ms < 1:
                    time_str = f"{(end_time - start_time) * 1e6:.{precision}f} us"
                elif elapsed_ms < 1000:
                    time_str = f"{elapsed_ms:.{precision}f} ms"
                else:
                    time_str = f"{end_time - start_time:.{precision}f} s"
                
                sync_tag = "[CUDA SYNC]" if cuda_available else ""
                msg = f"[TIMER]{sync_tag} {func_name}() executed in {time_str}"
                _timer_logger._log(level, msg, stacklevel=1)
        
        return wrapper
    return decorator
