import logging
import logging.handlers
import sys
import queue
from typing import Optional, Union, Any
import os
import atexit

try:
    import torch
    import torch.distributed as dist
except Exception:
    torch = None
    dist = None

try:
    import numpy as np
except Exception:
    np = None


def _get_rank() -> int:
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


class RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.rank = _get_rank()
        except Exception:
            record.rank = 0
        return True


class SingletonType(type):
    """单例元类：确保全局唯一的日志管理器"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        print(f"[SingletonType.__call__] 被调用，cls={cls.__name__}")
        print(f"[SingletonType.__call__] 当前缓存: {list(cls._instances.keys())}")
        
        if cls not in cls._instances:
            print(f"[SingletonType.__call__] 缓存中没有，创建新实例")
            instance = super(SingletonType, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
            print(f"[SingletonType.__call__] 新实例 id: {id(instance)}")
        else:
            print(f"[SingletonType.__call__] 从缓存返回已有实例 id: {id(cls._instances[cls])}")
        
        return cls._instances[cls]




class CustomLogger(metaclass=SingletonType):
    """
    单例模式的日志管理器
    """
    
    def __init__(self, 
                 name: str = "default_logger",
                 level: Union[int, str] = logging.INFO,
                 log_file: Optional[str] = None,
                 log_format: str = "[%(levelname)s][%(threadName)s] %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
                 use_queue: bool = True):
        """
        初始化Logger配置（单例模式下，后续调用不会重新初始化）
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径
            log_format: 日志格式
            use_queue: 是否使用队列模式（多线程安全，日志不会交错）
        """
        print(f"[CustomLogger.__init__] 被调用，self id: {id(self)}")
        
        # 单例模式下，只有第一次调用会执行初始化
        if hasattr(self, '_initialized') and self._initialized:
            print(f"[CustomLogger.__init__] 已初始化，跳过")
            return
        
        print(f"[CustomLogger.__init__] 开始初始化...")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # 添加 rank 过滤器
        self.logger.addFilter(RankFilter())
        self._formatter = logging.Formatter(log_format)
        
        self._use_queue = use_queue
        self._queue_listener = None
        self._log_queue = None
        self._actual_handlers = []  # 实际输出的处理器
        
        if use_queue:
            # 使用队列模式：多线程日志不会交错
            self._log_queue = queue.Queue(-1)  # 无限大小队列
            queue_handler = logging.handlers.QueueHandler(self._log_queue)
            self.logger.addHandler(queue_handler)
            
            # 创建实际的处理器
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(self._formatter)
            self._actual_handlers.append(ch)
            
            # 添加文件处理器（如果指定）
            if log_file:
                fh = self._create_file_handler(log_file)
                self._actual_handlers.append(fh)
            
            # 启动队列监听器
            self._queue_listener = logging.handlers.QueueListener(
                self._log_queue, *self._actual_handlers, respect_handler_level=True
            )
            self._queue_listener.start()
            
            # 注册退出时停止监听器
            atexit.register(self._stop_queue_listener)
        else:
            # 传统模式：直接添加处理器
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(self._formatter)
            self.logger.addHandler(ch)
            
            # 添加文件处理器（如果指定）
            if log_file:
                self.add_file_handler(log_file)
        
        self._initialized = True
        print(f"[CustomLogger.__init__] 初始化完成, use_queue={use_queue}")
    
    def _stop_queue_listener(self):
        """停止队列监听器"""
        if self._queue_listener:
            self._queue_listener.stop()
            self._queue_listener = None
    
    def _create_file_handler(self, log_file: str, mode: str = "a") -> logging.FileHandler:
        """创建文件处理器"""
        try:
            os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        except Exception:
            pass
        fh = logging.FileHandler(log_file, mode=mode, encoding="utf-8", delay=False)
        fh.setFormatter(self._formatter)
        return fh


    def _log(self, level, message, *args, **kwargs):
        """内部日志方法，调整stacklevel以正确记录调用位置
        
        支持通过 stacklevel 参数额外调整调用栈层级：
        - 默认 stacklevel=0，对应基础偏移量 3（调用者 -> debug/info -> _log -> logger.log）
        - 如果从 wrapper 函数（如 log_var）调用，设置 stacklevel=1 表示再往上跳1层
        """
        # 提取自定义的 stacklevel 参数（默认 0，表示不额外偏移）
        extra_stacklevel = kwargs.pop('stacklevel', 0)
        base_stacklevel = 3  # 基础偏移量
        
        if args:
            # 兼容 print 风格的多参数打印
            # 如果 message 是字符串且包含 %，尝试判断是否为格式化字符串
            is_formatting = False
            if isinstance(message, str) and '%' in message:
                try:
                    # 尝试格式化，如果成功则认为是 logging 格式化用法
                    # 注意：这里可能会有误判，比如 logger.info("Value: %s", "a", "b") 会抛错，然后走拼接逻辑
                    # 但这通常是用户写错了，拼接打印出来也能看到问题
                    _ = message % args
                    is_formatting = True
                except (TypeError, ValueError):
                    is_formatting = False
            
            if not is_formatting:
                message = " ".join(map(str, (message,) + args))
                args = ()

        self.logger.log(level, message, *args, stacklevel=base_stacklevel + extra_stacklevel, **kwargs)
        
    def debug(self, message, *args, **kwargs):
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._log(logging.CRITICAL, message, *args, **kwargs)
    
    def add_file_handler(self, log_file: str, mode: str = "a"):
        """动态添加文件 handler
        
        注意：在队列模式下，需要重启队列监听器才能生效
        """
        fh = self._create_file_handler(log_file, mode)
        
        if self._use_queue and self._queue_listener:
            # 队列模式：需要停止监听器，添加新处理器，然后重启
            self._queue_listener.stop()
            self._actual_handlers.append(fh)
            self._queue_listener = logging.handlers.QueueListener(
                self._log_queue, *self._actual_handlers, respect_handler_level=True
            )
            self._queue_listener.start()
        else:
            # 传统模式：直接添加
            self.logger.addHandler(fh)
        return fh
    
    def remove_console_handlers(self):
        """移除控制台输出"""
        handlers_to_remove = []
        for h in self.logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                stream = getattr(h, 'stream', None)
                if stream in (sys.stdout, sys.stderr):
                    handlers_to_remove.append(h)
        for h in handlers_to_remove:
            self.logger.removeHandler(h)
    
    def flush(self):
        """刷新所有 handler 的缓冲区"""
        for h in self.logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    
    def log_var(self, name: str, var: Any, extra_info: str = "", level: int = logging.DEBUG, stacklevel: int = 0):
        """
        辅助方法：记录变量信息，自动检测 tensor/numpy/list/dict 等类型并输出相关信息
        
        Args:
            name: 变量名（用于日志显示）
            var: 变量值
            extra_info: 额外信息（可选）
            level: 日志级别，默认 DEBUG
            stacklevel: 额外的栈层级偏移，用于从包装函数调用时正确定位（默认 0）
        
        Example:
            logger.log_var("input_tensor", x)
            # 输出: input_tensor: Tensor, shape=torch.Size([1, 3, 256, 256]), dtype=torch.float32, device=cuda:0
            
            logger.log_var("config", cfg_dict)
            # 输出: config: dict, keys=['lr', 'batch_size', 'epochs']
        """
        if var is None:
            msg = f"{name} = None {extra_info}"
        elif torch is not None and isinstance(var, torch.Tensor):
            msg = f"{name}: Tensor, shape={var.shape}, range=[{var.min()}, {var.max()}], dtype={var.dtype}, device={var.device} {extra_info}"
        elif np is not None and isinstance(var, np.ndarray):
            msg = f"{name}: ndarray, shape={var.shape}, range=[{var.min()}, {var.max()}], dtype={var.dtype} {extra_info}"
        elif isinstance(var, (list, tuple)):
            # 对于 list/tuple，额外显示第一个元素的类型（如果存在）
            if len(var) > 0:
                first_type = type(var[0]).__name__
                msg = f"{name}: {type(var).__name__}, len={len(var)}, first_elem_type={first_type} {extra_info}"
            else:
                msg = f"{name}: {type(var).__name__}, len={len(var)} {extra_info}"
        elif isinstance(var, dict):
            msg = f"{name}: dict, keys={list(var.keys())} {extra_info}"
        else:
            # 对于其他类型，显示类型和值（如果值不太长）
            var_str = str(var)
            if len(var_str) > 100:
                var_str = var_str[:100] + "..."
            msg = f"{name}: {type(var).__name__} = {var_str} {extra_info}"
        
        # stacklevel=1 表示在默认基础上再往上跳1层，定位到实际调用 log_var 的位置
        # 额外的 stacklevel 参数用于从包装函数调用时正确定位
        self._log(level, msg, stacklevel=stacklevel)
