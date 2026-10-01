#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台文件锁实现
支持Windows和Linux/macOS系统
"""

import os
import sys
import time
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FileLock:
    """跨平台文件锁实现"""

    def __init__(self, lock_file_path: str, timeout: float = 30.0):
        """
        初始化文件锁

        Args:
            lock_file_path: 锁文件路径
            timeout: 获取锁的超时时间（秒）
        """
        self.lock_file_path = Path(lock_file_path)
        self.timeout = timeout
        self.lock_file = None
        self.is_locked = False

        # 确保锁文件目录存在
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self, blocking: bool = True) -> bool:
        """
        获取文件锁

        Args:
            blocking: 是否阻塞等待锁

        Returns:
            True: 成功获取锁
            False: 无法获取锁
        """
        start_time = time.time()

        while True:
            try:
                # 尝试创建锁文件（原子操作）
                if not self.lock_file_path.exists():
                    # 使用O_EXCL | O_CREAT标志创建文件（跨平台方式）
                    try:
                        fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        self.lock_file = os.fdopen(fd, 'w')

                        # 写入进程信息
                        self.lock_file.write(f"PID: {os.getpid()}\n")
                        self.lock_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        self.lock_file.write(f"Command: {' '.join(sys.argv)}\n")
                        self.lock_file.flush()
                        os.fsync(fd)  # 确保写入磁盘

                        self.is_locked = True
                        logger.info(f"成功获取文件锁: {self.lock_file_path}")
                        return True

                    except FileExistsError:
                        # 文件已存在，说明锁被其他进程持有
                        pass

                # 检查锁文件是否过期（处理异常情况）
                if self._is_lock_stale():
                    logger.warning(f"发现过期的锁文件: {self.lock_file_path}，尝试删除")
                    try:
                        self.lock_file_path.unlink()
                        continue  # 重试获取锁
                    except OSError:
                        pass

                if not blocking:
                    return False

                # 检查超时
                if time.time() - start_time > self.timeout:
                    logger.error(f"获取文件锁超时: {self.lock_file_path}")
                    return False

                # 等待一段时间后重试
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"获取文件锁时发生错误: {e}")
                return False

    def release(self):
        """释放文件锁"""
        try:
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None

            if self.is_locked and self.lock_file_path.exists():
                self.lock_file_path.unlink()
                self.is_locked = False
                logger.info(f"文件锁已释放: {self.lock_file_path}")

        except Exception as e:
            logger.error(f"释放文件锁时发生错误: {e}")

    def _is_lock_stale(self) -> bool:
        """
        检查锁文件是否过期（处理异常终止的情况）

        Returns:
            True: 锁文件过期
            False: 锁文件有效
        """
        try:
            if not self.lock_file_path.exists():
                return False

            # 读取锁文件内容
            with open(self.lock_file_path, 'r') as f:
                content = f.read()

            # 尝试提取PID
            for line in content.split('\n'):
                if line.startswith('PID: '):
                    try:
                        pid = int(line.split(':')[1].strip())
                        return not self._is_process_running(pid)
                    except (ValueError, IndexError):
                        break

            # 如果无法提取PID，检查文件修改时间
            file_mtime = self.lock_file_path.stat().st_mtime
            current_time = time.time()

            # 如果文件超过1小时未修改，认为是过期的
            return (current_time - file_mtime) > 3600

        except Exception as e:
            logger.warning(f"检查锁文件状态时发生错误: {e}")
            return True  # 出错时认为过期，允许清理

    def _is_process_running(self, pid: int) -> bool:
        """检查指定PID的进程是否还在运行"""
        try:
            if sys.platform == "win32":
                import psutil
                process = psutil.Process(pid)
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            else:
                # Unix系统使用kill命令检查
                os.kill(pid, 0)  # 发送信号0（不杀死进程）
                return True
        except (OSError, ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()

    def __del__(self):
        """析构函数，确保锁被释放"""
        self.release()