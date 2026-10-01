#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发保护模块
防止多个评估进程同时运行导致数据重复或冲突
"""

import os
import sys
import time
import psutil
import logging
from typing import Optional
from pathlib import Path
from file_lock import FileLock

logger = logging.getLogger(__name__)

class ConcurrencyProtector:
    """并发保护器，防止多个进程同时运行"""

    def __init__(self, lock_dir: str = "locks"):
        """
        初始化并发保护器

        Args:
            lock_dir: 锁文件目录
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(exist_ok=True)
        self.lock_file = None
        self.pid_file = None

    def acquire_process_lock(self, process_name: str = "evaluation") -> bool:
        """
        获取进程锁，防止多个相同进程同时运行

        Args:
            process_name: 进程名称，用于生成锁文件名

        Returns:
            True: 成功获取锁
            False: 无法获取锁（已有其他进程运行）
        """
        try:
            # 1. 首先检查PID文件
            pid_file_path = self.lock_dir / f"{process_name}.pid"

            if pid_file_path.exists():
                # 读取PID文件
                try:
                    with open(pid_file_path, 'r') as f:
                        old_pid = int(f.read().strip())

                    # 检查该PID是否还在运行
                    if self._is_process_running(old_pid):
                        logger.error(f"发现已有运行中的评估进程 (PID: {old_pid})，当前进程退出")
                        return False
                    else:
                        logger.info(f"发现孤立的PID文件 {old_pid}，进程已不存在，继续执行")
                        pid_file_path.unlink()  # 删除孤立的PID文件

                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"读取PID文件失败，可能是损坏的文件: {e}")
                    pid_file_path.unlink(missing_ok=True)

            # 2. 创建当前进程的PID文件
            current_pid = os.getpid()
            with open(pid_file_path, 'w') as f:
                f.write(str(current_pid))

            self.pid_file = pid_file_path
            logger.info(f"创建PID文件: {pid_file_path} (PID: {current_pid})")

            # 3. 创建文件锁
            lock_file_path = self.lock_dir / f"{process_name}.lock"

            try:
                # 使用跨平台文件锁
                file_lock = FileLock(str(lock_file_path), timeout=10.0)

                if file_lock.acquire(blocking=False):
                    self.file_lock = file_lock
                    logger.info(f"成功获取进程锁: {lock_file_path}")
                    return True
                else:
                    logger.error(f"无法获取进程锁 {lock_file_path}，已有其他评估进程在运行")
                    # 清理PID文件
                    if pid_file_path.exists():
                        pid_file_path.unlink()
                    return False

            except Exception as e:
                logger.error(f"获取文件锁时发生错误: {e}")
                # 清理PID文件
                if pid_file_path.exists():
                    pid_file_path.unlink()
                return False

        except Exception as e:
            logger.error(f"获取进程锁时发生错误: {e}")
            self.release_lock()  # 清理已获取的资源
            return False

    def release_lock(self):
        """释放所有锁和文件"""
        try:
            # 释放文件锁
            if hasattr(self, 'file_lock') and self.file_lock:
                self.file_lock.release()
                self.file_lock = None
                logger.info("文件锁已释放")

            # 删除PID文件
            if self.pid_file and self.pid_file.exists():
                self.pid_file.unlink()
                logger.info(f"PID文件已删除: {self.pid_file}")
                self.pid_file = None

        except Exception as e:
            logger.error(f"释放锁时发生错误: {e}")

    def _is_process_running(self, pid: int) -> bool:
        """
        检查指定PID的进程是否还在运行

        Args:
            pid: 进程ID

        Returns:
            True: 进程正在运行
            False: 进程不存在
        """
        try:
            # 使用psutil检查进程是否存在
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def get_running_processes_info(self, process_name: str = "evaluation") -> Optional[dict]:
        """
        获取当前运行的相关进程信息

        Args:
            process_name: 进程名称

        Returns:
            进程信息字典，如果没有运行中的进程则返回None
        """
        try:
            pid_file_path = self.lock_dir / f"{process_name}.pid"
            lock_file_path = self.lock_dir / f"{process_name}.lock"

            if pid_file_path.exists() and lock_file_path.exists():
                try:
                    with open(pid_file_path, 'r') as f:
                        pid = int(f.read().strip())

                    if self._is_process_running(pid):
                        # 读取锁文件内容
                        with open(lock_file_path, 'r') as f:
                            lock_info = f.read().strip()

                        return {
                            'pid': pid,
                            'lock_file': str(lock_file_path),
                            'pid_file': str(pid_file_path),
                            'lock_info': lock_info
                        }
                except (ValueError, FileNotFoundError):
                    pass

            return None

        except Exception as e:
            logger.error(f"获取进程信息时发生错误: {e}")
            return None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动释放锁"""
        self.release_lock()


def check_running_evaluation_processes(lock_dir: str = "locks") -> bool:
    """
    检查是否有评估进程正在运行

    Args:
        lock_dir: 锁文件目录

    Returns:
        True: 有进程在运行
        False: 没有进程在运行
    """
    protector = ConcurrencyProtector(lock_dir)
    process_info = protector.get_running_processes_info("evaluation")

    if process_info:
        print("发现正在运行的评估进程:")
        print(f"   PID: {process_info['pid']}")
        print(f"   锁文件: {process_info['lock_file']}")
        print("\n进程详细信息:")
        print(process_info['lock_info'])
        print("\n请等待当前进程完成，或手动终止后重试。")
        return True

    return False


def force_clean_stale_locks(lock_dir: str = "locks"):
    """
    强制清理无效的锁文件（用于恢复）

    Args:
        lock_dir: 锁文件目录
    """
    try:
        lock_dir_path = Path(lock_dir)
        if not lock_dir_path.exists():
            return

        protector = ConcurrencyProtector(lock_dir)
        cleaned = False

        # 检查所有.pid文件
        for pid_file in lock_dir_path.glob("*.pid"):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())

                if not protector._is_process_running(pid):
                    # 进程不存在，删除相关文件
                    lock_file = pid_file.with_suffix('.lock')
                    pid_file.unlink(missing_ok=True)
                    lock_file.unlink(missing_ok=True)
                    print(f"清理无效锁: PID {pid}")
                    cleaned = True

            except (ValueError, FileNotFoundError):
                # 损坏的文件，直接删除
                pid_file.unlink(missing_ok=True)
                lock_file = pid_file.with_suffix('.lock')
                lock_file.unlink(missing_ok=True)
                cleaned = True

        if not cleaned:
            print("没有发现需要清理的无效锁")

    except Exception as e:
        print(f"清理锁文件时发生错误: {e}")


if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser(description="并发保护工具")
    parser.add_argument("--check", action="store_true", help="检查是否有评估进程在运行")
    parser.add_argument("--clean", action="store_true", help="清理无效的锁文件")
    parser.add_argument("--lock-dir", default="locks", help="锁文件目录")

    args = parser.parse_args()

    if args.check:
        if check_running_evaluation_processes(args.lock_dir):
            sys.exit(1)
        else:
            print("✅ 没有发现运行中的评估进程")
            sys.exit(0)

    elif args.clean:
        force_clean_stale_locks(args.lock_dir)

    else:
        print("请使用 --check 或 --clean 参数")