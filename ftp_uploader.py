#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTP 上传工具：将本地生成的视频/图片拷贝到 NAS，成功后删除容器内本地文件。
新增：NAS 远程 FTP 80% 容量超限自动巡检，按最旧时间顺序淘汰 MP4 视频与图片直到 60% 以下。
"""

import os
import ftplib
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


class FtpUploader:
    def __init__(self, ftp_cfg: dict):
        self.enabled = bool(ftp_cfg.get("enabled", False))
        self.host = ftp_cfg.get("host", "172.10.1.15")
        self.port = int(ftp_cfg.get("port", 21))
        self.username = ftp_cfg.get("username", "ftpuser")
        self.password = ftp_cfg.get("password", "Hirain123")
        self.passive = bool(ftp_cfg.get("passive", True))
        self.timeout = int(ftp_cfg.get("timeout", 30))
        
        # 容量淘汰策略参数：80% 触发清理，淘汰最旧数据直到回落到 60% 以下
        self.cleanup_threshold = float(ftp_cfg.get("cleanup_threshold", 0.80))
        self.cleanup_target_ratio = float(ftp_cfg.get("cleanup_target_ratio", 0.60))
        self.max_storage_size_gb = float(ftp_cfg.get("max_storage_size_gb", 720.0))  # 默认 720GB
        self.clean_paths: List[str] = ftp_cfg.get("clean_paths", ["nvme1/Algo-result"])

        self._lock = threading.Lock()
        self._clean_thread = None
        self._stop_clean = threading.Event()

        if self.enabled and self.clean_paths:
            self.start_remote_capacity_cleanup_loop()

    def _connect(self) -> ftplib.FTP:
        ftp = ftplib.FTP()
        ftp.connect(self.host, self.port, timeout=self.timeout)
        ftp.login(self.username, self.password)
        ftp.set_pasv(self.passive)
        return ftp

    @staticmethod
    def _normalize_remote(remote_path: str) -> str:
        return remote_path.replace("\\", "/").lstrip("/")

    def _ensure_remote_dirs(self, ftp: ftplib.FTP, remote_dir: str):
        remote_dir = self._normalize_remote(remote_dir)
        if not remote_dir:
            return
        parts = [p for p in remote_dir.split("/") if p]
        
        try:
            ftp.cwd("/")
        except Exception:
            pass

        for part in parts:
            try:
                ftp.cwd(part)
            except Exception:
                try:
                    ftp.mkd(part)
                except Exception as e2:
                    print(f"[FTP] 创建目录 '{part}' 失败: {e2}")
                try:
                    ftp.cwd(part)
                except Exception as e3:
                    raise RuntimeError(f"无法进入远程目录 '{part}': {e3}") from e3

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """上传单个文件到 NAS。返回是否成功。"""
        if not self.enabled:
            return False

        local_path = str(local_path)
        remote_path = self._normalize_remote(remote_path)
        if not os.path.isfile(local_path):
            print(f"[FTP] 本地文件不存在，跳过: {local_path}")
            return False

        remote_dir = os.path.dirname(remote_path).replace("\\", "/")
        remote_name = os.path.basename(remote_path)

        with self._lock:
            ftp = None
            try:
                ftp = self._connect()
                if remote_dir:
                    self._ensure_remote_dirs(ftp, remote_dir)
                else:
                    ftp.cwd("/")

                with open(local_path, "rb") as fp:
                    ftp.storbinary(f"STOR {remote_name}", fp)

                print(f"[FTP] 上传成功: {local_path} -> ftp://{self.host}/{remote_path}")
                return True
            except Exception as e:
                print(f"[FTP] 上传失败: {local_path} -> {remote_path}, 错误: {e}")
                return False
            finally:
                if ftp is not None:
                    try:
                        ftp.quit()
                    except Exception:
                        try:
                            ftp.close()
                        except Exception:
                            pass

    def upload_and_delete(self, local_path: str, remote_path: str) -> bool:
        """上传到 NAS 成功后删除容器内本地文件。"""
        ok = self.upload_file(local_path, remote_path)
        if not ok:
            return False
        try:
            os.remove(local_path)
            print(f"[FTP] 已删除容器本地文件: {local_path}")
            return True
        except OSError as e:
            print(f"[FTP] 上传成功但删除本地失败: {local_path}, 错误: {e}")
            return False

    def map_local_to_remote(self, local_path: str, local_root: str, remote_root: str) -> str:
        """将本地路径映射为 NAS 远程路径。"""
        local_path = Path(local_path).resolve()
        local_root = Path(local_root).resolve()
        try:
            rel = local_path.relative_to(local_root).as_posix()
        except ValueError:
            rel = local_path.name
        remote_root = remote_root.replace("\\", "/").rstrip("/")
        return f"{remote_root}/{rel}"

    # -----------------------------------------------------------------
    # NAS 远程 FTP 80% -> 60% 按时间淘汰清理核心逻辑
    # -----------------------------------------------------------------
    def start_remote_capacity_cleanup_loop(self):
        """启动后台线程，定时巡检 NAS 磁盘使用率，80% 触发淘汰至 60%"""
        self._stop_clean.clear()
        self._clean_thread = threading.Thread(target=self._remote_capacity_cleanup_worker, daemon=True)
        self._clean_thread.start()
        print(f"[FTP-Clean] 已开启 NAS 远程容量巡检: 阈值 {int(self.cleanup_threshold*100)}% 触发, 清理目标 {int(self.cleanup_target_ratio*100)}%")

    def _remote_capacity_cleanup_worker(self):
        time.sleep(30)
        while not self._stop_clean.is_set():
            try:
                self.run_remote_capacity_cleanup()
            except Exception as e:
                print(f"[FTP-Clean] NAS 远程容量巡检异常: {e}")
            
            # 每隔 10 分钟巡检一次容量
            for _ in range(60):
                if self._stop_clean.is_set():
                    break
                time.sleep(10)

    def _collect_ftp_files_recursive(self, ftp: ftplib.FTP, current_dir: str, file_list: List[Dict[str, Any]]):
        """递归收集 FTP 目录下的所有文件路径、大小及创建时间信息"""
        current_dir = self._normalize_remote(current_dir)
        try:
            ftp.cwd("/" + current_dir)
        except Exception:
            return

        entries = []
        try:
            ftp.retrlines("NLST", entries.append)
        except Exception:
            pass

        for entry in entries:
            name = os.path.basename(entry)
            if name in (".", ".."):
                continue
            item_path = f"{current_dir}/{name}"
            
            is_dir = False
            try:
                ftp.cwd("/" + item_path)
                is_dir = True
                ftp.cwd("/" + current_dir)
            except Exception:
                is_dir = False

            if is_dir:
                self._collect_ftp_files_recursive(ftp, item_path, file_list)
            else:
                # 获取文件大小
                file_size = 0
                try:
                    file_size = ftp.size(name) or 0
                except Exception:
                    file_size = 0

                # 记录文件路径（路径天然包含时间 07-29/001000...，可直接按路径排序）
                file_list.append({
                    "path": item_path,
                    "name": name,
                    "dir": current_dir,
                    "size": file_size
                })

    def _remove_empty_ftp_dirs(self, ftp: ftplib.FTP, base_dir: str):
        """递归清理空目录"""
        base_dir = self._normalize_remote(base_dir)
        try:
            ftp.cwd("/" + base_dir)
        except Exception:
            return

        entries = []
        try:
            ftp.retrlines("NLST", entries.append)
        except Exception:
            pass

        for entry in entries:
            name = os.path.basename(entry)
            if name in (".", ".."):
                continue
            sub_path = f"{base_dir}/{name}"
            try:
                ftp.cwd("/" + sub_path)
                self._remove_empty_ftp_dirs(ftp, sub_path)
                ftp.cwd("/" + base_dir)
                ftp.rmd(sub_path)
            except Exception:
                pass

    def run_remote_capacity_cleanup(self):
        """按容量阈值（80% 触发清理，淘汰最旧数据直到回落至 60%）清理 NAS 远程磁盘"""
        if not self.enabled or not self.clean_paths:
            return

        max_bytes = int(self.max_storage_size_gb * (1024 ** 3))
        cleanup_threshold_bytes = max_bytes * self.cleanup_threshold
        cleanup_target_bytes = max_bytes * self.cleanup_target_ratio

        with self._lock:
            ftp = None
            try:
                ftp = self._connect()
                for base_clean_path in self.clean_paths:
                    base_clean_path = self._normalize_remote(base_clean_path)
                    try:
                        ftp.cwd("/" + base_clean_path)
                    except Exception:
                        continue

                    all_files: List[Dict[str, Any]] = []
                    self._collect_ftp_files_recursive(ftp, base_clean_path, all_files)

                    total_bytes = sum(f["size"] for f in all_files)
                    usage_ratio = total_bytes / max_bytes if max_bytes > 0 else 0.0

                    if total_bytes >= cleanup_threshold_bytes:
                        print(
                            f"[FTP-Clean] ⚠️ 远程 NAS [{base_clean_path}] 容量超过 {int(self.cleanup_threshold*100)}% "
                            f"({total_bytes/(1024**3):.2f}GB / {self.max_storage_size_gb:.2f}GB, {usage_ratio*100:.1f}%)，"
                            f"开始按时间顺序淘汰最旧视频至 {int(self.cleanup_target_ratio*100)}% ({cleanup_target_bytes/(1024**3):.2f}GB)..."
                        )
                        # 按路径/名称（包含时间戳）升序排列，保证最旧的文件排在最前面
                        all_files.sort(key=lambda x: x["path"])

                        for file_info in all_files:
                            if total_bytes <= cleanup_target_bytes:
                                print(f"[FTP-Clean] ✅ 远程 NAS [{base_clean_path}] 容量已成功回落至 {int(self.cleanup_target_ratio*100)}% 以下 ({total_bytes/(1024**3):.2f}GB)，停止淘汰。")
                                break
                            try:
                                ftp.delete("/" + file_info["path"])
                                total_bytes -= file_info["size"]
                                print(f"[FTP-Clean] 已淘汰 NAS 最旧文件: {file_info['path']} ({file_info['size']/(1024**2):.1f}MB)")
                            except Exception as e:
                                print(f"[FTP-Clean] 删除远程文件失败 {file_info['path']}: {e}")

                        # 顺便清理空日期目录
                        self._remove_empty_ftp_dirs(ftp, base_clean_path)
            except Exception as e:
                print(f"[FTP-Clean] 远程 NAS 容量巡检淘汰失败: {e}")
            finally:
                if ftp is not None:
                    try:
                        ftp.quit()
                    except Exception:
                        try:
                            ftp.close()
                        except Exception:
                            pass


_global_uploader: Optional[FtpUploader] = None
_global_lock = threading.Lock()


def init_ftp_uploader(ftp_cfg: dict) -> Optional[FtpUploader]:
    global _global_uploader
    with _global_lock:
        if not ftp_cfg or not ftp_cfg.get("enabled", False):
            _global_uploader = None
            print("[FTP] 未启用 FTP 上传。")
            return None
        _global_uploader = FtpUploader(ftp_cfg)
        print(
            f"[FTP] 已启用 NAS 上传: {ftp_cfg.get('username')}@{ftp_cfg.get('host')}:{ftp_cfg.get('port', 21)}"
        )
        return _global_uploader


def get_ftp_uploader() -> Optional[FtpUploader]:
    return _global_uploader
