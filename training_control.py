# -*- coding: utf-8 -*-
"""Web 端与训练子进程之间的控制通道。

原实现里这条通道是断的：

- ``/api/train/pause`` 只把 progress.json 的 status 改成 ``paused``，而 main.py
  从不读这个字段，下一轮 epoch 结束又写回 ``running``——暂停按钮是假的。
- ``/api/train/stop`` 去 ``app.config['TRAINING_PROCESS']`` 取进程句柄，可这个键
  从来没人写过，所以停止接口永远返回 404。
- 子进程以 ``stdout=PIPE`` 启动却没人读管道。tqdm 每个 batch 都往 stdout 写，
  管道缓冲区（Linux 上 64KB）写满后子进程阻塞在 write() 上，训练就此卡死。
- progress.json 由两个进程同时读写，非原子写入时读方会读到半截 JSON。

这里把三件事收拢：控制指令走单独的 ``train_control.json``（Web 写、训练进程读），
子进程输出重定向到日志文件，所有 JSON 一律原子写入。
"""

import json
import os
import subprocess
import sys
import threading
import time

from config import Config

CONTROL_PATH = os.path.join(Config.runtime_dir, "train_control.json")
LOG_PATH = os.path.join(Config.runtime_dir, "training.log")
PID_PATH = os.path.join(Config.runtime_dir, "train_pid.txt")

COMMANDS = ("pause", "resume", "stop")

# 终态：训练进程不会再改写的状态
TERMINAL_STATUSES = ("completed", "failed", "stopped")


def atomic_write_json(path, data, **dump_kwargs):
    """先写临时文件再 os.replace，读方永远看不到写了一半的文件。"""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, **dump_kwargs)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 控制指令（Web 端写，训练进程读）
# ---------------------------------------------------------------------------
def send_command(command):
    if command not in COMMANDS:
        raise ValueError(f"未知的训练控制指令: {command}")
    atomic_write_json(CONTROL_PATH, {"command": command, "ts": time.time()})


def read_command():
    data = read_json(CONTROL_PATH) or {}
    command = data.get("command")
    return command if command in COMMANDS else None


def clear_command():
    try:
        os.remove(CONTROL_PATH)
    except FileNotFoundError:
        pass


class ControlPoller:
    """训练循环里每个 batch 调一次 ``check()``，按时间节流读控制文件。

    - 收到 ``pause``：调用 ``on_pause``，然后阻塞在这里直到 ``resume`` 或 ``stop``
    - 收到 ``stop``：返回 True，由训练循环负责保存好状态后退出

    暂停发生在 batch 之间，所以不会打断一次 optimizer.step() 或 torch.save()。
    """

    def __init__(self, interval=0.5, pause_poll=0.5, on_pause=None, on_resume=None):
        self.interval = interval
        self.pause_poll = pause_poll
        self.on_pause = on_pause
        self.on_resume = on_resume
        self._last = 0.0

    def check(self):
        now = time.monotonic()
        if now - self._last < self.interval:
            return False
        self._last = now

        command = read_command()
        if command == "stop":
            return True
        if command != "pause":
            return False

        if self.on_pause:
            self.on_pause()
        while True:
            time.sleep(self.pause_poll)
            command = read_command()
            if command == "stop":
                return True
            if command != "pause":
                if self.on_resume:
                    self.on_resume()
                return False


# ---------------------------------------------------------------------------
# 子进程管理（只在 Web 进程里用）
# ---------------------------------------------------------------------------
class TrainingProcessManager:
    """同一时刻最多一个训练进程。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._log_file = None

    def _reap(self):
        """子进程结束后关闭日志句柄、清掉 PID 文件。调用方需持有锁。"""
        if self._proc is not None and self._proc.poll() is not None:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            try:
                os.remove(PID_PATH)
            except FileNotFoundError:
                pass

    def returncode(self):
        with self._lock:
            return None if self._proc is None else self._proc.poll()

    def has_handle(self):
        with self._lock:
            return self._proc is not None

    def is_running(self):
        with self._lock:
            if self._proc is not None:
                running = self._proc.poll() is None
                if not running:
                    self._reap()
                return running
        # Web 进程重启过：手里没有句柄，只能靠 PID 文件判断
        return _pid_alive(_read_pid())

    def start(self, args=None, script=None):
        """启动训练脚本（默认 main.py）。已有训练在跑时抛 RuntimeError。"""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有训练任务正在运行")
            if self._proc is None and _pid_alive(_read_pid()):
                raise RuntimeError("已有训练任务正在运行")
            self._reap()

            clear_command()
            # 追加写：保留上一轮的日志尾巴，便于对照
            self._log_file = open(LOG_PATH, "a", encoding="utf-8")
            self._log_file.write(f"\n===== 训练启动 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._log_file.flush()

            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self._proc = subprocess.Popen(
                [sys.executable, script or os.path.join(Config.base_dir, "main.py"), *(args or [])],
                cwd=Config.base_dir,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            with open(PID_PATH, "w") as f:
                f.write(str(self._proc.pid))
            return self._proc.pid

    def terminate(self, timeout=10):
        """强制结束。优雅停止请用 send_command('stop')。"""
        with self._lock:
            proc = self._proc
        if proc is None:
            pid = _read_pid()
            if _pid_alive(pid):
                import signal
                os.kill(pid, signal.SIGTERM)
                return True
            return False
        if proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with self._lock:
            self._reap()
        return True


def _read_pid():
    try:
        with open(PID_PATH) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail_log(max_lines=10, path=None):
    """日志末尾若干行。tqdm 用 \\r 刷新进度条，按 \\r 切开只留最后一帧。"""
    path = path or LOG_PATH
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = []
    # 不能用 splitlines()：它把 \r 也当换行，tqdm 的每一帧都会变成单独一行
    for line in raw.split("\n"):
        line = line.split("\r")[-1].strip()
        if line:
            lines.append(line)
    return lines[-max_lines:]


manager = TrainingProcessManager()
