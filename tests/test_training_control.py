# -*- coding: utf-8 -*-
"""Web 端 ↔ 训练进程控制通道。

原实现里暂停只改 progress.json（训练进程从不读），停止去取一个从没人写过的
app.config 键（永远 404），子进程 stdout=PIPE 却没人读（输出满 64KB 即卡死）。
"""

import json
import os
import sys
import textwrap
import threading
import time

import pytest

import training_control as tc


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "CONTROL_PATH", str(tmp_path / "control.json"))
    monkeypatch.setattr(tc, "LOG_PATH", str(tmp_path / "training.log"))
    monkeypatch.setattr(tc, "PID_PATH", str(tmp_path / "pid.txt"))
    return tmp_path


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "p.json"
    tc.atomic_write_json(str(path), {"a": "中文"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "中文"}
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


def test_read_json_tolerates_missing_and_corrupt(tmp_path):
    assert tc.read_json(str(tmp_path / "nope.json"), default={}) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{half", encoding="utf-8")
    assert tc.read_json(str(bad)) is None


def test_command_roundtrip_and_validation(ctl):
    assert tc.read_command() is None
    tc.send_command("pause")
    assert tc.read_command() == "pause"
    tc.clear_command()
    assert tc.read_command() is None
    with pytest.raises(ValueError):
        tc.send_command("rm -rf")


def test_poller_returns_true_on_stop(ctl):
    poller = tc.ControlPoller(interval=0)
    assert poller.check() is False
    tc.send_command("stop")
    assert poller.check() is True


def test_poller_throttles_file_reads(ctl):
    poller = tc.ControlPoller(interval=60)
    assert poller.check() is False      # 第一次读文件
    tc.send_command("stop")
    assert poller.check() is False      # 60 秒内不再读


def test_poller_blocks_while_paused_until_resume(ctl):
    events = []
    poller = tc.ControlPoller(interval=0, pause_poll=0.01,
                              on_pause=lambda: events.append("paused"),
                              on_resume=lambda: events.append("resumed"))
    tc.send_command("pause")

    def resume_later():
        time.sleep(0.1)
        tc.send_command("resume")

    t = threading.Thread(target=resume_later)
    t.start()
    t0 = time.monotonic()
    assert poller.check() is False
    t.join()
    assert time.monotonic() - t0 >= 0.09, "暂停指令没有让训练循环停下来"
    assert events == ["paused", "resumed"]


def test_stop_while_paused_exits(ctl):
    poller = tc.ControlPoller(interval=0, pause_poll=0.01)
    tc.send_command("pause")
    threading.Timer(0.05, tc.send_command, args=("stop",)).start()
    assert poller.check() is True


def test_tail_log_keeps_last_tqdm_frame(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("start\n 10%|#\r 50%|###\r100%|#####\nend\n", encoding="utf-8")
    assert tc.tail_log(10, str(log)) == ["start", "100%|#####", "end"]
    assert tc.tail_log(1, str(log)) == ["end"]
    assert tc.tail_log(5, str(tmp_path / "missing.log")) == []


def _write_script(tmp_path, body):
    script = tmp_path / "fake_train.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(script)


def test_child_with_large_output_does_not_deadlock(ctl):
    """子进程输出远超管道缓冲区（64KB）也必须能跑完：输出进文件而不是没人读的 PIPE。"""
    script = _write_script(ctl, """
        import sys
        for i in range(20000):
            sys.stdout.write("x" * 50 + "\\n")
        print("DONE")
    """)
    mgr = tc.TrainingProcessManager()
    mgr.start(script=script)
    deadline = time.time() + 30
    while mgr.is_running() and time.time() < deadline:
        time.sleep(0.05)
    assert not mgr.is_running(), "子进程卡住了：输出很可能写进了没人读的管道"
    assert mgr.returncode() == 0
    assert tc.tail_log(1) == ["DONE"]
    assert not os.path.exists(tc.PID_PATH), "进程结束后 PID 文件应被清理"


def test_only_one_training_process_at_a_time(ctl):
    script = _write_script(ctl, "import time; time.sleep(30)\n")
    mgr = tc.TrainingProcessManager()
    mgr.start(script=script)
    try:
        assert mgr.is_running()
        with pytest.raises(RuntimeError):
            mgr.start(script=script)
    finally:
        assert mgr.terminate(timeout=5) is True
    assert not mgr.is_running()
    assert mgr.terminate() is False


def test_child_uses_current_interpreter(ctl):
    """原实现硬编码 'python'，在只有 python3 或用 venv 启动服务时会找错解释器。"""
    script = _write_script(ctl, "import sys; print(sys.executable)\n")
    mgr = tc.TrainingProcessManager()
    mgr.start(script=script)
    while mgr.is_running():
        time.sleep(0.05)
    assert os.path.realpath(tc.tail_log(1)[0]) == os.path.realpath(sys.executable)


def test_val_loss_is_mean_per_batch_not_sum():
    """原 eval.val_accuracy 返回所有 batch 损失之和，与 train_loss（均值）不可比。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from main import val_accuracy

    class Const(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 0.0]]).repeat(x.size(0), 1)

    x = torch.zeros(40, 5, dtype=torch.long)
    y = torch.ones(40, 1, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    acc, loss = val_accuracy(Const(), loader, "cpu", torch.nn.CrossEntropyLoss())
    assert acc == 0.0   # argmax 平局取 0，标签全是 1
    assert abs(loss - 0.6931) < 1e-3, f"应为单 batch 平均损失 ln2，实际 {loss}"
