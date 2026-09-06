# -*- coding: utf-8 -*-
"""路径必须与工作目录无关。

main.py 曾把作者本机的 /Users/joey/PycharmProjects/shebi/config/progress.json
写死在源码里，换任何机器运行训练都会抛 FileNotFoundError。
"""

import os
import pathlib
import subprocess

from config import Config

REPO = pathlib.Path(Config.base_dir)


def test_all_config_paths_are_absolute():
    for name in ("model_dir", "stopword_path", "train_path", "val_path", "test_path",
                 "word2id_path", "pre_word2vec_path", "progress_path", "params_path"):
        value = getattr(Config, name)
        assert os.path.isabs(value), f"Config.{name} 不是绝对路径: {value}"


def test_config_paths_live_inside_the_repo():
    for name in ("model_dir", "train_path", "progress_path", "params_path"):
        value = pathlib.Path(getattr(Config, name)).resolve()
        assert REPO.resolve() in value.parents or value.parent == REPO.resolve() \
            or str(value).startswith(str(REPO.resolve())), f"Config.{name} 指向仓库之外: {value}"


def test_no_developer_machine_paths_in_source():
    """代码里的字符串字面量不允许出现开发者本机绝对路径。

    只检查真正参与执行的字符串常量，文档字符串与注释里提及历史路径是允许的
    （config.py 的模块说明就记录了这段历史）。
    """
    import ast

    hits = []
    for py in sorted(REPO.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                if "/Users/" in node.value or "C:\\Users" in node.value:
                    hits.append(f"{py.name}:{node.lineno}: {node.value[:80]}")

    assert not hits, "源码中残留开发者本机路径:\n" + "\n".join(hits)


def test_data_files_resolve_from_any_cwd():
    """从系统临时目录启动也必须能定位到数据文件。"""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from config import Config; import os;"
        "print(os.path.exists(Config.train_path), os.path.exists(Config.stopword_path))"
        % str(REPO)
    )
    out = subprocess.run([os.sys.executable, "-c", code], cwd="/tmp",
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "True True", out.stderr


def test_runtime_dir_is_created():
    assert os.path.isdir(Config.runtime_dir)


def test_default_model_is_a_valid_choice():
    assert Config.default_model in Config.model_choices
