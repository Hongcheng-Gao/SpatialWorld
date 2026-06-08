"""Utilities for launching and resolving the VirtualHome Unity backend."""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_BACKEND_EXE = os.environ.get(
    "VIRTUALHOME_BACKEND_EXE",
    r"E:\科研内容\windows_exec\windows_exec.v2.2.4\VirtualHome.exe",
)
DEFAULT_BACKEND_ARGS = os.environ.get(
    "VIRTUALHOME_BACKEND_ARGS",
    "-windowed -screen-width 960 -screen-height 540",
)
DEFAULT_STARTUP_TIMEOUT = int(os.environ.get("VIRTUALHOME_BACKEND_STARTUP_TIMEOUT", "90"))
DEFAULT_BACKEND_HOST = os.environ.get("VIRTUALHOME_BACKEND_HOST", "127.0.0.1")
DEFAULT_BACKEND_PORT = int(os.environ.get("VIRTUALHOME_BACKEND_PORT", "8080"))


def _env_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    env = config.get("env")
    return env if isinstance(env, Mapping) else {}


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def resolve_backend_exe(
    config: Mapping[str, Any] | None = None,
    override: str | None = None,
) -> str:
    env_cfg = _env_config(config)
    return str(_coalesce(override, env_cfg.get("backend_exe"), DEFAULT_BACKEND_EXE))


def resolve_backend_args(
    config: Mapping[str, Any] | None = None,
    override: str | None = None,
) -> str:
    env_cfg = _env_config(config)
    return str(_coalesce(override, env_cfg.get("backend_args"), DEFAULT_BACKEND_ARGS))


def resolve_backend_startup_timeout(
    config: Mapping[str, Any] | None = None,
    override: int | str | None = None,
) -> int:
    env_cfg = _env_config(config)
    return int(_coalesce(override, env_cfg.get("backend_startup_timeout"), DEFAULT_STARTUP_TIMEOUT))


def resolve_backend_host(
    config: Mapping[str, Any] | None = None,
    override: str | None = None,
) -> str:
    env_cfg = _env_config(config)
    return str(_coalesce(override, env_cfg.get("host"), env_cfg.get("url"), DEFAULT_BACKEND_HOST))


def resolve_backend_port(
    config: Mapping[str, Any] | None = None,
    override: int | str | None = None,
) -> int:
    env_cfg = _env_config(config)
    return int(_coalesce(override, env_cfg.get("port"), DEFAULT_BACKEND_PORT))


def build_backend_command(
    backend_exe: str,
    backend_args_str: str,
    *,
    port: int,
) -> tuple[list[str], Path]:
    exe_path = Path(backend_exe).expanduser()
    args = shlex.split(backend_args_str or "", posix=False)
    return [str(exe_path), *args, "-http-port", str(port)], exe_path


def is_port_open(host: str, port: int, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    if timeout_sec <= 0:
        return True
    start = time.time()
    while time.time() - start < timeout_sec:
        if is_port_open(host, port):
            return True
        time.sleep(1.0)
    return False


def launch_backend(
    backend_exe: str,
    backend_args_str: str,
    *,
    port: int,
):
    cmd, exe_path = build_backend_command(backend_exe, backend_args_str, port=port)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(exe_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    print(f"Started VirtualHome backend (PID={proc.pid}): {exe_path.name}")
    return proc


def terminate_backend(proc, timeout_sec: int = 20) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
