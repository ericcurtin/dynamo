# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in media-decoder runtime installer.

The pip install is fully mocked -- these tests never touch the network or the
real site-packages.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

import pytest

from dynamo.common.utils import media_decoders

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def clean_env(monkeypatch):
    """Reset the three env vars, the process guard, and stub out the lock."""
    for key in (
        media_decoders.ENABLE_ENV,
        media_decoders.PACKAGES_ENV,
        media_decoders.PIP_ARGS_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    media_decoders._completed.clear()
    # Never take a real file lock or rewrite import caches in unit tests.
    monkeypatch.setattr(
        media_decoders, "_cross_process_lock", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr(media_decoders.importlib, "invalidate_caches", lambda: None)
    yield monkeypatch
    media_decoders._completed.clear()


def _record_pip(monkeypatch, *, fail: bool = False) -> list[list[str]]:
    """Replace subprocess.run with a recorder; return the list of commands."""
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))
        if fail:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media_decoders.subprocess, "run", fake_run)
    return calls


def _set_available(monkeypatch, present) -> None:
    """Stub _module_available; `present` is a set of importable module names."""
    monkeypatch.setattr(media_decoders, "_module_available", lambda mod: mod in present)


def test_disabled_by_default_is_noop(clean_env):
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())  # nothing installed
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []


def test_enabled_but_already_present_skips_install(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, {"cv2", "av"})
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []


def test_vllm_installs_video_and_audio_carriers(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "true")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "--break-system-packages" in cmd
    assert "opencv-python-headless" in cmd  # video carrier
    assert "av" in cmd  # audio (AAC) carrier
    # Never installed: pynvvideocodec because the image already ships it as the
    # NVDEC path, and the rest because no vLLM decode path imports them.
    for banned in ("torchcodec", "pynvvideocodec", "decord2", "libx264"):
        assert banned not in cmd


def test_sglang_installs_only_decord(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "yes")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert len(calls) == 1
    cmd = calls[0]
    assert "decord2" in cmd
    assert "opencv-python-headless" not in cmd
    assert "av" not in cmd


def test_installs_only_missing_modules(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, {"cv2"})  # cv2 present, av missing
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "av" in cmd
    assert "opencv-python-headless" not in cmd


def test_package_override_installs_verbatim(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(
        media_decoders.PACKAGES_ENV, "opencv-python-headless==4.10.0.84 av==12.0.0"
    )
    calls = _record_pip(clean_env)
    # Override bypasses the already-present short-circuit.
    _set_available(clean_env, {"cv2", "av"})
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "opencv-python-headless==4.10.0.84" in cmd
    assert "av==12.0.0" in cmd


def test_extra_pip_args_are_appended(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(
        media_decoders.PIP_ARGS_ENV, "--index-url https://mirror/simple --no-deps"
    )
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    cmd = calls[0]
    assert "--index-url" in cmd
    assert "https://mirror/simple" in cmd
    assert "--no-deps" in cmd


def test_install_failure_does_not_raise(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env, fail=True)
    _set_available(clean_env, set())
    # Must not propagate -- worker startup continues.
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1  # install was attempted


def test_trtllm_installs_opencv_only(clean_env):
    # TRT-LLM decodes video_url via cv2 (tensorrt_llm async_load_video); it has
    # no audio-input path, so only the video carrier is installed.
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("trtllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "opencv-python-headless" in cmd
    assert "av" not in cmd
    assert "decord2" not in cmd


def test_unknown_backend_is_noop(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("mystery")
    assert calls == []


def test_idempotent_second_call_does_not_reinstall(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    media_decoders.maybe_install_media_decoders("sglang")
    assert len(calls) == 1  # second call short-circuits via the process guard


def test_malformed_pip_args_does_not_raise(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.PIP_ARGS_ENV, "--find-links '/opt/un balanced")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    # Unbalanced quotes must be caught, not crash worker startup.
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []  # never reached the install


def test_failure_does_not_mark_complete_and_retries(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env, fail=True)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert "vllm" not in media_decoders._completed  # a failure is retryable
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 2  # retried on the next call


def test_default_install_uses_no_deps(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert "--no-deps" in calls[0]  # default install must not perturb the stack


def test_override_install_keeps_deps(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.PACKAGES_ENV, "av==12.0.0")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert "av==12.0.0" in calls[0]
    assert "--no-deps" not in calls[0]  # operator-chosen specs resolve deps


def test_pip_install_is_bounded_by_timeout(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    seen: dict = {}

    def fake_run(cmd, check=False, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    clean_env.setattr(media_decoders.subprocess, "run", fake_run)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert seen["kwargs"].get("timeout") == 600  # default bound


def test_custom_timeout_override(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.TIMEOUT_ENV, "42")
    seen: dict = {}

    def fake_run(cmd, check=False, **kwargs):
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    clean_env.setattr(media_decoders.subprocess, "run", fake_run)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert seen["kwargs"].get("timeout") == 42


def test_redact_masks_url_credentials():
    red = media_decoders._redact(
        "pip install --index-url https://user:tok@mirror/simple decord2"
    )
    assert "user:tok" not in red
    assert "https://***@mirror/simple" in red


def test_pending_subset_installs_only_still_missing(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    # vLLM decoders are (cv2, av). First scan: both missing. Under the lock a
    # peer has meanwhile installed cv2, so only `av` should reach pip.
    seen = {"n": 0}

    def avail(mod):
        seen["n"] += 1
        # Calls 1-2 are the initial missing-scan (both absent). Calls 3+ are the
        # under-lock re-check and post-verify; cv2 now reads as present.
        return seen["n"] > 2 and mod == "cv2"

    clean_env.setattr(media_decoders, "_module_available", avail)
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    assert "av" in calls[0]
    assert "opencv-python-headless" not in calls[0]


# --- Entrypoint coverage -----------------------------------------------------
#
# The install hook lives in each worker's __main__.py, which means every new
# entrypoint has to remember it. `python -m dynamo.vllm.omni` did not: `-m
# pkg.sub` executes pkg/sub/__main__.py and never pkg/__main__.py, so the call in
# vllm/__main__.py did not cover omni and DYN_ENABLE_MEDIA_DECODERS was silently
# inert there -- the switch appeared to be set and installed nothing.
#
# Reported by Harrison Saturley-Hall on #12051.
#
# This turns the next such omission into a failing test. Both sets are explicit,
# so a newly added entrypoint belongs to neither and fails here until somebody
# decides which it is. That is the point: the failure asks a question rather
# than guessing an answer.

# Entrypoints that decode media on the worker and therefore need the hook.
_ENTRYPOINTS_NEEDING_DECODERS = {
    "sglang/__main__.py",
    "trtllm/__main__.py",
    "vllm/__main__.py",
    "vllm/omni/__main__.py",
}

# Entrypoints that do not decode media: control-plane components, and the
# frontend, whose optional Rust decoder links FFmpeg's compiled-in decoders and
# so is not extended by a runtime pip install.
_ENTRYPOINTS_WITHOUT_DECODERS = {
    "frontend/__main__.py",
    "global_planner/__main__.py",
    "global_router/__main__.py",
    "kv_dc_relay/__main__.py",
    "mocker/__main__.py",
    "planner/__main__.py",
    "profiler/__main__.py",
    "replay/__main__.py",
    "router/__main__.py",
    "squeeze_evolve/__main__.py",
    "thunderagent_router/__main__.py",
    "tokenspeed/__main__.py",
}


def _entrypoints() -> dict[str, str]:
    """Every `__main__.py` under components/src/dynamo, keyed by relative path."""
    root = Path(media_decoders.__file__).resolve().parents[2]
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("__main__.py"))
    }


def test_every_entrypoint_is_classified():
    """A new entrypoint must be declared as needing decoders or not."""
    found = set(_entrypoints())
    classified = _ENTRYPOINTS_NEEDING_DECODERS | _ENTRYPOINTS_WITHOUT_DECODERS
    unclassified = found - classified
    assert not unclassified, (
        f"new entrypoint(s) {sorted(unclassified)} are not classified. Add each to "
        "_ENTRYPOINTS_NEEDING_DECODERS (and call maybe_install_media_decoders in "
        "its __main__.py) or to _ENTRYPOINTS_WITHOUT_DECODERS."
    )
    # Paired the other way: a removed/renamed entrypoint should not linger here.
    stale = classified - found
    assert not stale, f"classified entrypoint(s) no longer exist: {sorted(stale)}"


def test_media_entrypoints_install_decoders():
    """Every entrypoint that decodes media calls the installer."""
    sources = _entrypoints()
    missing = [
        name
        for name in sorted(_ENTRYPOINTS_NEEDING_DECODERS)
        if "maybe_install_media_decoders" not in sources.get(name, "")
    ]
    assert not missing, (
        f"{missing} decode media but never call maybe_install_media_decoders, so "
        "DYN_ENABLE_MEDIA_DECODERS is silently inert for them"
    )


def test_non_media_entrypoints_do_not_install_decoders():
    """Sanity for the check above: it is not simply true of every file.

    Without this, adding the call everywhere would satisfy the previous test
    while saying nothing about whether it lands where it matters.
    """
    sources = _entrypoints()
    unexpected = [
        name
        for name in sorted(_ENTRYPOINTS_WITHOUT_DECODERS)
        if "maybe_install_media_decoders" in sources.get(name, "")
    ]
    assert not unexpected, (
        f"{unexpected} are declared as not decoding media but call the installer; "
        "move them to _ENTRYPOINTS_NEEDING_DECODERS or drop the call"
    )
