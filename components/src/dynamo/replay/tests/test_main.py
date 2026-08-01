# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.replay import ReplayReport
from dynamo.replay import main as replay_main

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def _stub_cli_dependencies(monkeypatch, seen: dict) -> None:
    monkeypatch.setattr(replay_main, "_load_engine_args", lambda value: value)
    monkeypatch.setattr(
        replay_main,
        "_load_router_config",
        lambda config, policy: (config, policy),
    )
    monkeypatch.setattr(replay_main, "_load_aic_perf_config", lambda args: None)
    monkeypatch.setattr(
        replay_main,
        "write_report_json",
        lambda payload, path: seen.setdefault("report_payload", payload) or path,
    )
    monkeypatch.setattr(replay_main, "format_report_table", lambda summary: "table")


def test_offline_cli_serializes_report_and_per_request_jsonl(
    monkeypatch, tmp_path
) -> None:
    seen: dict = {}
    _stub_cli_dependencies(monkeypatch, seen)

    def run_synthetic(*args, **kwargs):
        seen["native_args"] = args
        seen["native_kwargs"] = kwargs
        return ReplayReport(
            summary={"completed_requests": 1},
            per_request=[{"request_id": "request-1"}],
            coverage={"captured_request_count": 1},
            planner=None,
        )

    monkeypatch.setattr(replay_main, "run_synthetic_trace_replay", run_synthetic)
    per_request_path = tmp_path / "requests.jsonl"

    assert (
        replay_main.main(
            [
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--per-request-jsonl",
                str(per_request_path),
            ]
        )
        == 0
    )

    assert seen["native_kwargs"]["capture_per_request"] is True
    assert seen["report_payload"] == {
        "summary": {"completed_requests": 1},
        "per_request": [{"request_id": "request-1"}],
        "coverage": {"captured_request_count": 1},
        "planner": None,
    }
    assert json.loads(per_request_path.read_text()) == {"request_id": "request-1"}


def test_online_cli_keeps_flat_report(monkeypatch) -> None:
    seen: dict = {}
    _stub_cli_dependencies(monkeypatch, seen)

    def run_synthetic(*args, **kwargs):
        seen["native_kwargs"] = kwargs
        return {"completed_requests": 1}

    monkeypatch.setattr(replay_main, "run_synthetic_trace_replay", run_synthetic)

    assert (
        replay_main.main(
            [
                "--input-tokens",
                "8",
                "--output-tokens",
                "4",
                "--request-count",
                "1",
                "--replay-concurrency",
                "1",
                "--replay-mode",
                "online",
            ]
        )
        == 0
    )

    assert seen["native_kwargs"]["capture_per_request"] is False
    assert seen["report_payload"] == {"completed_requests": 1}
