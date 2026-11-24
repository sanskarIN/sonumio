"""GitHub Actions concurrency invariants for source, CLI and research gates.

GitHub concurrency groups allow only one *pending* member even when
cancel-in-progress is false. Main and dispatch therefore need unique groups;
PR revisions need a stable group to discard superseded runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"
CANCELLABLE_PR_GROUP = (
    "group: ${{ github.workflow }}-"
    "${{ github.event_name == 'pull_request' && "
    "github.event.pull_request.number || github.run_id }}"
)
PR_ONLY_CANCELLATION = (
    "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
)
FILES = (
    "ci.yml",
    "audio-core-ci.yml",
    "cli-wav-publication-ci.yml",
    "source-release-ci.yml",
    "research-ci.yml",
)


@pytest.mark.parametrize("filename", FILES)
def test_pending_main_and_manual_runs_are_not_evicted_by_a_shared_group(filename):
    """A PR has a stable key; non-PR executions use unique github.run_id."""
    text = (ROOT / filename).read_text(encoding="utf-8")
    assert text.count("\nconcurrency:\n") == 1
    assert text.count(CANCELLABLE_PR_GROUP) == 1
    assert text.count(PR_ONLY_CANCELLATION) == 1
    assert "github.event.pull_request.number || github.ref" not in text
    assert "github.event_name == 'pull_request' &&" in text
    assert "github.run_id" in text
    assert text.index("\nconcurrency:\n") < text.index("\njobs:\n")


def test_python_matrix_continues_even_if_one_version_fails():
    text = (ROOT / "ci.yml").read_text(encoding="utf-8")
    assert "fail-fast: false" in text
    for version in ("'3.11'", "'3.12'", "'3.13'"):
        assert version in text


def test_source_gate_is_still_manual_and_pr_triggered():
    text = (ROOT / "source-release-ci.yml").read_text(encoding="utf-8")
    before_jobs = text[:text.index("\njobs:\n")]
    assert "pull_request:" in before_jobs
    assert "workflow_dispatch:" in before_jobs
    assert "cancel-in-progress: true" not in before_jobs
