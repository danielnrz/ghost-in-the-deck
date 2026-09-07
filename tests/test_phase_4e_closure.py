"""Phase 4E policy, boundary, and independent-review regressions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.audio.beat_phase import (
    SampleSpan,
    TransformedSampleMapping,
    build_beat_phase_contract,
)
from ghost_in_the_deck.audio.phase_corrected_material import (
    MAX_INITIAL_CUE_CORRECTION_SAMPLES,
)

from synthetic import make_features


ROOT = Path(__file__).resolve().parents[1]


def _timeline(beats: list[float]) -> BeatTimeline:
    return BeatTimeline(make_features(beats, duration=8.0, bpm=120.0))


@pytest.mark.parametrize(
    ("outgoing_time", "incoming_time", "expected_correction"),
    [(1.24, 2.36, 1), (1.36, 2.24, -1), (1.24, 2.24, 0)],
)
def test_phase_4e_is_nearest_integer_and_one_sample_bounded(
    outgoing_time: float, incoming_time: float, expected_correction: int
):
    result = build_beat_phase_contract(
        _timeline([outgoing_time]),
        _timeline([incoming_time]),
        outgoing_cue_index=0,
        incoming_cue_index=0,
        sample_rate=10,
        source_sample_count=10,
        transformed_sample_count=12,
    )

    assert result.initial_correction_samples == expected_correction
    assert (
        abs(result.initial_correction_samples)
        <= MAX_INITIAL_CUE_CORRECTION_SAMPLES
    )
    assert abs(result.residual_offset_samples) <= 0.5


def test_phase_4e_modules_remain_offline_and_planner_independent():
    material_source = (
        ROOT / "src/ghost_in_the_deck/audio/phase_corrected_material.py"
    ).read_text()
    preview_source = (ROOT / "src/ghost_in_the_deck/transition_preview.py").read_text()

    for source in (material_source, preview_source):
        assert "panda3d" not in source.lower()
        assert "dj_planner" not in source
        assert "dj_behavior" not in source
    assert "execute_transition" in preview_source


def test_git_guard_passes_for_the_checked_out_branch():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/git_guard.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Git Guard: PASS" in result.stdout


def test_private_audio_and_evaluation_outputs_are_ignored():
    for path in ("testMusic/private.wav", "out/evaluation/probe.wav"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            cwd=ROOT,
        )
        assert result.returncode == 0, path

    tracked = subprocess.run(
        ["git", "ls-files", "--", "testMusic", "out"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert tracked in ([], ["testMusic/.gitkeep"])


def test_mapping_contract_is_half_open_at_both_boundaries():
    mapping = TransformedSampleMapping(
        source_span=SampleSpan(10, 20),
        transformed_span=SampleSpan(30, 42),
    )
    assert mapping.map_source_span(SampleSpan(10, 20)) == SampleSpan(30, 42)
    with pytest.raises(ValueError, match="outside"):
        mapping.map_source_span(SampleSpan(9, 20))
