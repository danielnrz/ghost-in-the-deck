"""Analytical tests for the deterministic PCM transition metrics."""

from __future__ import annotations

import math

import numpy as np

from ghost_in_the_deck.audio.metric_primitives import (
    energy_continuity,
    measure_peak_headroom,
    measure_transient_interference,
    rms_energy,
)


def test_peak_and_rms_are_analytical_and_finite():
    result = measure_peak_headroom(np.array([-0.25, 0.5, -0.75, 0.25]))
    assert result.peak == 0.75
    assert result.headroom == 0.25
    assert math.isfinite(result.headroom_db)
    assert rms_energy([1.0, -1.0, 1.0, -1.0]) == 1.0


def test_energy_continuity_reports_a_known_boundary_dip():
    result = energy_continuity([1.0, 1.0, 0.0, 1.0, 1.0], 2, window=2)
    assert result.before_rms == 1.0
    assert result.after_rms == math.sqrt(0.5)
    assert result.dip_rms == math.sqrt(0.75)
    assert result.dip_ratio == math.sqrt(0.75)
    assert result.boundary_jump == 1.0


def test_phase_interference_distinguishes_in_phase_and_cancellation():
    in_phase = measure_transient_interference([1.0, 1.0], [1.0, 1.0])
    opposite = measure_transient_interference([1.0, 1.0], [-1.0, -1.0])
    assert math.isclose(in_phase.correlation, 1.0)
    assert in_phase.sum_rms == 2.0
    assert in_phase.interference_ratio == 2.0
    assert math.isclose(opposite.correlation, -1.0)
    assert opposite.sum_rms == 0.0
    assert opposite.interference_ratio == 0.0
    assert opposite.cancellation == 1.0
