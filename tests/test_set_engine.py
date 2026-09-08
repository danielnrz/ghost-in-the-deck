from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from ghost_in_the_deck.library import Library, LibraryTrack, load_library
from ghost_in_the_deck.set_engine import CEILING, SAMPLE_RATE, SetEngine, load_pcm
from synthetic import make_beat_track
from test_library_selection import track


def mock_library(count=3, bpm=120, duration=100):
    return Library([track(str(i), bpm, duration) for i in range(count)], [])


def pcm(seconds=100):
    t = np.arange(round(SAMPLE_RATE*seconds))/SAMPLE_RATE
    return np.repeat((.2*np.sin(2*np.pi*220*t))[:, None], 2, axis=1)


def test_repeated_handoffs_and_exact_sample_ownership():
    source = pcm(); before = source.copy()
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: source.copy()):
        engine = SetEngine(mock_library())
        segments = list(engine.segments())
    assert np.array_equal(source, before)
    assert len(engine.handoffs) == 2
    assert all(h[3] == 'phase-aligned' for h in engine.handoffs)
    assert [str(s.span.active.path) for s in segments if s.span.plan is None] == ['0', '1', '2']
    for a, b in zip(segments, segments[1:]):
        assert a.span.end == b.span.start
        if a.span.plan:
            assert b.span.active == a.span.incoming
            expected = a.span.plan.incoming_time + a.span.plan.incoming_duration_seconds
            assert b.span.source_seconds(b.span.start) == pytest.approx(expected, abs=1/SAMPLE_RATE)
            assert b.span.deck != a.span.deck
    assert all(np.isfinite(s.samples).all() and np.max(np.abs(s.samples)) <= CEILING for s in segments)
    with pytest.raises(RuntimeError): list(engine.segments())


@pytest.mark.parametrize('count', [1, 2, 4])
def test_short_sequential_sets_and_end(count):
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: pcm(.2)):
        engine = SetEngine(mock_library(count, duration=.2))
        segments = list(engine.segments())
    assert len(engine.handoffs) == count-1
    assert len(segments) == count
    assert segments[-1].span.end == count*round(SAMPLE_RATE*.2)
    assert all(h[3] == 'sequential' for h in engine.handoffs)


def test_decoder_failure_skipped_and_transition_failure_falls_back():
    def decode(path):
        if str(path) == '1': raise ValueError('broken')
        return pcm(100)
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=decode), patch(
            'ghost_in_the_deck.set_engine.phase_correct_incoming_transition_with_relationship',
            side_effect=ValueError('unusable phase')):
        engine = SetEngine(mock_library())
        list(engine.segments())
    assert len(engine.handoffs) == 1
    assert str(engine.handoffs[0][2]) == '2'
    assert engine.errors


@pytest.mark.parametrize('rate,channels', [(8000, 1), (48000, 2), (44100, 2)])
def test_load_normalizes_and_preserves_source(tmp_path, rate, channels):
    path = tmp_path/'audio.wav'
    sf.write(path, np.full((rate, channels), 1.8), rate, subtype='FLOAT')
    before = path.read_bytes()
    samples = load_pcm(path)
    assert samples.shape == (44100, 2)
    assert np.isfinite(samples).all()
    assert np.max(np.abs(samples)) <= CEILING + 1e-12
    assert path.read_bytes() == before


def test_synthetic_library_to_two_real_transitions(tmp_path):
    sources = tmp_path/'music'; sources.mkdir()
    for i, bpm in enumerate((120, 118, 122)):
        make_beat_track(sources/f'{i}.wav', bpm=bpm, seconds=50)
    before = {p: p.read_bytes() for p in sources.iterdir()}
    library = load_library(sources, tmp_path/'analysis')
    engine = SetEngine(library, transition_bars=2, dwell_seconds=5)
    segments = list(engine.segments())
    assert len(engine.handoffs) == 2
    assert all(h[3] == 'phase-aligned' for h in engine.handoffs)
    assert [s.span.deck for s in segments if s.span.plan] == ['l', 'r']
    assert segments[-1].span.active.path not in [h[1] for h in engine.handoffs]
    assert all(np.max(np.abs(s.samples)) <= CEILING + 1e-12 for s in segments)
    assert all(p.read_bytes() == data for p, data in before.items())
    # Cached and fresh analysis must make the same decisions and exact PCM.
    repeated = SetEngine(load_library(sources, tmp_path/'analysis'), transition_bars=2, dwell_seconds=5)
    again = list(repeated.segments())
    assert engine.handoffs == repeated.handoffs
    for a, b in zip(segments, again):
        assert np.array_equal(a.samples, b.samples)
