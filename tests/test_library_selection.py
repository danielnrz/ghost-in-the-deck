from pathlib import Path
import os
from unittest.mock import patch

import pytest

from ghost_in_the_deck.library import (LibraryTrack, cached_analysis, discover,
                                       load_library)
from ghost_in_the_deck.deck import TrackDeck
from ghost_in_the_deck.selection import rank_candidates
from synthetic import make_features, regular_beats


def track(name, bpm=120, duration=100):
    f = make_features(regular_beats(bpm=bpm, count=int(duration*bpm/60)-1), duration=duration)
    f.bpm = bpm
    f.track = name
    return LibraryTrack(Path(name), TrackDeck.from_features(f))


def test_discovery_nested_and_aliases(tmp_path):
    (tmp_path/'b').mkdir()
    source = tmp_path/'b'/'song.WAV'
    source.write_bytes(b'example')
    (tmp_path/'alias.wav').symlink_to(source)
    (tmp_path/'hard.wav').hardlink_to(source)
    (tmp_path/'x.txt').write_text('not audio')
    assert len(discover(tmp_path)) == 1
    assert discover(tmp_path) == discover(tmp_path)


def test_cache_freshness_corruption_and_same_basename(tmp_path):
    a = tmp_path/'a.wav'; a.write_bytes(b'first')
    folder = tmp_path/'b'; folder.mkdir()
    b = folder/'a.wav'; b.write_bytes(b'other')
    cache = tmp_path/'cache'
    with patch('ghost_in_the_deck.library.analyse', return_value=track('a').deck.features) as fn:
        first = cached_analysis(a, cache)
        assert cached_analysis(a, cache).to_dict() == first.to_dict()
        assert fn.call_count == 1
        stat = a.stat(); a.write_bytes(b'last!'); os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        cached_analysis(a, cache)
        cached_analysis(b, cache)
        assert fn.call_count == 3
        for entry in cache.glob('*.json'): entry.write_text('bad')
        cached_analysis(a, cache)
        assert fn.call_count == 4
    assert a.read_bytes() == b'last!'
    assert not list(cache.glob('.analysis-*'))


def test_failed_candidate_analysis_does_not_abort_library(tmp_path):
    for name in ('a.wav', 'b.wav'): (tmp_path/name).write_bytes(b'x')
    with patch('ghost_in_the_deck.library.analyse', side_effect=[ValueError('bad'), track('b').deck.features]):
        result = load_library(tmp_path, tmp_path/'cache')
    assert len(result.tracks) == len(result.errors) == 1


def test_empty_library(tmp_path):
    with pytest.raises(ValueError, match='No usable'):
        load_library(tmp_path)


def test_rank_ties_duplicates_and_dwell():
    a, b, c = track('a'), track('b'), track('c')
    result = rank_candidates(a, [c, b, a, b], 0)
    assert [x.track.path for x in result] == [b.path, c.path]
    assert result[0].plan.outgoing_time >= 65
    assert result[0].plan.incoming_time <= 35
    assert rank_candidates(a, [b], 85) == []
    assert rank_candidates(a, [], 0) == []
    assert rank_candidates(a, [a], 0) == []


@pytest.mark.parametrize('bpm', [50, 200, 0])
def test_incompatible_tempos(bpm):
    candidate = track('b', bpm=bpm or 120)
    candidate.deck.features.bpm = bpm
    assert rank_candidates(track('a'), [candidate], 0) == []


def test_short_tracks():
    assert rank_candidates(track('a', duration=4), [track('b')], 0) == []


@pytest.mark.parametrize('dwell', [-1, float('nan'), float('inf')])
def test_invalid_dwell(dwell):
    with pytest.raises(ValueError):
        rank_candidates(track('a'), [], 0, dwell_seconds=dwell)
