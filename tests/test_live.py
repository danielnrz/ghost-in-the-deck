import queue
import time
from unittest.mock import patch

import numpy as np
import pytest

from ghost_in_the_deck.dj_app import transition_action
from ghost_in_the_deck.live import AudioChunk, SetBuffer, StreamLedger
from ghost_in_the_deck.set_engine import DeckSpan, Segment, SetEngine, SAMPLE_RATE
from test_set_engine import mock_library, pcm


def test_ledger_skipped_frames_and_repeated_handoffs():
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: pcm()):
        segments = list(SetEngine(mock_library()).segments())
    ledger = StreamLedger()
    for segment in segments:
        ledger.append(AudioChunk(segment.span, b'', len(segment.samples)))
    for segment in segments:
        assert ledger.at(segment.span.start/SAMPLE_RATE) is segment.span
        if segment.span.plan:
            action = transition_action(segment.span, (segment.span.start+segment.span.end)/2/SAMPLE_RATE)
            assert action.is_active
            assert action.side != segment.span.deck
            assert transition_action(segment.span, (segment.span.end+1)/SAMPLE_RATE) is None
    last = segments[-1].span
    # A renderer jumping past BOTH handoffs immediately gets the right owner.
    assert ledger.at(last.start/SAMPLE_RATE + 1).active == last.active
    assert ledger.at(last.start/SAMPLE_RATE + 1).source_seconds(last.start + SAMPLE_RATE) == pytest.approx(last.source_start/SAMPLE_RATE+1)


def test_buffer_pcm_and_clean_shutdown():
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: pcm(.2)):
        buffer = SetBuffer(SetEngine(mock_library(3, duration=.2)))
        chunks = []
        while not buffer.finished.is_set() or not buffer.queue.empty():
            try: chunks.append(buffer.queue.get(timeout=.1))
            except queue.Empty: pass
        buffer.close()
    assert not buffer.thread.is_alive()
    assert buffer.error is None
    assert len(chunks) == 3
    assert all(len(c.data) == c.frames*4 for c in chunks)
    for chunk in chunks:
        assert np.max(np.abs(np.frombuffer(chunk.data, dtype='<i2'))) < 32767


def test_cancel_full_queue():
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: pcm(20)):
        buffer = SetBuffer(SetEngine(mock_library(1, duration=20)))
        deadline = time.monotonic()+3
        while not buffer.queue.full() and time.monotonic() < deadline:
            time.sleep(.01)
        buffer.close()
    assert not buffer.thread.is_alive()


def test_producer_error_is_reported():
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=ValueError('bad PCM')):
        buffer = SetBuffer(SetEngine(mock_library(1)))
        assert buffer.finished.wait(3)
        buffer.close()
    assert 'None of the analyzed' in buffer.error
    assert buffer.queue.empty()


def test_transition_pose_clearance_on_both_decks():
    import math
    import panda_env
    from ghost_in_the_deck.animation.controller import AvatarAnimator
    from ghost_in_the_deck.dj_app import write_set_pose
    from ghost_in_the_deck.selection import rank_candidates
    from reach_clearance import ClearanceHarness
    from synthetic import groove_for, regular_beats
    from test_library_selection import track
    if not panda_env.has_window():
        pytest.skip('requires rendered rig')
    harness = ClearanceHarness()
    try:
        a, b = track('a'), track('b')
        plan = rank_candidates(a, [b], 0)[0].plan
        groove = groove_for(regular_beats(count=200), duration=100, energy=.8)
        animator = AvatarAnimator(harness.rig, groove)
        for side in ('l', 'r'):
            span = DeckSpan(0, 8*SAMPLE_RATE, a, 0, side, b, plan)
            hand = 'r' if side == 'l' else 'l'
            for t in np.linspace(0, 8, 101):
                action = transition_action(span, t)
                write_set_pose(animator, groove.state_at(t+30), action, transition=True)
                margin, joint, node = harness._worst_mesh(hand, harness.furniture, math.inf)
                assert margin > 0, (side, t, joint, node, margin)
    finally:
        harness.rig.actor.cleanup()
        harness.rig.actor.removeNode()
        harness.workstation.removeNode()
