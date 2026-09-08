"""Long causal timeline and production pose regressions; no private media."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ghost_in_the_deck.animation.visual_intent import (
    VisualTimeline, MIN_MAJOR_INTERVAL, REPEAT_INTERVAL, HANDOFF_COOLDOWN, other_deck,
)
from ghost_in_the_deck.dj_app import LiveDJApp
from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.live import StreamLedger, AudioChunk
from ghost_in_the_deck.set_engine import SetEngine, SAMPLE_RATE
from test_set_engine import mock_library, pcm


def fake_set(effects=True, count=5):
    with patch('ghost_in_the_deck.set_engine.load_pcm', side_effect=lambda _: pcm()):
        return list(SetEngine(mock_library(count), effects=effects).segments())


def app_for(spans):
    app = LiveDJApp.__new__(LiveDJApp)
    app.args = SimpleNamespace(no_actions=False)
    app.ledger = StreamLedger()
    for span in spans:
        app.ledger.append(AudioChunk(span,b'',span.end-span.start))
    app.visual = VisualTimeline()
    app.animator = AvatarAnimator(None)
    app._grooves = {}; app._broad_energy = {}
    return app


def timeline_record(spans):
    visual = VisualTimeline(); visual.update(spans)
    ledger = app_for(spans).ledger
    records = []
    for now in np.arange(0, spans[-1].end/SAMPLE_RATE, .1):
        span = ledger.at(float(now)); intent = visual.at(float(now),span.deck)
        action = intent.action_at(float(now))
        if action:
            assert intent.major and intent.operation in ('crossfade','filter_sweep','gain_riser')
            assert action.side == intent.deck
            if intent.operation == 'crossfade':
                assert any(s.plan and other_deck(s.deck)==intent.deck and
                           s.start/SAMPLE_RATE == intent.operation_start for s in spans)
            else:
                assert span.plan is None and span.deck == intent.deck
                assert any(a.action == intent.operation and
                           a.start+(span.start-span.source_start)/SAMPLE_RATE == intent.operation_start
                           for a in span.audio_actions)
        records.append((round(float(now),3), 'mix' if span.plan else 'solo', span.deck,
                        intent.operation, intent.kind, intent.phase_at(now) if action else 'monitor/groove',
                        action.side if action else None))
    return records, visual


def test_long_set_causality_restraint_and_repeatability(tmp_path):
    segments = fake_set(); spans = [s.span for s in segments]
    records, visual = timeline_record(spans)
    again, _ = timeline_record(spans)
    assert records == again
    # Five tracks, four real handoffs: A -> B -> A -> B -> A.
    assert [s.deck for s in spans if s.plan is None] == ['l','r','l','r','l']
    assert len([a for a in visual.interactions if a.operation=='crossfade']) == 4
    assert sum(r[4]=='IDLE_GROOVE' for r in records) > len(records)*.7
    assert any(a.operation=='filter_sweep' for a in visual.interactions)
    for a,b in zip(visual.interactions,visual.interactions[1:]):
        assert a.end <= b.begin
        assert b.begin-a.begin >= MIN_MAJOR_INTERVAL
        if a.kind == b.kind and a.operation != 'crossfade':
            assert b.begin-a.begin >= REPEAT_INTERVAL
    for intent in visual.interactions:
        if intent.operation != 'crossfade':
            assert all(not (s.end/SAMPLE_RATE <= intent.begin < s.end/SAMPLE_RATE+HANDOFF_COOLDOWN)
                       for s in spans if s.plan)
    import json
    (tmp_path/'behavior-timeline.json').write_text(json.dumps(records))


def test_effects_disabled_never_touches_solo_and_gain_uses_owned_deck():
    spans = [s.span for s in fake_set(False,3)]
    visual = VisualTimeline(); visual.update(spans)
    assert all(not s.audio_actions for s in spans)
    assert all(i.operation=='crossfade' for i in visual.interactions)
    from ghost_in_the_deck.dj_planner import PlannedAudioAction
    solo = spans[-1]
    action = PlannedAudioAction(solo.source_start/SAMPLE_RATE+15, .7, 'gain_riser', None, .9)
    visual.update([replace(solo,audio_actions=(action,))])
    assert len(visual.interactions)==1
    assert visual.interactions[0].kind=='GAIN_ADJUST'
    assert visual.interactions[0].deck==solo.deck


def test_incremental_commit_cannot_cancel_a_started_solo_reach():
    spans = [s.span for s in fake_set()]
    for length in range(1,len(spans)):
        early=VisualTimeline(); early.update(spans[:length])
        late=VisualTimeline(); late.update(spans[:length+1])
        for intent in early.interactions:
            assert intent in late.interactions


def test_stalls_and_handoff_pose_continuity():
    spans=[s.span for s in fake_set(False,3)]; app=app_for(spans)
    app.visual.update(spans)
    boundaries = [t for i in app.visual.interactions for t in
                  (i.begin,i.operation_start,i.contact_end or i.operation_end,i.operation_end,i.end)]
    boundaries += [s.end/SAMPLE_RATE for s in spans if s.plan]
    for t in boundaries:
        before,_,_=app.pose_at(t-.0001); after,_,_=app.pose_at(t+.0001)
        worst=max(abs(a-b) for j in before.keys()|after.keys()
                  for a,b in zip(before.get(j,(0,0,0)),after.get(j,(0,0,0))))
        assert worst < .05, (t,worst)
        assert app.pose_at(t)==app.pose_at(t)  # absolute time; no interpolation cursor
    end=spans[-1].start/SAMPLE_RATE+3
    expected=app.pose_at(end)
    app.pose_at(1); app.pose_at(50)
    assert app.pose_at(end)==expected


def test_conservative_energy_response_without_deck_actions():
    from synthetic import groove_for, regular_beats
    spans=[s.span for s in fake_set(False,1)]; app=app_for(spans)
    track=spans[0].active
    groove=groove_for(regular_beats(count=200),duration=100,
                     energy=lambda t: .1 if t < 50 else .95)
    app._grooves[track.path]=groove
    from ghost_in_the_deck.animation.energy import EnergyTrack
    app._broad_energy[track.path]=EnergyTrack(groove.features,smoothing_seconds=8)
    levels=[app._groove_state(track,t).intensity for t in (20,70)]
    assert app.pose_at(20)[2] is None and app.pose_at(70)[2] is None
    assert .1 <= levels[0] < levels[1] <= .62


def test_dense_transition_policy_monitors_instead_of_conflicting():
    spans=[s.span for s in fake_set(False,2) if s.span.plan]
    a=spans[0]
    b=replace(a,start=a.start+5*SAMPLE_RATE,end=a.end+5*SAMPLE_RATE,deck='r')
    visual=VisualTimeline(); visual.update([a,b])
    assert len(visual.interactions)==1


def test_production_contact_clearance_and_frame_continuity():
    import math
    import panda_env
    from reach_clearance import ClearanceHarness, SAFETY_MARGIN
    if not panda_env.has_window():
        pytest.skip('requires rendered rig')
    spans=[s.span for s in fake_set(True,3)]
    app=app_for(spans); app.visual.update(spans)
    harness=ClearanceHarness(); app.animator=AvatarAnimator(harness.rig)
    worst_margin=math.inf
    try:
        for intent in app.visual.interactions:
            last=None
            for now in np.arange(intent.begin,intent.end+1/60,1/60):
                offsets,_,action=app.pose_at(float(now))
                harness.rig.reset()
                for name,(h,p,r) in offsets.items(): harness.rig.set_offset(name,h,p,r)
                harness.rig.force_update()
                margin,_,_=harness._worst_mesh(intent.deck,harness.furniture,math.inf)
                worst_margin=min(worst_margin,margin)
                assert margin > SAFETY_MARGIN, (intent.kind,intent.deck,now,margin)
                if last:
                    maximum=max(abs(a-b) for j in last.keys()|offsets.keys()
                                for a,b in zip(last.get(j,(0,0,0)),offsets.get(j,(0,0,0))))
                    assert maximum < 8, (now,maximum)  # catch frame-scale joint snaps
                last=offsets
            offsets,_,_=app.pose_at(intent.operation_start+.1)
            harness.rig.reset()
            for name,(h,p,r) in offsets.items(): harness.rig.set_offset(name,h,p,r)
            harness.rig.force_update()
            finger=harness.rig.expose(f'middle_03_{intent.deck}').getPos(harness.base.render)
            target_x=(.204 if intent.deck=='l' else -.204)-.0288
            # Each physical panel's knob is offset the same way in world X,
            # not mirrored. Contact geometry must use that actual scene detail.
            assert abs(finger.x-target_x) < .003
            assert abs(finger.z-1.033) < .003
            assert abs(finger.y+.378) < .003
    finally:
        harness.rig.actor.cleanup(); harness.rig.actor.removeNode(); harness.workstation.removeNode()


def test_short_dwell_respects_handoff_cooldown_and_blends_attention():
    a=[s.span for s in fake_set(False,2) if s.span.plan][0]
    # Begins twelve seconds after the prior start: meets start spacing, but
    # its anticipation is inside the prior handoff cooldown and must be skipped.
    b=replace(a,start=a.start+12*SAMPLE_RATE,end=a.end+12*SAMPLE_RATE,deck='r')
    visual=VisualTimeline(); visual.update([a,b])
    assert len(visual.interactions)==1
    b=replace(a,start=a.end+SAMPLE_RATE,end=a.end+9*SAMPLE_RATE,deck='r')
    visual.update([a,b])
    last=None
    for now in np.arange(a.end/SAMPLE_RATE-3,b.start/SAMPLE_RATE+4,1/60):
        side,weight,nod=visual.attention_at(now)
        turn=weight*(1 if side=='l' else -1)
        if last is not None:
            assert abs(turn-last)<.05
        last=turn
