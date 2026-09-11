"""Performance accents and physical target variants without private media."""
from dataclasses import replace
import math

import numpy as np
import pytest

from ghost_in_the_deck.animation.dj_behavior import DJActionState
from ghost_in_the_deck.animation.performance_pose import set_pose_offsets, hype_lift
from ghost_in_the_deck.animation.rig import PerformanceRig
from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.visual_intent import VisualTimeline
from ghost_in_the_deck.set_engine import DeckSpan,SAMPLE_RATE
from ghost_in_the_deck.deck import TrackDeck
from synthetic import make_features,regular_beats,groove_for
from test_visual_intent import fake_set


def test_variants_follow_real_operation_and_tempo_policy():
    spans=[s.span for s in fake_set(True,3)]
    visual=VisualTimeline();visual.update(spans)
    assert all(i.variant=='knob' for i in visual.interactions if i.operation!='crossfade')
    assert all(i.variant=='button' for i in visual.interactions if i.operation=='crossfade')
    mixes=[s for s in spans if s.plan]
    changed=[replace(s,plan=replace(s.plan,bpm_b=s.plan.bpm_a/1.1)) for s in mixes]
    visual.update(changed)
    assert [i.variant for i in visual.interactions]==['platter','platter']
    assert [i.deck for i in visual.interactions]==['r','l']
    assert all(i.contact_end==i.operation_start and i.contact_start<i.contact_end for i in visual.interactions)


def hype_spans(build=True):
    spans=[]
    for i in range(4):
        features=make_features(regular_beats(count=240),duration=120,
            energy=(lambda t:min(.95,.1+max(0,t-20)*.06)) if build else .7)
        # Use the production deck analysis wrappers, with known measured input.
        from test_library_selection import track
        original=track(str(i),duration=120)
        deck=TrackDeck.from_features(features)
        entry=replace(original,deck=deck)
        spans.append(DeckSpan(i*120*SAMPLE_RATE,(i+1)*120*SAMPLE_RATE,entry,0,'l' if i%2==0 else 'r'))
    return spans


def test_hype_requires_build_is_rare_and_deterministic():
    visual=VisualTimeline();visual.update(hype_spans(False));assert not visual.hypes
    spans=hype_spans();visual.update(spans)
    assert len(visual.hypes)==2
    assert visual.hypes[1].begin-visual.hypes[0].begin>=240
    assert {int(i.begin//120) for i in visual.hypes}=={0,2}
    again=VisualTimeline();again.update(spans);assert visual.hypes==again.hypes
    for intent in visual.hypes:
        assert not intent.major and intent.operation=='measured_energy_lift'
        action=intent.action_at(intent.begin+1)
        assert action.action=='small_hype'
        assert hype_lift(action)>0
        assert hype_lift(intent.action_at(intent.end))==0
    for n in range(1,len(spans)):
        early=VisualTimeline();early.update(spans[:n])
        assert all(i in visual.hypes for i in early.hypes)


def test_contact_is_not_a_frozen_pose_and_body_is_decomposed():
    animator=AvatarAnimator(None);state=groove_for().state_at(10)
    for side in ('l','r'):
        for variant in ('knob','button','platter'):
            start=DJActionState(10,'hand_to_deck',.5,1,side,1,variant,0)
            moving=replace(start,contact_phase=.5)
            assert set_pose_offsets(animator,state,start)!=set_pose_offsets(animator,state,moving)
    a=set_pose_offsets(animator,state,None)
    b=set_pose_offsets(animator,replace(state,beat_phase=state.beat_phase+.12),None)
    deltas=[b[j][1]-a[j][1] for j in ('head','neck_01','spine_03','clavicle_l')]
    assert len({round(v,4) for v in deltas})==4


@pytest.mark.parametrize('side',['l','r'])
@pytest.mark.parametrize('variant',['knob','button','platter','cheer'])
def test_both_hands_clear_equipment_through_all_variants(side,variant):
    import panda_env
    from reach_clearance import ClearanceHarness,SAFETY_MARGIN
    if not panda_env.has_window():pytest.skip('needs rendered rig')
    harness=ClearanceHarness(rig_class=PerformanceRig)
    animator=AvatarAnimator(harness.rig);groove=groove_for()
    try:
        for progress in np.linspace(0,1,181):
            kind='small_hype' if variant=='cheer' else 'hand_to_deck'
            from ghost_in_the_deck.animation.dj_behavior import _envelope_weight
            action=DJActionState(10,kind,progress,_envelope_weight(kind,progress),side,1,variant,.5)
            offsets=set_pose_offsets(animator,groove.state_at(10+progress),action)
            harness.rig.reset();harness.rig.actor.setZ(hype_lift(action))
            for joint,hpr in offsets.items():
                assert all(abs(v)<=limit+1e-6 for v,limit in zip(hpr,harness.rig.limit_for(joint)))
                harness.rig.set_offset(joint,*hpr)
            for hand in ('l','r'):
                margin,joint,node=harness._worst_mesh(hand,harness.furniture,math.inf)
                assert margin>SAFETY_MARGIN,(variant,side,hand,progress,margin,joint,node)
            if progress == .5 and variant != 'cheer':
                # Physical scene coordinates, including asymmetric panel controls.
                sign = 1 if side == 'l' else -1
                targets = {'knob': (.204*sign-.0288, -.378, 1.035),
                           'button': (.204*sign+.045, -.322, 1.030),
                           'platter': (.34*sign, -.40, 1.032)}
                harness.rig.force_update()
                finger = harness.rig.expose(f'index_03_{side}').getPos(harness.base.render)
                assert np.linalg.norm(np.array(finger)-targets[variant]) < .015
        harness.rig.actor.setZ(0)
    finally:
        harness.rig.actor.cleanup();harness.rig.actor.removeNode();harness.workstation.removeNode()


def test_rare_accent_reserves_space_without_changing_audio_records():
    from ghost_in_the_deck.dj_planner import PlannedAudioAction
    spans=hype_spans()
    audio=PlannedAudioAction(24,.7,'gain_riser',None,.9)
    spans[0]=replace(spans[0],audio_actions=(audio,))
    visual=VisualTimeline();visual.update(spans)
    assert visual.hypes
    assert spans[0].audio_actions==(audio,)
    assert all(a.end+8<=h.begin or a.begin>=h.end+8
               for a in visual.interactions for h in visual.hypes)
    from test_visual_intent import app_for
    app=app_for(spans);now=visual.hypes[0].begin+1
    assert app.pose_at(now)[2].action=='small_hype'
    app.args.no_actions=True
    assert app.pose_at(now)[2] is None
    assert hype_lift(app.pose_at(now)[2])==0


@pytest.mark.parametrize('side', ['l', 'r'])
def test_hype_has_no_frame_scale_joint_snap_at_entry_or_recovery(side):
    from ghost_in_the_deck.animation.visual_intent import VisualIntent
    intent = VisualIntent('HYPE', side, 'measured_energy_lift', 10, 13.2,
                          10, 13.2, variant='cheer')
    animator = AvatarAnimator(None)
    groove = groove_for()
    previous = None
    for tick in range(211):
        now = 9.9+tick/60
        pose = set_pose_offsets(animator, groove.state_at(now), intent.action_at(now))
        if previous is not None:
            maximum = max(abs(a-b) for joint in pose
                          for a, b in zip(pose[joint], previous.get(joint, (0, 0, 0))))
            assert maximum < 8, (now, maximum)
        previous = pose
