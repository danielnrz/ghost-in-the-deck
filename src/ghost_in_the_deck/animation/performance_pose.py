"""Small live performance layer: articulated hands and calibrated control poses.

All joint solves are offline. Runtime is a handful of eased interpolations.
"""
from __future__ import annotations

import math
from .dj_behavior import _smoothstep

# Offline calibration against the mesh and physical targets. Each entry holds
# upper arm, forearm, then wrist offsets in heading/pitch/roll degrees.
CONTACTS = {'l_button': [[6.028, 0.845, -55.72],
              [29.552, 71.802, 7.182],
              [24.99, -20.999, -14.592]],
 'l_knob': [[0.819, 17.692, -50.038],
            [37.282, 64.103, 1.796],
            [23.387, -17.04, -12.14]],
 'l_platter': [[7.651, 9.489, -46.818],
               [37.72, 46.192, -21.5],
               [14.224, -10.977, -11.275]],
 'l_ready': [[-0.003, -24.399, -60.0],
             [-4.453, 89.39, 26.367],
             [19.269, -7.803, -3.961]],
 'r_button': [[4.386, 11.54, 53.221],
              [-28.72, 75.349, -25.2],
              [-24.973, -24.034, 12.247]],
 'r_knob': [[-3.641, 11.638, 51.641],
            [-35.905, 62.415, 2.849],
            [-22.074, -16.331, 12.569]],
 'r_platter': [[-7.669, 9.485, 46.795],
               [-37.746, 46.182, 21.576],
               [-14.222, -10.957, 11.26]],
 'r_ready': [[-0.297, -24.298, 60.0],
             [4.182, 89.249, -25.914],
             [-19.358, -7.809, 4.123]]}
CHEER = {'l_cheer': [[-27.72, -17.066, -84.956],
             [-12.367, 75.684, -13.124],
             [8.876, 4.142, -8.802]],
 'r_cheer': [[27.699, -17.085, 84.97],
             [12.423, 75.692, 13.073],
             [-8.861, 4.117, 8.801]]}

def finger_offsets(side, amount=1.0):
    """Relax the fan-shaped baked hand; flex toward the palm on both sides."""
    return {f'{finger}_0{segment}_{side}': (0,curl*amount,0)
            for finger,curl in [('index',5),('middle',12),('ring',20),('pinky',24),('thumb',10)]
            for segment in (1,2)}


def set_pose_offsets(animator, state, action):
    offsets = animator.pose_offsets(state)
    # Head nod leads; shoulders and chest respond later with less amplitude.
    # Local neck counter-rotation prevents inherited torso motion adding twice.
    phase=2*math.pi*state.beat_phase
    for joint, amplitude, lag in [('head',3.0,0),('neck_01',1.2,.35),
                                  ('spine_03',.8,.8),('clavicle_l',-.7,1.0),
                                  ('clavicle_r',-.6,1.25)]:
        h,p,r=offsets.get(joint,(0,0,0))
        offsets[joint]=(h,amplitude*(.5+.5*math.cos(phase-lag))*state.intensity,r)
    h,p,r=offsets['head']
    offsets['head']=(h-.6*offsets['spine_02'][0],p+1.1*state.breath,r*.6)
    h,p,r=offsets['neck_01']
    offsets['neck_01']=(h,p,r-.45*offsets['spine_03'][2])
    for side in ('l','r'):
        offsets.update(finger_offsets(side))
    if action is None or not action.is_active:
        return offsets
    if action.action == 'small_hype':
        side = action.side
        w = action.weight
        sign=1 if side=='l' else -1
        clearance=((0,0,-55*sign),(0,-30,0),(0,0,0))
        if w < .4:
            first,last,blend=((0,0,0),)*3,clearance,_smoothstep(w/.4)
        else:
            first,last,blend=clearance,CHEER[f'{side}_cheer'],_smoothstep((w-.4)/.6)
        for joint,a,b in zip(('upperarm','lowerarm','hand'),first,last):
            name = f'{joint}_{side}'
            base = offsets.get(name, (0, 0, 0))
            offsets[name]=tuple(x+(y-x)*blend+g*(1-w) for x,y,g in zip(a,b,base))
        for finger in ('index','middle','ring','pinky','thumb'):
            for segment in (1,2,3):
                name=f'{finger}_0{segment}_{side}'
                base=offsets.get(name,(0,0,0))[1]
                curl=(20 if finger=='thumb' else 35) if segment < 3 else 10
                offsets[name]=(0,base+(curl-base)*w,0)
        # One modest greeting wave and a soft knee accent; no flailing loop.
        h,p,r=offsets[f'hand_{side}']
        offsets[f'hand_{side}']=(h+4*math.sin(2*math.pi*action.progress)*w,p,r)
        for leg in ('l','r'):
            h, p, r = offsets[f'calf_{leg}']
            offsets[f'calf_{leg}'] = (h, p+3*w, r)
        return offsets
    if action.action != 'hand_to_deck':
        return offsets
    side=action.side
    contact=CONTACTS[f'{side}_{action.variant}']
    ready=CONTACTS[f'{side}_ready']
    u=2*min(action.progress,1-action.progress)
    presence=_smoothstep(u/.35)
    sign=1 if side=='l' else -1
    tuck=((0,0,0),(0,-55,0),(0,0,0))
    lifted=((0,-10,-60*sign),(0,-55,0),(0,45,-20*sign))
    if u < .20:
        a,b,w=((0,0,0),)*3,tuck,_smoothstep(u/.20)
    elif u < .40:
        a,b,w=tuck,lifted,_smoothstep((u-.20)/.20)
    elif u < .70:
        a,b,w=lifted,ready,_smoothstep((u-.40)/.30)
    else:
        a,b,w=ready,contact,_smoothstep((u-.70)/.30)
    for joint,first,last in zip(('upperarm','lowerarm','hand'),a,b):
        name = f'{joint}_{side}'
        groove = offsets.get(name, (0, 0, 0))
        offsets[name]=tuple(x+(y-x)*w+g*(1-presence) for x,y,g in zip(first,last,groove))
    # Keep the reaching chain steady, while the upper neck and head retain
    # independent listening motion. The non-reaching shoulder can breathe.
    for joint in ('pelvis','spine_01','spine_02','spine_03'):
        offsets[joint]=tuple(v*(1-.9*presence) for v in offsets.get(joint,(0,0,0)))
    name=f'clavicle_{side}'
    offsets[name]=tuple(a*(1-presence)+b*presence for a,b in zip(offsets.get(name,(0,0,0)),(0,-3,0)))
    # One deliberate control movement, never a looping hand oscillation.
    contact_weight=_smoothstep((u-.92)/.08)
    motion=math.sin(math.pi*action.contact_phase)**2*contact_weight
    hand = f'hand_{side}'
    h, p, r = offsets[hand]
    if action.variant=='knob':
        offsets[hand]=(h,p,r+3*motion)
    elif action.variant=='button':
        offsets[f'index_02_{side}']=(0,5+7*motion,0)
    else:
        offsets[hand]=(h+1.2*motion,p,r)
    return offsets


def hype_lift(action):
    """One small vertical bounce; absolute time, always returns to floor height."""
    if action is None or action.action != 'small_hype':
        return 0.0
    u=(action.progress-.30)/.40
    return .035*math.sin(math.pi*u)**2 if 0 < u < 1 else 0.0
