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
 'l_filter_knob': [[0.819, 17.692, -50.038],
            [37.282, 64.103, 1.796],
            [23.387, -17.04, -12.14]],
 'l_channel_fader': [[3.821, 22.286, -44.197],
                     [39.6, 72.346, 4.165],
                     [23.387, -17.04, -12.14]],
 'l_crossfader': [[5.353, 26.359, -38.936],
                  [40.551, 74.759, 5.104],
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
 'r_filter_knob': [[-3.641, 11.638, 51.641],
            [-35.905, 62.415, 2.849],
            [-22.074, -16.331, 12.569]],
 'r_channel_fader': [[-6.867, 20.338, 42.338],
                     [-39.029, 72.058, -0.335],
                     [-22.074, -16.331, 12.569]],
 'r_crossfader': [[-8.241, 24.259, 37.24],
                  [-40.284, 74.465, -1.581],
                  [-22.074, -16.331, 12.569]],
 'r_platter': [[-7.669, 9.485, 46.795],
               [-37.746, 46.182, 21.576],
               [-14.222, -10.957, 11.26]],
 'r_ready': [[-0.297, -24.298, 60.0],
             [4.182, 89.249, -25.914],
             [-19.358, -7.809, 4.123]]}
# Diagnostic compatibility for old callers; production intent uses the more
# specific filter-knob name.
CONTACTS['l_knob'] = CONTACTS['l_filter_knob']
CONTACTS['r_knob'] = CONTACTS['r_filter_knob']
CHEER = {'l_cheer': [[-27.72, -17.066, -84.956],
             [-12.367, 75.684, -13.124],
             [8.876, 4.142, -8.802]],
 'r_cheer': [[27.699, -17.085, 84.97],
             [12.423, 75.692, 13.073],
             [-8.861, 4.117, 8.801]]}

def finger_offsets(side, amount=1.0):
    """Give every phalanx a loose, graduated resting curve."""
    curls = {
        'index': (5, 5, 3), 'middle': (12, 12, 5),
        'ring': (20, 20, 6), 'pinky': (24, 24, 7),
        'thumb': (10, 10, 4),
    }
    return {f'{finger}_0{segment}_{side}': (0, values[segment-1]*amount, 0)
            for finger, values in curls.items() for segment in (1, 2, 3)}


def set_pose_offsets(animator, state, action, monitor_side=None):
    offsets = animator.pose_offsets(state)
    # The beat travels upward through the body.  Each layer keeps part of its
    # slower groove and receives a smaller, phase-delayed response; this avoids
    # rotating the whole torso/head hierarchy as one rigid stick.
    phase=2*math.pi*state.beat_phase
    for joint, retain, amplitude, lag in [
            ('spine_03', .72, .65, 1.05),
            ('neck_01', .58, 1.15, .48),
            ('head', .46, 2.05, .08),
            ('clavicle_l', .76, -.55, .88),
            ('clavicle_r', .76, -.48, 1.18)]:
        h,p,r=offsets.get(joint,(0,0,0))
        travel=math.sin(phase-lag)*amplitude*state.intensity
        offsets[joint]=(h,p*retain+travel,r)
    h,p,r=offsets['head']
    offsets['head']=(h-.34*(offsets['spine_02'][0]+offsets['spine_03'][0]),
                     p+1.05*state.breath,r*.68)
    h,p,r=offsets['neck_01']
    offsets['neck_01']=(h-.22*offsets['spine_03'][0],p,
                        r-.42*offsets['spine_03'][2])
    for side in ('l','r'):
        offsets.update(finger_offsets(side))
        sign=1 if side=='l' else -1
        # Hands are alive even at rest: soft elbows and neutral wrists, never
        # the straight dangling mannequin chain.
        h,p,r=offsets.get(f'lowerarm_{side}',(0,0,0))
        offsets[f'lowerarm_{side}']=(h,p+3.0+.8*state.breath,r)
        offsets[f'hand_{side}']=(0,-1.8,2.8*sign)
    if monitor_side in ('l','r'):
        sign=1 if monitor_side=='l' else -1
        amount=.08+.035*(.5+.5*sign*state.weight_shift)
        for joint,target in zip(('upperarm','lowerarm','hand'),
                                CONTACTS[f'{monitor_side}_ready']):
            name=f'{joint}_{monitor_side}'
            base=offsets.get(name,(0,0,0))
            offsets[name]=tuple(a+b*amount for a,b in zip(base,target))
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
    # Bend the elbow first, then carry the hand forward above the near edge.
    # The previous sideways clearance pose produced a conspicuous horizontal
    # arm followed by a drop onto the control.
    tuck=((0,0,0),(0,-55,0),(0,0,0))
    lifted=((0,-10,-45*sign),(0,-55,0),(0,38,-14*sign))
    if u < .16:
        a,b,w=((0,0,0),)*3,tuck,_smoothstep(u/.16)
    elif u < .34:
        a,b,w=tuck,lifted,_smoothstep((u-.16)/.18)
    elif u < .66:
        a,b,w=lifted,ready,_smoothstep((u-.34)/.32)
    else:
        a,b,w=ready,contact,_smoothstep((u-.66)/.34)
    for joint,first,last in zip(('upperarm','lowerarm','hand'),a,b):
        name = f'{joint}_{side}'
        groove = offsets.get(name, (0, 0, 0))
        offsets[name]=tuple(x+(y-x)*w+g*(1-presence) for x,y,g in zip(first,last,groove))
    # Keep the reaching chain steady, while the upper neck and head retain
    # independent listening motion. The non-reaching shoulder can breathe.
    for joint in ('pelvis','spine_01','spine_02','spine_03'):
        offsets[joint]=tuple(v*(1-.72*presence) for v in offsets.get(joint,(0,0,0)))
    name=f'clavicle_{side}'
    offsets[name]=tuple(a*(1-presence)+b*presence for a,b in zip(offsets.get(name,(0,0,0)),(0,-3,0)))
    free='r' if side=='l' else 'l'
    free_sign=1 if free=='l' else -1
    # The supporting arm stays relaxed and slightly ready instead of becoming
    # an inert straight limb while the working hand reaches.
    support={f'upperarm_{free}': (0,-2,-5*free_sign),
             f'lowerarm_{free}': (0,12,3*free_sign),
             f'hand_{free}': (0,-3,4*free_sign)}
    for name,target in support.items():
        base=offsets.get(name,(0,0,0))
        offsets[name]=tuple(a+(b-a)*.55*presence for a,b in zip(base,target))
    # One deliberate control movement, never a looping hand oscillation.
    contact_weight=_smoothstep((u-.92)/.08)
    motion=math.sin(math.pi*action.contact_phase)**2*contact_weight
    hand = f'hand_{side}'
    h, p, r = offsets[hand]
    if action.variant in ('knob','filter_knob'):
        offsets[hand]=(h,p,r+3*motion)
    elif action.variant=='button':
        base=offsets[f'index_02_{side}'][1]
        offsets[f'index_02_{side}']=(0,base+7*motion,0)
    elif action.variant=='channel_fader':
        lower=f'lowerarm_{side}'
        lh,lp,lr=offsets[lower]
        offsets[lower]=(lh,lp+.5*motion,lr)
    elif action.variant=='crossfader':
        # A small lateral sweep at the centre mixer, not a looping flourish.
        offsets[hand]=(h+.45*sign*motion,p,r)
    else:
        offsets[hand]=(h+1.2*motion,p,r)
    return offsets


def hype_lift(action):
    """One small vertical bounce; absolute time, always returns to floor height."""
    if action is None or action.action != 'small_hype':
        return 0.0
    u=(action.progress-.30)/.40
    return .035*math.sin(math.pi*u)**2 if 0 < u < 1 else 0.0
