"""Turning a DJActionState into joint offsets, and how it composes with the groove.

Composition rule, precisely, because additive layers are easy to get wrong:

    final = clamp(groove(joint) * damping(joint) + gesture(joint))

The rig applies that clamp; this module supplies the two pieces that feed it.
A gesture's own offsets are pre-scaled by its envelope weight (0 outside the
gesture, ramping smoothly in and out), so they are safe to add directly.
Groove is *damped*, not replaced, on only the joints a given gesture actually
means to control - deck_glance damps head/neck/spine-3, lean_in damps the
lower spine, hand_to_deck damps only the reaching arm's shoulder chain, hype
damps the arms lightly. Every other joint keeps the groove at full strength,
which is what "the groove continues underneath most actions" means concretely:
legs and the untouched arm never even notice a gesture is happening.

Damping itself is scaled by the same envelope weight as the gesture, so it
ramps in and out with it - never a step change on its own.

Two joints is where a real conflict could occur: GrooveEngine's beat-driven
bounce and pulse already push head/neck/spine-3 pitch in the *same direction*
a deck glance wants (both are a downward nod), so without damping a beat
landing mid-glance would add constructively and read as an exaggerated bow.
The damping above is what stops that; it is not decorative.

Head/neck attention (deck_glance) uses a small, real trig calculation from a
fixed head pivot to the target, established by directly probing the rig (see
the module-level comments below) rather than assumed: positive heading turns
the avatar toward its own left, positive pitch tilts the gaze down. The
heading half of that calculation is used as-is - it is a small, well-behaved
angle. The pitch half is not: converging exactly on a target roughly half a
metre away and well below eye height would demand around 50-plus degrees of
head pitch alone, which no human achieves without also bending the neck and
spine, and which this rig's limits do not allow. Rather than chase an
unreachable exact gaze, deck_glance uses a fixed, modest downward tilt spread
across head/neck/spine-3 - the same distribution the groove's own beat-nod
already uses - and lets the trig heading alone carry the "which way" part of
looking toward the deck.

Arm directions for lean_in and small_hype were established the same way: by
rotating each joint on the built avatar and watching which way the hand or
head actually moved, not by assuming a convention.

hand_to_deck is different: it is not a fixed pose, it is a real two-bone IK
reach toward the workstation's control targets (see
``animation.arm_ik.solve_elbow`` for the geometry and
``scripts/solve_arm_ik.py`` for how the geometry was converted into this
rig's joint offsets - a naive world-space aim does not work on this rig's
skeleton, confirmed by testing). ``IK_REACH`` below is that solve's output:
fixed peak offsets, blended in by the gesture's own envelope weight exactly
like every other gesture's constants. The IK computation itself happens once,
offline; nothing here re-solves it at runtime.

small_hype is asymmetric - one arm, chosen by the scheduled event's own
``side`` - rather than both arms mirrored. An earlier symmetric version read
as a T-pose rather than a performance accent; one arm raised, with the other
left to keep grooving normally, reads far more like an actual gesture.

Nothing here imports Panda3D.
"""

from __future__ import annotations

import math

from .dj_behavior import DJActionState
from .workstation import HEAD_PIVOT, DJWorkstationTargets

Offsets = dict[str, tuple[float, float, float]]

# ------------------------------------------------------------- attention trig
# Confirmed by probing the rig: rotating the head joint's own heading/pitch and
# watching which way its forward axis turned in world space. Positive heading
# swings the face toward the avatar's own left (+X); positive pitch tilts the
# gaze down.
_NEUTRAL_FORWARD = (0.0, -1.0, 0.0)   # the avatar faces -Y


def _heading_toward(target: tuple[float, float, float]) -> float:
    """Degrees of head heading to turn from neutral gaze toward ``target``."""
    dx = target[0] - HEAD_PIVOT[0]
    dy = target[1] - HEAD_PIVOT[1]
    return math.degrees(math.atan2(dx, -dy))


# --------------------------------------------------------------- gesture gains
# Peak degrees at full weight and full strength. A real "look straight down at
# the mixer" gaze would need roughly 50 degrees of head pitch alone (see the
# module docstring); GLANCE_* is a documented, deliberately modest stand-in for
# that, distributed the same way the groove's own beat-nod is.
GLANCE_HEAD_PITCH = 9.0
GLANCE_NECK_PITCH = 5.0
GLANCE_SPINE3_PITCH = 3.0
GLANCE_HEADING_LIMIT = 10.0   # cap on the trig heading, so a wide deck offset
                              # cannot turn the glance into a head-swivel

LEAN_SPINE1_PITCH = 4.0
LEAN_SPINE2_PITCH = 5.0
LEAN_SPINE3_PITCH = 4.0
LEAN_CLAVICLE_PITCH = -2.0    # a small forward droop of the shoulders

# Peak arm offsets from scripts/solve_arm_ik.py: a two-bone IK solve against
# the real rig, aimed at 90% of the distance from the shoulder to
# controls_for(side) (see that script's docstring for why not 100%). At full
# weight this lands the wrist about 4.9 cm from the actual control target,
# with a 135 degree elbow bend (about 45 degrees off dead straight) - a real
# reach, not a locked-straight arm. Heading and roll mirror sign between
# sides, matching every other gesture's convention; pitch does not.
REACH_CLAVICLE_PITCH = -3.0
IK_REACH = {
    "l": {
        "upperarm": (-18.563, 11.218, -14.108),
        "lowerarm": (10.780, 28.217, 15.000),
    },
    "r": {
        "upperarm": (18.563, 11.218, 14.108),
        "lowerarm": (-10.780, 28.217, -15.000),
    },
}

# small_hype: one arm raised up and out, established by rendering candidates
# and comparing hand height and hand-to-shoulder distance against neutral (a
# pure pitch-dominant raise pointed the arm down-forward instead of up - roll
# is what actually lifts it on this rig). The lowerarm's pitch limit (35
# degrees) caps how much the elbow can fold; +28 is close to that cap and is
# the most fold the rig allows without clamping away the rest of the raise.
HYPE_UPPERARM_ROLL = 42.0     # sign flipped per side; raises the arm up and out
HYPE_UPPERARM_PITCH = -18.0
HYPE_LOWERARM_PITCH = 28.0
HYPE_CLAVICLE_PITCH = -6.0
HYPE_SPINE3_PITCH = -3.0      # chest lifts - opposite sign from lean_in
HYPE_NECK_PITCH = -2.5
HYPE_HEAD_PITCH = -2.5

# Left-side rotation signs, probed directly on the rig; the right side mirrors
# roll and heading (established convention throughout this project - the
# NEUTRAL_POSE stance and every existing groove term mirror the same way) and
# keeps pitch unmirrored, matching how the groove's own arm terms work.
_SIDE_SIGN = {"l": 1.0, "r": -1.0}


def _add(offsets: Offsets, joint: str, heading: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
    h, p, r = offsets.get(joint, (0.0, 0.0, 0.0))
    offsets[joint] = (h + heading, p + pitch, r + roll)


def pose_offsets(state: DJActionState, targets: DJWorkstationTargets) -> Offsets:
    """The gesture's own joint offsets, already scaled by weight and strength.

    Zero contribution when no action is active - the returned dict is then
    simply empty, so composing it is always safe.
    """
    if not state.is_active:
        return {}

    scale = state.weight * (0.6 + 0.4 * state.strength)
    offsets: Offsets = {}

    if state.action == "deck_glance":
        target = targets.deck_for(state.side)
        heading = max(-GLANCE_HEADING_LIMIT, min(GLANCE_HEADING_LIMIT, _heading_toward(target)))
        _add(offsets, "head", heading=heading * state.weight, pitch=GLANCE_HEAD_PITCH * scale)
        _add(offsets, "neck_01", pitch=GLANCE_NECK_PITCH * scale)
        _add(offsets, "spine_03", pitch=GLANCE_SPINE3_PITCH * scale)

    elif state.action == "lean_in":
        _add(offsets, "spine_01", pitch=LEAN_SPINE1_PITCH * scale)
        _add(offsets, "spine_02", pitch=LEAN_SPINE2_PITCH * scale)
        _add(offsets, "spine_03", pitch=LEAN_SPINE3_PITCH * scale)
        _add(offsets, "clavicle_l", pitch=LEAN_CLAVICLE_PITCH * scale)
        _add(offsets, "clavicle_r", pitch=LEAN_CLAVICLE_PITCH * scale)

    elif state.action == "hand_to_deck":
        side = state.side or "l"
        clavicle = f"clavicle_{side}"
        reach = IK_REACH[side]
        uh, up, ur = reach["upperarm"]
        lh, lp, lr = reach["lowerarm"]
        _add(offsets, f"upperarm_{side}", heading=uh * scale, pitch=up * scale, roll=ur * scale)
        _add(offsets, f"lowerarm_{side}", heading=lh * scale, pitch=lp * scale, roll=lr * scale)
        _add(offsets, clavicle, pitch=REACH_CLAVICLE_PITCH * scale)

    elif state.action == "small_hype":
        # One arm only - see the module docstring for why. roll is negated
        # per side: probing found NEGATIVE roll on upperarm_l is what raises
        # the arm up and away from the body, not positive.
        side = state.side or "l"
        sign = _SIDE_SIGN[side]
        _add(offsets, f"upperarm_{side}", pitch=HYPE_UPPERARM_PITCH * scale,
             roll=-HYPE_UPPERARM_ROLL * sign * scale)
        _add(offsets, f"lowerarm_{side}", pitch=HYPE_LOWERARM_PITCH * scale)
        _add(offsets, f"clavicle_{side}", pitch=HYPE_CLAVICLE_PITCH * scale)
        _add(offsets, "spine_03", pitch=HYPE_SPINE3_PITCH * scale)
        _add(offsets, "neck_01", pitch=HYPE_NECK_PITCH * scale)
        _add(offsets, "head", pitch=HYPE_HEAD_PITCH * scale)

    return offsets


# Which joints each action kind damps the groove on, and by how much at full
# weight. A joint absent from a kind's map keeps the groove at full strength -
# that is "the groove continues underneath the action" made literal.
_DAMPING = {
    "deck_glance": {"head": 0.35, "neck_01": 0.35, "spine_03": 0.55},
    "lean_in": {"spine_01": 0.55, "spine_02": 0.55, "spine_03": 0.55},
    "hand_to_deck": {"upperarm": 0.30, "lowerarm": 0.35, "clavicle": 0.45},
    "small_hype": {"upperarm": 0.55, "lowerarm": 0.55, "clavicle": 0.65,
                   "spine_03": 0.75, "neck_01": 0.75, "head": 0.8},
}


# Both hand_to_deck and small_hype are single-arm actions now, so both damp
# only the joints on the event's own side. The other arm keeps grooving at
# full strength - only the joints an action actually means to control are
# ever damped.
_BOTH_SIDED_ACTIONS: set[str] = set()


def groove_damping(state: DJActionState) -> dict[str, float]:
    """Per-joint multiplier (default 1.0) to apply to the groove's own offsets.

    Ramps with the same envelope weight the gesture itself uses, so damping
    appears and disappears exactly as smoothly as the gesture does.
    """
    if not state.is_active:
        return {}

    joints = _DAMPING.get(state.action, {})
    sides = ("l", "r") if state.action in _BOTH_SIDED_ACTIONS else (state.side or "l",)
    result: dict[str, float] = {}

    for key, floor in joints.items():
        multiplier = 1.0 - state.weight * (1.0 - floor)
        if key in ("upperarm", "lowerarm", "clavicle"):
            for side in sides:
                result[f"{key}_{side}"] = multiplier
        else:
            result[key] = multiplier

    return result
