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
fixed peak offsets that land the wrist hovering just above the control point
(``scripts/solve_arm_ik.py``'s REACH_HOVER_HEIGHT / REACH_HOVER_BACK aim a
few centimetres above and in front of it, not on it - the un-posed fingers
splay below the wrist and a wrist exactly at control height leaves them in
the slab). The IK computation itself happens once, offline; nothing here
re-solves it at runtime.

Unlike every other gesture, hand_to_deck does not blend straight from neutral
to its peak offset by the generic envelope weight - a straight joint-space
blend measurably carried the hand through the tabletop during attack and
release (Phase 1B.2's finding: interpolating neutral -> final reach in one
sweep passes low and forward before it passes high, and the tabletop sits
exactly in that path). Instead the reach is staged through a second calibrated
pose, ``IK_CLEARANCE``, so the path goes neutral -> clearance -> target ->
clearance -> neutral, always passing over the table rather than through it.
``_reach_phase_weights`` below derives which two of those five legs are
active, and how far along, directly from ``state.progress`` rather than from
the generic single-hump envelope weight, because the generic envelope has no
way to express "two waypoints, in order, and back" - see its own docstring
for the exact split.

IK_CLEARANCE is not solved against an independent world point. Phase 1B.2
first tried that (aim the same two-bone solve at a hover point above and in
front of the tabletop's near edge); a bounded, widened-limit damped
least-squares search from a single start did not find a solution in this
rig's HPR parameterisation (reproduced by
``scripts/solve_arm_ik.py``'s ``_diagnose_independent_clearance_target``).
That is a limited negative result, not a proof that no such rotation exists,
so the calibration does not depend on it. Instead IK_CLEARANCE reuses
IK_REACH's *upperarm* offset verbatim (the shoulder's own well-converging
rotation) and folds the *forearm* up with a fixed ``CLEARANCE_ELBOW_PITCH``,
so the clearance-pose hand sits high and in front of the table rather than
dangling at its edge - on this rig, abducting the shoulder alone swings the
hand out over the near edge before it gains enough height. On top of that the
reach adds several small, separately named terms, none of which touch the
IK_REACH endpoint: ``CLEARANCE_LIFT_ROLL`` (extra abduction, a trapezoid over
the attack and, tracking the hand back down, over the release),
``REACH_HAND_PITCH`` + ``REACH_TRANSIT_HAND_PITCH`` (tilt the un-posed fingers
up across the controls), ``REACH_SWING_OUT_ROLL`` / ``REACH_SWING_OUT_HEADING``
(a transient corner detour on the neutral<->clearance legs), and
``REACH_HOVER_HEADING`` (a few degrees inward at the hold, off the platter on
the hand's own side). The safety terms - the clearance elbow fold, the lift
roll and the wrist pitch - are deliberately NOT strength-scaled: a timid
low-energy reach has to clear the slab just as well as an emphatic one, and
the low-strength cases are exactly where the fingertip joints were measured
grazing it. ``CLEARANCE_LIFT_ROLL``'s magnitude is derived end to end by
``scripts/solve_arm_ik.py`` (see its comment); every term is re-checked
against the real composed pose - groove included, the full scheduled event
population - by ``tests/reach_clearance.py``, which asserts a real margin on
every test run.

small_hype is asymmetric - one arm, chosen by the scheduled event's own
``side`` - rather than both arms mirrored. An earlier symmetric version read
as a T-pose rather than a performance accent; one arm raised, with the other
left to keep grooving normally, reads far more like an actual gesture.

Nothing here imports Panda3D.
"""

from __future__ import annotations

import math

from .dj_behavior import ENVELOPE_SHAPE, DJActionState
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
        "upperarm": (-14.006, 8.864, -19.379),
        "lowerarm": (14.728, 42.352, 15.000),
    },
    "r": {
        "upperarm": (14.006, 8.864, 19.379),
        "lowerarm": (-14.728, 42.352, -15.000),
    },
}

# The clearance pose: see the module docstring for why the upper arm is
# IK_REACH's own (not an independently aimed target). scripts/solve_arm_ik.py
# reproduces these too. The forearm is NOT at rest: CLEARANCE_ELBOW_PITCH folds
# it up so the clearance-pose hand sits high and *in front of* the table rather
# than dangling at the table's edge - measured on this rig, abducting the
# shoulder alone swings the hand out over the near edge before it gains enough
# height, and the trailing fingers clip the slab there. Negative pitch is
# flexion on this joint (IK_REACH's own ~+42 is extension toward the low
# control target); -60 folds the forearm most of the way up, lifting the wrist
# ~20 cm with almost no sideways travel (probed directly on the rig).
CLEARANCE_ELBOW_PITCH = -60.0
IK_CLEARANCE = {
    "l": {
        "upperarm": IK_REACH["l"]["upperarm"],
        "lowerarm": (0.0, CLEARANCE_ELBOW_PITCH, 0.0),
    },
    "r": {
        "upperarm": IK_REACH["r"]["upperarm"],
        "lowerarm": (0.0, CLEARANCE_ELBOW_PITCH, 0.0),
    },
}

# The extra upperarm roll (see the module docstring and ``_reach_phase_weights``
# for its shape) that lifts the hand over the tabletop's near edge on the
# neutral<->clearance legs. Mirrors sign between sides like every other roll
# term here.
#
# It is derived end to end, no hand-tuned number:
#
#   scripts/solve_arm_ik.py's _search_lift_roll sweeps the *real* composed
#   reach pose (this module's own pose_offsets, minus the groove) at full
#   strength and finds the diminishing-returns knee for the worst hand+finger
#   margin against the built furniture: LIFT_ROLL_KNEE.
#
#   COMPOSED_MARGIN_BUFFER_ROLL is the one policy choice - how much further to
#   roll past that knee, since the isolated knee search sees neither the round
#   platter geometry nor the groove sway that tests/reach_clearance.py's full
#   population sweep does, and both erode the margin. Its size was set from that
#   sweep's worst-margin report.
#
#   CLEARANCE_LIFT_ROLL = LIFT_ROLL_KNEE + COMPOSED_MARGIN_BUFFER_ROLL, and
#   test_reach_trajectory.py asserts exactly that, so the number cannot drift
#   and its provenance is one re-runnable script plus one named constant.
LIFT_ROLL_KNEE = -34.0
COMPOSED_MARGIN_BUFFER_ROLL = -6.0
CLEARANCE_LIFT_ROLL = LIFT_ROLL_KNEE + COMPOSED_MARGIN_BUFFER_ROLL   # -40.0

# Phase 1B.2 finding F1: the wrist clears the workstation, but the fifteen
# un-posed finger joints each hand carries (the furthest, ``middle_03``, ~160
# mm past the wrist) do not - at the hold they drape below and ahead of the
# wrist and were measured up to 25 mm inside the tabletop, and through the deck
# platter and the side control panel. The reach endpoint (``IK_REACH``) is
# already aimed a few centimetres above and in front of the control point
# rather than at it (scripts/solve_arm_ik.py's REACH_HOVER_HEIGHT /
# REACH_HOVER_BACK) so the settled hold pose hovers rather than presses; these
# extra terms keep the *path* to and from it clear as well, without touching
# that endpoint:
#
#   REACH_HAND_PITCH   - pitches the whole hand up so the fingers angle across
#                        the controls rather than down into them. Same sign
#                        both sides (pitch is never mirrored here). Ramps in
#                        with the reach so the fingers are already tilted before
#                        the hand reaches the table.
#   REACH_TRANSIT_HAND_PITCH - extra wrist tilt-up while the hand is actually
#                        crossing over the table (tied to ``lift``, so it is
#                        gone again at the settled target): the trailing finger
#                        joints are the hazard during that crossing, not at the
#                        hold.
#   REACH_HOVER_HEADING- swings the upper arm a few degrees further in toward
#                        the centre line during the hold, lifting the fingers
#                        off the deck platter / deck base on the hand's own
#                        side. Mirrors sign like IK_REACH's heading.
#   REACH_SWING_OUT_HEADING / REACH_SWING_OUT_ROLL - a transient detour on the
#                        neutral<->clearance legs only (see
#                        ``_reach_phase_weights``'s ``swing``). The roll term
#                        does most of the work: extra shoulder abduction carries
#                        the hand out past the table's *side* edge and in front
#                        of it, so the neutral<->clearance swing routes around
#                        the near corner instead of dragging the trailing
#                        fingers through the slab. Both are zero at the
#                        clearance pose and beyond, so the endpoint is untouched.
#
# These are tuning constants, not solver output. Their magnitudes were chosen
# by sweeping them against the built-geometry population harness in
# tests/reach_clearance.py, whose worst-margin report is deterministic and
# which every test run re-asserts. Wrist pitch (peak ~54 deg with the transit
# term) stays inside its 60 deg guard rail; the reach's total upper-arm
# abduction is capped at REACH_UPPERARM_ROLL_CAP, below its 60 deg rail, and
# the headings stay well inside their 25 deg one.
REACH_HAND_PITCH = 48.0
REACH_HAND_PITCH_RAMP = 0.40   # full base wrist pitch by 40% of the way in
REACH_TRANSIT_HAND_PITCH = 6.0
REACH_HOVER_HEADING = -3.0
REACH_SWING_OUT_HEADING = 10.0
REACH_SWING_OUT_ROLL = -8.0
# Cap on the reach's own upperarm abduction, below the joint's 60 deg guard
# rail, leaving headroom for a groove sway composed on top.
REACH_UPPERARM_ROLL_CAP = 57.0
# Release-only: how fast the lift roll follows the hand back down. On the
# attack the lift is a trapezoid (arm up before it swings forward); on the
# release it instead tracks ``w_clear`` down this ramp, so the abduction
# relaxes only as the hand actually returns toward neutral and never drops the
# arm while it is still swung out over the table.
_LIFT_RELEASE_RAMP = 0.50

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


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# The attack lift trapezoid: full by this fraction of the attack, easing back
# to zero over the last (1 - _LIFT_FALL) of it. It leads the forward swing so
# the arm is abducted and the forearm folded up before the hand travels out
# over the table, but not so abruptly that the wrist teleports on the first
# frame of the gesture (tests/test_reach_trajectory's continuity bound).
_LIFT_RISE = 0.22
_LIFT_FALL = 0.92

# How much of the neutral->clearance leg is spent moving (the rest sits parked
# at the clearance pose): ease to the lifted pose over most of the leg, then
# hold. On the way back the clearance pose is instead held through the first
# ``_RELEASE_HOLD`` of the clearance->neutral leg and dropped over what remains
# - the fingers have to stay high until the hand is back past the table's near
# edge, and a straight mirror of the attack would start dropping while still
# over it.
_WAYPOINT_SNAP = 0.85
_RELEASE_HOLD = 0.20


def _plateau(u: float, rise: float, fall: float) -> float:
    """0..1..0 over ``u`` in [0, 1]: ramp up by ``rise``, hold, ramp down after ``fall``."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if u < rise:
        return _smoothstep(u / rise)
    if u <= fall:
        return 1.0
    return 1.0 - _smoothstep((u - fall) / (1.0 - fall))


def _trapezoid_lift(u: float) -> float:
    return _plateau(u, _LIFT_RISE, _LIFT_FALL)


def _reach_phase_weights(progress: float) -> tuple[float, float, float, float]:
    """(clearance_weight, target_weight, lift, swing) at this point in a hand_to_deck event.

    Five legs: rise to the clearance pose, cross from clearance to the final
    control target, hold at the target, return to clearance, and descend from
    clearance back to neutral. Each leg eases with the same smoothstep every
    other envelope in this project uses; the two weights sum to at most 1 and
    are never both nonzero outside the clearance<->target crossings, so a caller
    blending ``IK_CLEARANCE`` and ``IK_REACH`` by these never needs to worry
    about the two "fighting" each other.

    ``lift`` is 0 at neutral, at the final target and through the hold. On the
    *attack* it is a trapezoid (``_LIFT_RISE`` / ``_LIFT_FALL``) that leads the
    forward swing: the shoulder abducts and the forearm folds up (the clearance
    pose) before the hand travels out over the table, because measured against
    the real finger joints - not just the wrist - a path that gains height and
    forward reach together drags the trailing fingers through the slab as they
    cross the table's near edge. On the *release* ``lift`` instead tracks
    ``w_clear`` back down (``_LIFT_RELEASE_RAMP``), so the abduction relaxes
    only as the hand actually returns toward neutral; an earlier timer-based
    fall dropped the arm while it was still swung out over the table.

    ``swing`` is a 0..1..0 plateau nonzero *only* on the two neutral<->clearance
    legs, at full through the middle of each. It drives the detour terms
    ``REACH_SWING_OUT_ROLL`` (extra shoulder abduction) and
    ``REACH_SWING_OUT_HEADING``: a straight joint-space interpolation between
    neutral and the clearance pose sweeps the hand across the table's near edge,
    and the fingers dangling behind the wrist clip it. Detouring the hand out
    toward the table's *side* edge for the middle of that swing routes it around
    the corner. Identically zero on the clearance<->target crossing and the
    hold, so the reach endpoint is untouched.

    All-zero outside [0, 1] - the event has not started or has already ended.
    """
    attack, hold, release = ENVELOPE_SHAPE["hand_to_deck"]
    if progress <= 0.0 or progress >= 1.0:
        return 0.0, 0.0, 0.0, 0.0

    half_attack = attack / 2.0
    hold_end = attack + hold
    half_release = (1.0 - hold_end) / 2.0

    if progress < attack:
        lift = _trapezoid_lift(progress / attack)
        if progress < half_attack:
            # neutral -> clearance: reach the clearance posture quickly (by
            # _WAYPOINT_SNAP of this leg) and sit there until the crossing
            # begins. The slowest part of this swing is the hand passing the
            # table's near edge; getting to the lifted clearance pose sooner
            # keeps the trailing fingers from dragging through the slab.
            w_clear = _smoothstep(progress / (half_attack * _WAYPOINT_SNAP))
            return w_clear, 0.0, lift, _plateau((w_clear - 0.05) / 0.90, 0.22, 0.80)
        # clearance -> target
        t = _smoothstep((progress - half_attack) / half_attack)
        return 1.0 - t, t, lift, 0.0
    if progress < hold_end:
        # hold at target
        return 0.0, 1.0, 0.0, 0.0
    # target -> clearance -> neutral
    release_progress = progress - hold_end
    if release_progress < half_release:
        t = _smoothstep(release_progress / half_release)
        return t, 1.0 - t, _trapezoid_lift(1.0 - release_progress / release), 0.0
    # clearance -> neutral: hold the clearance pose through _RELEASE_HOLD of
    # this leg, then drop to neutral over what remains. The lift roll follows
    # w_clear down rather than the clock, so it is still full while the hand is
    # over the table and only gone once it is back at the side.
    s = (release_progress - half_release) / half_release
    w = _smoothstep((s - _RELEASE_HOLD) / (1.0 - _RELEASE_HOLD))
    w_clear = 1.0 - w
    lift = _smoothstep(min(1.0, w_clear / _LIFT_RELEASE_RAMP))
    return w_clear, 0.0, lift, _plateau((w_clear - 0.05) / 0.90, 0.22, 0.80)


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
        # Not a single neutral -> peak blend like every other gesture (see the
        # module docstring for why): a staged path through IK_CLEARANCE, keyed
        # off progress directly rather than the generic envelope weight.
        side = state.side or "l"
        clavicle = f"clavicle_{side}"
        strength_scale = 0.6 + 0.4 * state.strength
        w_clear, w_target, lift, swing = _reach_phase_weights(state.progress)
        clearance = IK_CLEARANCE[side]
        reach = IK_REACH[side]
        ch, cp, cr = clearance["upperarm"]
        lch, lcp, lcr = clearance["lowerarm"]
        uh, up, ur = reach["upperarm"]
        lh, lp, lr = reach["lowerarm"]
        # ``reach_presence`` is 0 at true neutral and 1 from the clearance pose
        # onward (the two weights always sum to 1 once past it); the wrist pitch
        # and clavicle droop scale by it so neither is applied to an arm still
        # hanging at rest. The lift roll instead follows the trapezoid ``lift``
        # directly - it needs to be at full strength early in the swing, while
        # ``reach_presence`` is still ramping, to carry the hand over the near
        # edge; the trapezoid is already 0 at true neutral.
        reach_presence = w_clear + w_target
        lift_roll = CLEARANCE_LIFT_ROLL * lift * _SIDE_SIGN[side]
        hover_heading = REACH_HOVER_HEADING * _SIDE_SIGN[side] * w_target
        swing_heading = REACH_SWING_OUT_HEADING * _SIDE_SIGN[side] * swing
        swing_roll = REACH_SWING_OUT_ROLL * _SIDE_SIGN[side] * swing
        # The wrist reaches full base pitch by the time the hand is ~40% of the
        # way into the swing and holds it from there, so the finger joints are
        # already angled up before the hand crosses the table. The transit term
        # adds more tilt while the hand is actually over the slab (tied to
        # ``lift``) and is gone again at the settled target.
        hand_pitch_weight = _smoothstep(min(1.0, reach_presence / REACH_HAND_PITCH_RAMP))
        hand_pitch = REACH_HAND_PITCH * hand_pitch_weight + REACH_TRANSIT_HAND_PITCH * lift
        # The clearance-pose abduction and the lift/swing roll carry the hand
        # clear of the table; like the wrist pitch they are NOT strength-scaled,
        # so a timid low-energy reach clears the slab just as well as an
        # emphatic one (the low-strength cases are where the fingertips grazed).
        # Capped below the joint's guard rail so a groove sway composed on top
        # can never push the total past it - the gesture stays safe without
        # relying on the rig's own clamp.
        reach_roll = (cr * w_clear + ur * w_target) * strength_scale + lift_roll + swing_roll
        reach_roll = max(-REACH_UPPERARM_ROLL_CAP, min(REACH_UPPERARM_ROLL_CAP, reach_roll))
        _add(
            offsets, f"upperarm_{side}",
            heading=(ch * w_clear + uh * w_target + hover_heading + swing_heading) * strength_scale,
            pitch=(cp * w_clear + up * w_target) * strength_scale,
            roll=reach_roll,
        )
        _add(
            offsets, f"lowerarm_{side}",
            heading=(lch * w_clear + lh * w_target) * strength_scale,
            # Clearance-pose elbow flexion (lcp) is safety geometry: unscaled.
            # The reach's own extension (lp) is the gesture: strength-scaled.
            pitch=lcp * w_clear + lp * w_target * strength_scale,
            roll=(lcr * w_clear + lr * w_target) * strength_scale,
        )
        # Wrist pitch is NOT strength-scaled: the finger tilt-up that keeps the
        # trailing joints out of the slab is a safety measure, and a timid
        # (low-energy) reach needs it just as much as an emphatic one - the
        # low-strength cases are exactly where the un-posed fingertips were
        # measured grazing the tabletop.
        _add(offsets, f"hand_{side}", pitch=hand_pitch)
        _add(offsets, clavicle, pitch=REACH_CLAVICLE_PITCH * reach_presence * strength_scale)

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
    "hand_to_deck": {"upperarm": 0.25, "lowerarm": 0.30, "clavicle": 0.42},
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
