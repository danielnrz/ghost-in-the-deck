"""Derive the hand_to_deck IK joint offsets from the real, built avatar.

    PYTHONPATH=src .venv/bin/python scripts/solve_arm_ik.py

This is a one-time calibration, not something run at play time. It exists so
the constants in ``gesture_pose.py`` (``IK_REACH``, ``IK_CLEARANCE`` and
``CLEARANCE_LIFT_ROLL``) are reproducible and can be regenerated if the avatar
asset ever changes - re-run this script and update the constants with its
output.

Why this can't just be done live, every frame, with a closed-form formula:
``animation.arm_ik.solve_elbow`` gives the elbow's correct WORLD position
analytically (verified: it satisfies both bone-length constraints exactly).
Turning that into a Panda3D joint rotation is the hard part. The rig's
``controlJoint`` transform is not a plain scene-graph-relative transform -
calling ``NodePath.lookAt()`` on it directly produces a direction with a
strongly negative dot product against the intended target, i.e. it aims
roughly backwards. Rather than guess at what the internal mapping actually is,
this solves numerically: nudge the joint's heading/pitch/roll by a small
amount, watch how the exposed child joint's world position actually moves,
and take repeated small, damped steps toward the target. Damped because a
single bone's rotation only has two real degrees of freedom over where its tip
ends up - spinning it about its own long axis barely moves the tip - so the
raw 3x3 Jacobian is close to singular; an undamped least-squares solve on it
was confirmed to propose a step of several hundred thousand degrees.

IK_REACH - two calibration choices worth recording:

* The target used is ``controls_for(side)``, not the platter centre
  (``deck_for``) - the platters sit noticeably beyond this avatar's reach
  (10-27 cm too far), while the front control row is within about a
  centimetre of it, so "hand toward the controls" is the honest reading of
  what this arm can actually do.
* The aim point is deliberately pulled to 90% of the true target's distance
  from the shoulder, not aimed at exactly 100%, and lifted a few centimetres
  above and back from the control point (REACH_HOVER_HEIGHT / REACH_HOVER_BACK)
  so the hand hovers over the controls rather than resting on them and its
  un-posed fingers do not splay into the tabletop. The result: a visibly bent
  elbow (~120 degrees, i.e. ~60 off dead straight) with the wrist landing about
  7 cm from the true control point - inside the 5-8 cm this project treats as
  "close enough to read as reaching for the controls."

IK_CLEARANCE - NOT solved the same way. Phase 1B.2 first tried the obvious
thing: aim the same two-bone solve at an independent hover point above and in
front of the tabletop's near edge. In this rig's HPR parameterisation that
solve did not converge: with the lowerarm joint's limit box widened to
+-90 degrees on every axis and the iteration budget raised to 900, and even
once the upper arm had converged to within a centimetre of the
analytically-correct elbow position, no rotation the damped-least-squares
search reached brought the wrist within 25 cm of the hover target, though the
target sat within 2 mm of the correct distance from the elbow. That is the
extent of what was actually checked - a bounded, widened-limit DLS search from
a single start (reproduced by ``_diagnose_independent_clearance_target``
below). It is *not* an exhaustive search of the joint's rotation space and is
not evidence that no such rotation exists; it only means the obvious solve did
not find one, so this calibration does not rely on it.

IK_CLEARANCE instead reuses IK_REACH's *upperarm* offset verbatim (the
shoulder's own well-converging rotation) and folds the *forearm* up with a
fixed CLEARANCE_ELBOW_PITCH imported from gesture_pose - a big negative
lowerarm pitch that lifts the wrist nearly straight up, so the clearance-pose
hand sits high and in front of the table instead of dangling at its edge
(abducting the shoulder alone swings the hand out over the near edge before it
is high enough, and the trailing fingers clip the slab). ``CLEARANCE_LIFT_ROLL``
is an extra upperarm abduction on top of that, a trapezoid over the attack and,
tracking the hand back down, over the release. ``_search_lift_roll`` below
finds its knee by sweeping the *real composed* reach pose - the exact runtime
pose ``gesture_pose.pose_offsets`` produces, all its wrist-pitch / swing /
elbow-fold terms included, minus only the groove - and measuring every one of
the sixteen hand and finger joints per side against the built workstation
furniture. That knee plus one named policy buffer
(gesture_pose.COMPOSED_MARGIN_BUFFER_ROLL, covering the round-platter geometry
and the groove sway that this isolated search does not see) is the committed
constant. ``tests/test_reach_trajectory.py`` asserts the script still
reproduces it exactly, and ``tests/reach_clearance.py`` re-checks the whole
composed reach against the full scheduled event population, groove included.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REACH_AIM_SHRINK = 0.90   # see module docstring
REACH_HOVER_HEIGHT = 0.022   # metres above the control point the wrist aims for
REACH_HOVER_BACK = 0.020   # metres toward the avatar (away from the platter)
EPS = 0.5           # degrees, finite-difference step for the Jacobian
MAX_STEP = 4.0      # degrees, per-iteration step cap
LAMBDA = 1e-3        # Tikhonov damping
ITERATIONS = 200
CLAVICLE_REACH_PITCH = -3.0   # existing small forward droop, unchanged

# The lift-roll search sweeps the *isolated* composed reach pose (the real
# runtime pose minus the groove) and reports the point of diminishing returns:
# the smallest |roll| whose worst hand-joint margin is within LIFT_SEARCH_TOL
# of the best any larger roll achieves. Past the reach's own abduction cap
# (gesture_pose.REACH_UPPERARM_ROLL_CAP) more roll cannot help the peak of the
# sweep at all, and past the knee it stops helping the off-peak parts too - the
# remaining trailing-finger dip is then a path-timing limit that the elbow
# fold, the wrist pitch and the corner detour in gesture_pose address instead.
# The knee is LIFT_ROLL_KNEE; gesture_pose adds one named policy buffer past it
# (see gesture_pose's CLEARANCE_LIFT_ROLL comment). tests/test_reach_trajectory
# and tests/reach_clearance re-check the composed result (groove included).
# LIFT_SEARCH_SAMPLES covers the whole attack and release; 81 is dense enough
# that halving the step does not move the knee.
LIFT_SEARCH_TOL = 0.003
LIFT_SEARCH_SAMPLES = 81
LIFT_SEARCH_MAX = 40.0


def main() -> None:
    import numpy as np
    from panda3d.core import Point3, Vec3, loadPrcFileData

    loadPrcFileData("", "window-type none")
    loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()

    from ghost_in_the_deck.animation.arm_ik import solve_elbow
    from ghost_in_the_deck.animation.rig import AvatarRig
    from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS

    rig = AvatarRig(ROOT / "assets" / "avatar" / "ghost_test.bam", parent=base.render)

    def measure(joint: str) -> Point3:
        return rig.expose(joint).getPos(base.render)

    def jacobian(joint: str, probe: str, hpr: list[float]):
        cols = []
        for axis in range(3):
            plus, minus = list(hpr), list(hpr)
            plus[axis] += EPS
            minus[axis] -= EPS
            rig.set_offset(joint, heading=plus[0], pitch=plus[1], roll=plus[2])
            rig.force_update()
            p_plus = measure(probe)
            rig.set_offset(joint, heading=minus[0], pitch=minus[1], roll=minus[2])
            rig.force_update()
            p_minus = measure(probe)
            cols.append((p_plus - p_minus) / (2 * EPS))
        rig.set_offset(joint, heading=hpr[0], pitch=hpr[1], roll=hpr[2])
        rig.force_update()
        return cols

    def solve_joint(joint: str, probe: str, target: Point3, limit, iterations=ITERATIONS, max_step=MAX_STEP):
        hpr = [0.0, 0.0, 0.0]
        for _ in range(iterations):
            rig.set_offset(joint, heading=hpr[0], pitch=hpr[1], roll=hpr[2])
            rig.force_update()
            error = target - measure(probe)
            if error.length() < 0.0005:
                break
            cols = jacobian(joint, probe, hpr)
            J = np.array([[c.x for c in cols], [c.y for c in cols], [c.z for c in cols]])
            e = np.array([error.x, error.y, error.z])
            delta = J.T @ np.linalg.solve(J @ J.T + LAMBDA * np.eye(3), e)
            step_len = float(np.linalg.norm(delta))
            if step_len > max_step:
                delta *= max_step / step_len
            hpr = [max(-limit[i], min(limit[i], hpr[i] + delta[i])) for i in range(3)]
        rig.set_offset(joint, heading=hpr[0], pitch=hpr[1], roll=hpr[2])
        rig.force_update()
        final = measure(probe)
        return hpr, (target - final).length()

    upper_limit = AvatarRig.LIMITS["upperarm_l"]
    lower_limit = AvatarRig.LIMITS["lowerarm_l"]

    # ------------------------------------------------------------- IK_REACH
    print(f"{'pose':<12}{'side':<6}{'joint':<14}{'heading':>10}{'pitch':>10}{'roll':>10}")
    ik_reach: dict[str, dict[str, tuple[float, float, float]]] = {}
    for side in ("l", "r"):
        rig.reset()
        rig.set_offset(f"clavicle_{side}", pitch=CLAVICLE_REACH_PITCH)
        rig.force_update()

        shoulder = measure(f"upperarm_{side}")
        upper_len = (measure(f"lowerarm_{side}") - shoulder).length()
        fore_len = (measure(f"hand_{side}") - measure(f"lowerarm_{side}")).length()

        outward = Vec3(1, 0, 0) if side == "l" else Vec3(-1, 0, 0)
        pole = Vec3(0, 0, -1) * 0.7 + outward * 0.5 + Vec3(0, -1, 0) * 0.2
        pole.normalize()

        target = Point3(*DEFAULT_TARGETS.controls_for(side))
        # Aim a few centimetres *above* the control point, not at it: the DJ's
        # hand hovers over the controls, it does not rest on them, and the
        # fifteen un-posed finger joints splay a few centimetres below the
        # wrist - aiming the wrist exactly at control height leaves the pinky
        # grazing the tabletop once a real groove sways underneath (Phase 1B.2
        # F1). REACH_HOVER_HEIGHT is small enough that the wrist still lands
        # inside the 5-8 cm this project treats as "reads as reaching".
        # REACH_HOVER_BACK nudges the aim toward the avatar (+Y), away from the
        # deck platter that sits further forward than the control row - the
        # outer fingers were grazing the platter's edge at the hold.
        hover_target = Point3(target.x, target.y + REACH_HOVER_BACK,
                              target.z + REACH_HOVER_HEIGHT)
        aim = Point3(shoulder + (hover_target - shoulder) * REACH_AIM_SHRINK)

        elbow_solution = solve_elbow(tuple(shoulder), tuple(aim), upper_len, fore_len, tuple(pole))
        elbow_target = Point3(*elbow_solution.elbow)

        upper_hpr, upper_err = solve_joint(f"upperarm_{side}", f"lowerarm_{side}", elbow_target, upper_limit)
        lower_hpr, wrist_err = solve_joint(f"lowerarm_{side}", f"hand_{side}", aim, lower_limit)

        wrist_final = measure(f"hand_{side}")
        elbow_final = measure(f"lowerarm_{side}")
        bend = 180.0 - math.degrees(
            math.acos(
                max(-1.0, min(1.0, (elbow_final - shoulder).normalized().dot(
                    (wrist_final - elbow_final).normalized()
                )))
            )
        )

        ik_reach[side] = {
            "upperarm": (round(upper_hpr[0], 3), round(upper_hpr[1], 3), round(upper_hpr[2], 3)),
            "lowerarm": (round(lower_hpr[0], 3), round(lower_hpr[1], 3), round(lower_hpr[2], 3)),
        }
        print(f"{'IK_REACH':<12}{side:<6}{'upperarm':<14}{upper_hpr[0]:>10.3f}{upper_hpr[1]:>10.3f}{upper_hpr[2]:>10.3f}")
        print(f"{'IK_REACH':<12}{side:<6}{'lowerarm':<14}{lower_hpr[0]:>10.3f}{lower_hpr[1]:>10.3f}{lower_hpr[2]:>10.3f}")
        print(
            f"            elbow solve err={upper_err*100:.2f} cm  "
            f"wrist-to-aim err={wrist_err*100:.2f} cm  "
            f"wrist-to-TRUE-target={(target - wrist_final).length()*100:.2f} cm  "
            f"elbow_bend={bend:.1f} deg"
        )
        print()

    # -------------------------------------------------------- IK_CLEARANCE
    # See the module docstring: not an independent IK solve. The upperarm offset
    # is reused verbatim from IK_REACH; the forearm is folded up by the fixed
    # CLEARANCE_ELBOW_PITCH so the clearance-pose hand clears the table without
    # the shoulder having to swing it out over the near edge.
    from ghost_in_the_deck.animation.gesture_pose import CLEARANCE_ELBOW_PITCH

    ik_clearance = {
        side: {"upperarm": ik_reach[side]["upperarm"],
               "lowerarm": (0.0, CLEARANCE_ELBOW_PITCH, 0.0)}
        for side in ("l", "r")
    }
    print("IK_CLEARANCE derived from IK_REACH's own upperarm (forearm folded up):")
    for side in ("l", "r"):
        print(f"  {side}: upperarm={ik_clearance[side]['upperarm']}  lowerarm={ik_clearance[side]['lowerarm']}")
    print()

    # ------------------------------------------- _diagnose_independent_clearance_target
    # Reproduces the negative result the module docstring describes, so this
    # script is also the record of *why* IK_CLEARANCE isn't solved that way.
    # Skipped by default - it exists for someone re-verifying the finding,
    # not for routine calibration - since it takes several minutes.
    RUN_DIAGNOSTIC = False
    if RUN_DIAGNOSTIC:
        _diagnose_independent_clearance_target(
            rig, measure, solve_joint, solve_elbow, DEFAULT_TARGETS, upper_limit, lower_limit
        )

    # -------------------------------------------------------- lift-roll search
    from ghost_in_the_deck.scene.workstation import build_workstation

    from ghost_in_the_deck.animation.gesture_pose import COMPOSED_MARGIN_BUFFER_ROLL

    workstation = build_workstation(base.render)
    furniture = _furniture_boxes(workstation)
    lift_roll_knee = _search_lift_roll(rig, furniture)
    lift_roll = round(lift_roll_knee + COMPOSED_MARGIN_BUFFER_ROLL, 1)
    print(
        f"LIFT_ROLL_KNEE search (isolated composed pose, all 16 hand joints per side, "
        f"diminishing-returns knee): {lift_roll_knee:.1f}\n"
        f"CLEARANCE_LIFT_ROLL = knee + COMPOSED_MARGIN_BUFFER_ROLL "
        f"({COMPOSED_MARGIN_BUFFER_ROLL:+.1f}) = {lift_roll:.1f}  "
        f"- tests/reach_trajectory.py and tests/reach_clearance.py re-check it "
        f"with the groove composed on top."
    )
    print()

    rig.reset()

    print()
    print("--- gesture_pose.py constants ---")
    print("IK_REACH = {")
    for side in ("l", "r"):
        print(f'    "{side}": {{')
        print(f'        "upperarm": {ik_reach[side]["upperarm"]!r},')
        print(f'        "lowerarm": {ik_reach[side]["lowerarm"]!r},')
        print("    },")
    print("}")
    print()
    print("IK_CLEARANCE = {")
    for side in ("l", "r"):
        print(f'    "{side}": {{')
        print(f'        "upperarm": IK_REACH["{side}"]["upperarm"],')
        print(f'        "lowerarm": {ik_clearance[side]["lowerarm"]!r},')
        print("    },")
    print("}")
    print()
    print(f"LIFT_ROLL_KNEE = {lift_roll_knee:.1f}")
    print(f"CLEARANCE_LIFT_ROLL = LIFT_ROLL_KNEE + COMPOSED_MARGIN_BUFFER_ROLL   # {lift_roll:.1f}")

    # A machine-readable summary on its own line, so tests/test_reach_trajectory.py
    # can check this script still reproduces the committed constants without
    # parsing the human-readable output above.
    import json

    print("JSON:" + json.dumps({
        "ik_reach": {side: {k: list(v) for k, v in ik_reach[side].items()} for side in ("l", "r")},
        "ik_clearance_lowerarm": {side: list(ik_clearance[side]["lowerarm"]) for side in ("l", "r")},
        "lift_roll_knee": lift_roll_knee,
        "clearance_lift_roll": lift_roll,
    }))


def _hand_joints(side: str) -> list[str]:
    return [f"hand_{side}"] + [
        f"{finger}_0{seg}_{side}"
        for finger in ("index", "middle", "pinky", "ring", "thumb")
        for seg in (1, 2, 3)
    ]


def _furniture_boxes(workstation):
    """Axis-aligned world bounds of every built node that stands below control
    height - the tabletop, legs, deck platters and their bases. These are the
    structures a reach only ever passes over; the search keeps the hand off
    them. (Boxes, not the round platters' true shape - a conservative simplification
    for a calibration lower bound; tests/reach_clearance.py measures the real shapes.)"""
    boxes = []

    def walk(node_path):
        if node_path.node().getType().getName() == "GeomNode":
            lo, hi = node_path.getTightBounds()
            if hi.z < 0.995:
                boxes.append(((lo.x, lo.y, lo.z), (hi.x, hi.y, hi.z)))
        for child in node_path.getChildren():
            walk(child)

    walk(workstation)
    return boxes


def _worst_box_margin(x, y, z, boxes) -> float:
    best = float("inf")
    for lo, hi in boxes:
        dx = max(lo[0] - x, 0.0, x - hi[0])
        dy = max(lo[1] - y, 0.0, y - hi[1])
        dz = max(lo[2] - z, 0.0, z - hi[2])
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            m = -min(x - lo[0], hi[0] - x, y - lo[1], hi[1] - y, z - lo[2], hi[2] - z)
        else:
            m = math.sqrt(dx * dx + dy * dy + dz * dz)
        best = min(best, m)
    return best


def _search_lift_roll(rig, furniture) -> float:
    """The lift-roll magnitude at the point of diminishing returns (see LIFT_SEARCH_TOL).

    Composes the *real* runtime reach pose - exactly what
    ``gesture_pose.pose_offsets`` produces for a full-strength ``hand_to_deck``
    at each progress, every term (elbow fold, wrist pitch, corner detour,
    trapezoid lift) included - minus only the groove, then measures every one
    of the sixteen hand and finger joints per side (F1: the wrist clears but
    the fingers, ~160 mm past it, did not) against the built workstation
    furniture, over the whole attack and release. Both sides, because the pose
    is not a perfect mirror once the wrist joint is in play. Monkeypatches
    ``gesture_pose.CLEARANCE_LIFT_ROLL`` to test a candidate - this is a
    calibration script, and that constant is the one thing it is deriving.
    """
    from ghost_in_the_deck.animation import gesture_pose
    from ghost_in_the_deck.animation.dj_behavior import DJActionState
    from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS

    render = rig.actor.getParent()
    probes = {name: rig.expose(name) for side in ("l", "r") for name in _hand_joints(side)}
    progresses = [(i + 1) / (LIFT_SEARCH_SAMPLES + 1) for i in range(LIFT_SEARCH_SAMPLES)]

    def worst_margin(magnitude: float) -> float:
        gesture_pose.CLEARANCE_LIFT_ROLL = -magnitude
        worst = float("inf")
        for side in ("l", "r"):
            for progress in progresses:
                state = DJActionState(0.0, "hand_to_deck", progress, 1.0, side, 1.0)
                rig.reset()
                for name, (h, p, r) in gesture_pose.pose_offsets(state, DEFAULT_TARGETS).items():
                    rig.set_offset(name, heading=h, pitch=p, roll=r)
                rig.force_update()
                for name in _hand_joints(side):
                    pos = probes[name].getPos(render)
                    worst = min(worst, _worst_box_margin(pos.x, pos.y, pos.z, furniture))
        return worst

    candidates = [10.0 + i for i in range(int(LIFT_SEARCH_MAX - 10.0) + 1)]
    margins = {m: worst_margin(m) for m in candidates}
    best = max(margins.values())
    knee = next(m for m in candidates if margins[m] >= best - LIFT_SEARCH_TOL)
    gesture_pose.CLEARANCE_LIFT_ROLL = -knee
    return -knee


def _diagnose_independent_clearance_target(rig, measure, solve_joint, solve_elbow, targets, upper_limit, lower_limit):
    """Reproduces Phase 1B.2's negative result for the record. See the module docstring."""
    from panda3d.core import Point3, Vec3

    side = "l"
    rig.reset()
    rig.set_offset(f"clavicle_{side}", pitch=CLAVICLE_REACH_PITCH)
    rig.force_update()
    shoulder = measure(f"upperarm_{side}")
    upper_len = (measure(f"lowerarm_{side}") - shoulder).length()
    fore_len = (measure(f"hand_{side}") - measure(f"lowerarm_{side}")).length()

    target = Point3(0.204, -0.139, 1.15)   # an example independent hover point
    pole = Vec3(0, 0, -1) * 0.7 + Vec3(1, 0, 0) * 0.5 + Vec3(0, -1, 0) * 0.2
    pole.normalize()
    elbow_solution = solve_elbow(tuple(shoulder), tuple(target), upper_len, fore_len, tuple(pole))
    elbow_target = Point3(*elbow_solution.elbow)

    big_limit = (90.0, 90.0, 90.0)
    upper_hpr, upper_err = solve_joint(f"upperarm_{side}", f"lowerarm_{side}", elbow_target, big_limit, iterations=900, max_step=4.0)
    lower_hpr, wrist_err = solve_joint(f"lowerarm_{side}", f"hand_{side}", target, big_limit, iterations=900, max_step=4.0)
    print(
        f"diagnostic: independent target {tuple(target)} -> "
        f"upper_err={upper_err*100:.2f}cm wrist_err={wrist_err*100:.2f}cm "
        f"(expected: wrist_err far from zero despite a well-placed elbow - see module docstring)"
    )


if __name__ == "__main__":
    main()
