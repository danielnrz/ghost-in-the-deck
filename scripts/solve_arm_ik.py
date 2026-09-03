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
  from the shoulder, not aimed at exactly 100%. At 100% the target sits close
  enough to the edge of this arm's reach that the elbow ends up almost
  perfectly straight (measured: about 22 degrees of bend from dead straight).
  Real people do not reach with a locked elbow; pulling the aim in gives a
  visibly bent elbow (about 45 degrees of bend) while the wrist still lands
  within about 5 cm of the real target - a shortfall well inside the 5-8 cm
  this project treats as "close enough to read as reaching for the controls."

IK_CLEARANCE - NOT solved the same way, and that is itself a finding worth
recording. Phase 1B.2 first tried the obvious thing: aim the same two-bone
solve at an independent hover point above and in front of the tabletop's near
edge. It does not converge on this rig. Diagnosis (reproduced below in
``_diagnose_independent_clearance_target``): even with the joint-limit box
opened up to +-90 degrees in every axis and the iteration budget raised to
900, and even once the upper arm converges to within a centimetre of the
analytically-correct elbow position, no achievable lowerarm rotation - checked
both by the numeric solve and by an exhaustive coarse grid over the joint's
full Euler range - brings the wrist within 25 cm of that target, though the
target sits within 2 mm of the correct distance from the true elbow. The
forearm's local rotation range, shaped around IK_REACH's own nearly-straight
reach direction, simply does not span the very different, more sharply bent
direction a closer, higher hover point needs; this is a property of the
rig's own bone-local axes, not a solver tuning problem (raising the joint
limits and the iteration budget by 4-5x did not change the outcome).

IK_CLEARANCE is instead derived directly from IK_REACH: the *upperarm* offset
is identical to IK_REACH's own (the shoulder simply rises and swings toward
the reach direction; the whole arm's own well-converging rotation is reused
verbatim), and the *lowerarm* offset is zero - the forearm stays at rest
while the shoulder does the lifting. ``CLEARANCE_LIFT_ROLL`` is a small extra
upperarm roll, on top of that, needed only in a hump spanning the attack (and
mirrored across the release) - see ``_search_lift_roll`` below for how its
value is found and verified.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REACH_AIM_SHRINK = 0.90   # see module docstring
EPS = 0.5           # degrees, finite-difference step for the Jacobian
MAX_STEP = 4.0      # degrees, per-iteration step cap
LAMBDA = 1e-3        # Tikhonov damping
ITERATIONS = 200
CLAVICLE_REACH_PITCH = -3.0   # existing small forward droop, unchanged

# The margin (above the tabletop's own measured top) the lift-roll search
# requires before accepting a candidate value, and the number of progress
# samples it checks the whole neutral<->clearance<->target approach at. The
# margin is deliberately generous: the isolated arm's own margin at a given
# lift is not the margin a real, composed pose (groove sway on top of it)
# ends up with. This search only ever moves the isolated-arm case, so it is
# not itself the last word - the committed CLEARANCE_LIFT_ROLL in
# gesture_pose.py carries a further, separately-verified buffer past what
# this produces (currently -19.0 here vs. -23.0 committed): Phase 1B.2
# measured real composed margins (dense sampling of the actual runtime pose,
# groove included, across many energies/tempos/times/strengths - see
# tests/test_reach_trajectory.py) run 1-4 cm lower than the isolated number
# at the same lift value, and -19.0 alone left a composed worst case under a
# centimetre. If this search's own output changes after re-running against a
# new avatar asset, re-verify the composed margin with that test suite before
# just carrying its number over - do not assume the same buffer still holds.
LIFT_SAFETY_MARGIN = 0.08
LIFT_SEARCH_SAMPLES = 161


def main() -> None:
    import numpy as np
    from panda3d.core import Point3, Vec3, loadPrcFileData

    loadPrcFileData("", "window-type none")
    loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()

    from ghost_in_the_deck.animation.arm_ik import solve_elbow
    from ghost_in_the_deck.animation.rig import AvatarRig
    from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS, tabletop_bounds

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
        aim = Point3(shoulder + (target - shoulder) * REACH_AIM_SHRINK)

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
    # See the module docstring: not an independent IK solve, the upperarm
    # offset reused verbatim from IK_REACH, forearm at rest.
    ik_clearance = {
        side: {"upperarm": ik_reach[side]["upperarm"], "lowerarm": (0.0, 0.0, 0.0)}
        for side in ("l", "r")
    }
    print("IK_CLEARANCE derived from IK_REACH's own upperarm (forearm at rest):")
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
    lo, hi = tabletop_bounds()
    lift_roll = _search_lift_roll(rig, measure, ik_reach, lo, hi)
    print(
        f"CLEARANCE_LIFT_ROLL search (isolated arm, {LIFT_SAFETY_MARGIN*100:.0f} cm margin): {lift_roll:.1f}"
        " - the committed constant carries a further composed-pose buffer past this; see this"
        " script's LIFT_SAFETY_MARGIN comment and tests/test_reach_trajectory.py."
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
    print(f"CLEARANCE_LIFT_ROLL = {lift_roll:.1f}   # + composed-pose buffer; see LIFT_SAFETY_MARGIN comment above")

    # A machine-readable summary on its own line, so tests/test_reach_trajectory.py
    # can check this script still reproduces the committed constants without
    # parsing the human-readable output above.
    import json

    print("JSON:" + json.dumps({
        "ik_reach": {side: {k: list(v) for k, v in ik_reach[side].items()} for side in ("l", "r")},
        "ik_clearance_lowerarm": {side: list(ik_clearance[side]["lowerarm"]) for side in ("l", "r")},
        "clearance_lift_roll_isolated_search": lift_roll,
    }))


def _attack_pose_wrist(rig, measure, side: str, ik_reach, attack_progress: float, lift_deg: float):
    """Wrist world position at ``attack_progress`` (0=neutral, 1=final target).

    Reproduces gesture_pose.py's actual runtime shape for the attack half of
    hand_to_deck: the upperarm offset is IK_REACH's own, constant across the
    whole attack (identical in IK_CLEARANCE and IK_REACH by construction);
    only the lowerarm offset sweeps from rest to full extension, crossing the
    clearance pose at the attack's own midpoint. ``lift_deg`` is the extra
    upperarm roll at this exact point - the caller passes
    ``magnitude * sin(pi * attack_progress)``, peaking at the midpoint
    (the clearance pose itself) and zero at both ends, matching
    ``_reach_phase_weights``.
    """
    def smoothstep(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    if attack_progress < 0.5:
        w_clear, w_target = smoothstep(2.0 * attack_progress), 0.0
    else:
        w_target = smoothstep(2.0 * attack_progress - 1.0)
        w_clear = 1.0 - w_target
    total = w_clear + w_target   # == upperarm's own fraction, ramping 0 -> 1 by the midpoint

    uh, up, ur = ik_reach[side]["upperarm"]
    lh, lp, lr = ik_reach[side]["lowerarm"]
    rig.reset()
    rig.set_offset(f"clavicle_{side}", pitch=CLAVICLE_REACH_PITCH * total)
    rig.set_offset(f"upperarm_{side}", heading=uh * total, pitch=up * total, roll=ur * total + lift_deg)
    rig.set_offset(f"lowerarm_{side}", heading=lh * w_target, pitch=lp * w_target, roll=lr * w_target)
    rig.force_update()
    return measure(f"hand_{side}")


def _search_lift_roll(rig, measure, ik_reach, lo, hi) -> float:
    """The smallest lift-roll magnitude whose worst margin clears LIFT_SAFETY_MARGIN.

    Samples the whole attack (neutral -> clearance -> target), since the lift
    itself spans all of it and peaks exactly at the clearance pose, not at
    the midpoint of only the neutral<->clearance leg - see
    ``_attack_pose_wrist``. The release mirrors the attack by construction
    (``_reach_phase_weights``), so is not independently searched here; the
    reach-trajectory test suite samples it directly against the real,
    composed (groove included) runtime pose rather than trusting that
    mirror. Left side only; the right side mirrors sign
    (``_SIDE_SIGN`` in gesture_pose.py), likewise confirmed by that suite
    rather than re-searched here.
    """
    side = "l"

    def worst_margin(magnitude: float) -> float:
        worst = None
        for i in range(LIFT_SEARCH_SAMPLES):
            t = i / (LIFT_SEARCH_SAMPLES - 1)
            lift = magnitude * math.sin(math.pi * t)
            wrist = _attack_pose_wrist(rig, measure, side, ik_reach, t, lift)
            in_xy = lo[0] <= wrist.x <= hi[0] and lo[1] <= wrist.y <= hi[1]
            margin = (wrist.z - hi[2]) if in_xy else float("inf")
            if worst is None or margin < worst:
                worst = margin
        return worst

    magnitude = 0.0
    while worst_margin(-magnitude) < LIFT_SAFETY_MARGIN and magnitude < 60.0:
        magnitude += 1.0
    return -magnitude


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
