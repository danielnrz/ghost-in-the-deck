"""Derive the hand_to_deck IK joint offsets from the real, built avatar.

    PYTHONPATH=src .venv/bin/python scripts/solve_arm_ik.py

This is a one-time calibration, not something run at play time. It exists so
the constants in ``gesture_pose.py`` (``IK_REACH``) are reproducible and can be
regenerated if the avatar asset ever changes - re-run this script and update
the constants with its output.

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

Two calibration choices worth recording:

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
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AIM_SHRINK = 0.90   # see module docstring
EPS = 0.5           # degrees, finite-difference step for the Jacobian
MAX_STEP = 4.0      # degrees, per-iteration step cap
LAMBDA = 1e-3        # Tikhonov damping
ITERATIONS = 200
CLAVICLE_REACH_PITCH = -3.0   # existing small forward droop, unchanged


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

    def solve_joint(joint: str, probe: str, target: Point3, limit):
        hpr = [0.0, 0.0, 0.0]
        for _ in range(ITERATIONS):
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
            if step_len > MAX_STEP:
                delta *= MAX_STEP / step_len
            hpr = [max(-limit[i], min(limit[i], hpr[i] + delta[i])) for i in range(3)]
        rig.set_offset(joint, heading=hpr[0], pitch=hpr[1], roll=hpr[2])
        rig.force_update()
        final = measure(probe)
        return hpr, (target - final).length()

    upper_limit = AvatarRig.LIMITS["upperarm_l"]
    lower_limit = AvatarRig.LIMITS["lowerarm_l"]

    print(f"{'side':<6}{'joint':<14}{'heading':>10}{'pitch':>10}{'roll':>10}")
    for side in ("l", "r"):
        rig.reset()
        rig.set_offset(f"clavicle_{side}", pitch=CLAVICLE_REACH_PITCH)
        rig.force_update()

        shoulder = measure(f"upperarm_{side}")
        upper_len = (measure(f"lowerarm_{side}") - shoulder).length()
        fore_len = (measure(f"hand_{side}") - measure(f"lowerarm_{side}")).length()

        target_name = "left_controls" if side == "l" else "right_controls"
        target = Point3(*getattr(DEFAULT_TARGETS, target_name))
        aim = Point3(shoulder + (target - shoulder) * AIM_SHRINK)

        outward = Vec3(1, 0, 0) if side == "l" else Vec3(-1, 0, 0)
        pole = Vec3(0, 0, -1) * 0.7 + outward * 0.5 + Vec3(0, -1, 0) * 0.2
        pole.normalize()

        elbow_solution = solve_elbow(
            tuple(shoulder), tuple(aim), upper_len, fore_len, tuple(pole)
        )
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

        print(f"{side:<6}{'upperarm':<14}{upper_hpr[0]:>10.3f}{upper_hpr[1]:>10.3f}{upper_hpr[2]:>10.3f}")
        print(f"{side:<6}{'lowerarm':<14}{lower_hpr[0]:>10.3f}{lower_hpr[1]:>10.3f}{lower_hpr[2]:>10.3f}")
        print(
            f"      elbow solve err={upper_err*100:.2f} cm  "
            f"wrist-to-aim err={wrist_err*100:.2f} cm  "
            f"wrist-to-TRUE-target={(target - wrist_final).length()*100:.2f} cm  "
            f"elbow_bend={bend:.1f} deg"
        )
        print()

    rig.reset()


if __name__ == "__main__":
    main()
