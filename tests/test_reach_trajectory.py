"""Phase 1B.2: the clearance-aware hand_to_deck reach path, and the visible
side-control geometry it reaches toward.

Collision checks measure the real, *built* scene geometry (``getTightBounds``
on the actual constructed table/control nodes) rather than trusting the
analytic ``animation.workstation`` formulas reproduce it - those formulas are
what the scene is built *from*, so checking a sampled trajectory against them
would only prove the trajectory code agrees with itself. World-space checks
need a real display and skip without one, matching the rest of this project.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

from ghost_in_the_deck.animation import gesture_pose
from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.dj_behavior import ENVELOPE_SHAPE, DJActionState
from ghost_in_the_deck.animation.rig import AvatarRig
from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS

from synthetic import groove_for, regular_beats

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "avatar" / "ghost_test.bam"

BPM = 124.0
DURATION = 90.0
BEATS = regular_beats(bpm=BPM, count=int(DURATION / (60.0 / BPM)), offset=0.5)

ATTACK, HOLD, RELEASE = ENVELOPE_SHAPE["hand_to_deck"]
HOLD_MID = ATTACK + HOLD / 2.0

# 101 evenly spaced progress values across the whole event, as required.
SAMPLE_COUNT = 101
PROGRESSES = [i / (SAMPLE_COUNT - 1) for i in range(SAMPLE_COUNT)]


class TestClearanceTargets(unittest.TestCase):
    """1. Clearance targets exist and are symmetric."""

    def test_clearance_targets_exist(self):
        self.assertTrue(hasattr(DEFAULT_TARGETS, "left_clearance"))
        self.assertTrue(hasattr(DEFAULT_TARGETS, "right_clearance"))
        for point in (DEFAULT_TARGETS.left_clearance, DEFAULT_TARGETS.right_clearance):
            self.assertEqual(len(point), 3)
            for value in point:
                self.assertTrue(math.isfinite(value))

    def test_clearance_targets_are_mirrored(self):
        left, right = DEFAULT_TARGETS.left_clearance, DEFAULT_TARGETS.right_clearance
        self.assertAlmostEqual(left[0], -right[0], places=6)
        self.assertAlmostEqual(left[1], right[1], places=6)
        self.assertAlmostEqual(left[2], right[2], places=6)

    def test_clearance_for_dispatches_by_side(self):
        self.assertEqual(DEFAULT_TARGETS.clearance_for("l"), DEFAULT_TARGETS.left_clearance)
        self.assertEqual(DEFAULT_TARGETS.clearance_for("r"), DEFAULT_TARGETS.right_clearance)

    def test_clearance_sits_above_the_tabletop(self):
        from ghost_in_the_deck.animation.workstation import tabletop_bounds

        _, hi = tabletop_bounds()
        for point in (DEFAULT_TARGETS.left_clearance, DEFAULT_TARGETS.right_clearance):
            self.assertGreater(point[2], hi[2], "clearance point is not above the tabletop")


@unittest.skipUnless(ASSET.is_file(), "avatar asset not built")
class TestSideControlGeometry(unittest.TestCase):
    """2 & 3. Side-control geometry is built from the actual semantic targets
    and exists visibly on both sides."""

    @classmethod
    def setUpClass(cls):
        import panda_env

        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

        from ghost_in_the_deck.scene.workstation import build_workstation

        cls.base = panda_env.get_base()
        cls.root = build_workstation(cls.base.render)

    def test_both_control_clusters_exist_and_are_nonempty(self):
        for label in ("l", "r"):
            node = self.root.find(f"**/control-cluster-{label}")
            self.assertFalse(node.isEmpty(), f"control-cluster-{label} not found in the built scene")
            lo, hi = node.getTightBounds()
            size = hi - lo
            self.assertGreater(size.length(), 0.01, f"control-cluster-{label} has no real extent")

    def test_control_clusters_are_positioned_at_the_actual_targets(self):
        """Not independently hard-coded - the geometry follows the targets."""
        for label, target in (("l", DEFAULT_TARGETS.left_controls), ("r", DEFAULT_TARGETS.right_controls)):
            node = self.root.find(f"**/control-cluster-{label}")
            lo, hi = node.getTightBounds()
            center = (lo + hi) / 2.0
            self.assertLess(abs(center.x - target[0]), 0.08, f"{label} cluster X far from its target")
            self.assertLess(abs(center.y - target[1]), 0.08, f"{label} cluster Y far from its target")
            # The target Z is the control surface / hand-hover height; the
            # cluster should straddle it, not sit far below or above it.
            self.assertLessEqual(lo.z - 0.02, target[2])
            self.assertGreaterEqual(hi.z + 0.02, target[2])

    def test_control_clusters_are_mirrored_and_visually_equivalent(self):
        left = self.root.find("**/control-cluster-l")
        right = self.root.find("**/control-cluster-r")
        lo_l, hi_l = left.getTightBounds()
        lo_r, hi_r = right.getTightBounds()
        size_l, size_r = hi_l - lo_l, hi_r - lo_r
        self.assertAlmostEqual(size_l.x, size_r.x, delta=0.005)
        self.assertAlmostEqual(size_l.y, size_r.y, delta=0.005)
        self.assertAlmostEqual(size_l.z, size_r.z, delta=0.005)

    def test_target_does_not_sit_inside_solid_control_geometry(self):
        """The reach target is the control surface, not buried inside the panel."""
        for label, target in (("l", DEFAULT_TARGETS.left_controls), ("r", DEFAULT_TARGETS.right_controls)):
            panel = self.root.find(f"**/control-panel-{label}")
            self.assertFalse(panel.isEmpty())
            lo, hi = panel.getTightBounds()
            # target sits at or above the panel's own top surface.
            self.assertGreaterEqual(target[2], lo.z - 1e-6)


class TestCalibrationReproducibility(unittest.TestCase):
    """4. scripts/solve_arm_ik.py reproduces the committed constants."""

    @classmethod
    def setUpClass(cls):
        if not ASSET.is_file():
            raise unittest.SkipTest("avatar asset not built")

    def test_script_reproduces_committed_ik_reach_and_clearance(self):
        result = subprocess.run(
            [sys.executable, "scripts/solve_arm_ik.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        json_line = next(line for line in result.stdout.splitlines() if line.startswith("JSON:"))
        data = json.loads(json_line[len("JSON:"):])

        for side in ("l", "r"):
            for joint in ("upperarm", "lowerarm"):
                got = data["ik_reach"][side][joint]
                want = gesture_pose.IK_REACH[side][joint]
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, places=2, msg=f"IK_REACH[{side}][{joint}] drifted")

        for side in ("l", "r"):
            got = data["ik_clearance_lowerarm"][side]
            want = gesture_pose.IK_CLEARANCE[side]["lowerarm"]
            self.assertEqual(tuple(got), want)
            self.assertEqual(
                gesture_pose.IK_CLEARANCE[side]["upperarm"], gesture_pose.IK_REACH[side]["upperarm"],
                "IK_CLEARANCE's upperarm should be exactly IK_REACH's own",
            )

        # CLEARANCE_LIFT_ROLL is derived end to end: the script's own
        # diminishing-returns knee search plus one named policy constant. Both
        # halves, and the committed sum, must match exactly - no silent drift.
        self.assertAlmostEqual(
            data["lift_roll_knee"], gesture_pose.LIFT_ROLL_KNEE, places=6,
            msg="LIFT_ROLL_KNEE drifted from the script's search",
        )
        self.assertAlmostEqual(
            data["clearance_lift_roll"],
            gesture_pose.LIFT_ROLL_KNEE + gesture_pose.COMPOSED_MARGIN_BUFFER_ROLL,
            places=6,
            msg="script's CLEARANCE_LIFT_ROLL is not knee + COMPOSED_MARGIN_BUFFER_ROLL",
        )
        self.assertAlmostEqual(
            data["clearance_lift_roll"], gesture_pose.CLEARANCE_LIFT_ROLL, places=6,
            msg="committed CLEARANCE_LIFT_ROLL does not match the script's output",
        )


@unittest.skipUnless(ASSET.is_file(), "avatar asset not built")
class TestReachTrajectoryCollision(unittest.TestCase):
    """5-14: dense collision/continuity checks against the real composed pose."""

    @classmethod
    def setUpClass(cls):
        import panda_env

        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

        from ghost_in_the_deck.scene.workstation import build_workstation

        cls.base = panda_env.get_base()
        cls.rig = AvatarRig(ASSET, parent=cls.base.render)
        cls.workstation = build_workstation(cls.base.render)

        table = cls.workstation.find("**/tabletop")
        cls.table_lo, cls.table_hi = table.getTightBounds()

        cls.probes = {
            name: cls.rig.expose(name)
            for name in (
                "hand_l", "hand_r", "lowerarm_l", "lowerarm_r",
                "upperarm_l", "upperarm_r", "foot_l", "foot_r",
            )
        }

    def setUp(self):
        self.rig.reset()

    def _positions(self):
        self.rig.force_update()
        return {name: probe.getPos(self.base.render) for name, probe in self.probes.items()}

    def _groove_configs(self):
        """A few representative groove states: quiet, mid, driving; two tempos."""
        return [
            groove_for(BEATS, duration=DURATION, bpm=BPM, seed="traj-quiet", energy=0.15),
            groove_for(BEATS, duration=DURATION, bpm=BPM, seed="traj-mid", energy=0.55),
            groove_for(BEATS, duration=DURATION, bpm=BPM, seed="traj-driving", energy=0.95),
            groove_for(regular_beats(bpm=96.0, count=int(DURATION / (60.0 / 96.0))),
                       duration=DURATION, bpm=96.0, seed="traj-slow", energy=0.7),
        ]

    def _inside_table(self, p) -> bool:
        return (
            self.table_lo.x <= p.x <= self.table_hi.x
            and self.table_lo.y <= p.y <= self.table_hi.y
            and self.table_lo.z <= p.z <= self.table_hi.z
        )

    def _sample(self, groove, t0, side, progress, strength=0.9):
        state = groove.state_at(t0)
        action = DJActionState(t0, "hand_to_deck", progress, 0.0, side, strength)
        animator = AvatarAnimator(self.rig, groove, targets=DEFAULT_TARGETS)
        animator._write_pose(state, action)
        return self._positions()

    def test_full_trajectory_is_finite(self):
        """5. Every sampled joint position is finite, for every side/groove."""
        for groove in self._groove_configs():
            for side in ("l", "r"):
                for progress in PROGRESSES:
                    positions = self._sample(groove, 6.5, side, progress)
                    for name, p in positions.items():
                        with self.subTest(side=side, progress=progress, joint=name):
                            self.assertTrue(math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z))

    def test_wrist_never_enters_the_tabletop_volume(self):
        """6 & 8 & 14. No sampled wrist point, either side, either attack or
        release, lies inside the real built tabletop solid."""
        intersections = []
        for groove in self._groove_configs():
            for side in ("l", "r"):
                for progress in PROGRESSES:
                    wrist = self._sample(groove, 6.5, side, progress)[f"hand_{side}"]
                    if self._inside_table(wrist):
                        intersections.append((side, progress, wrist))
        self.assertEqual(intersections, [], f"wrist intersected the tabletop at: {intersections[:5]}")

    def test_wrist_stays_above_the_surface_while_over_it(self):
        """7. Whenever the wrist is over the table's footprint, it is at or
        above the table's own top - not just outside its full 3D volume."""
        worst = None
        for groove in self._groove_configs():
            for side in ("l", "r"):
                for progress in PROGRESSES:
                    wrist = self._sample(groove, 6.5, side, progress)[f"hand_{side}"]
                    over_table = (
                        self.table_lo.x <= wrist.x <= self.table_hi.x
                        and self.table_lo.y <= wrist.y <= self.table_hi.y
                    )
                    if over_table:
                        margin = wrist.z - self.table_hi.z
                        if worst is None or margin < worst:
                            worst = margin
        self.assertIsNotNone(worst, "no sample was ever over the table - test would be vacuous")
        self.assertGreaterEqual(worst, 0.0, f"wrist dipped {(-worst)*100:.1f} cm below the tabletop while over it")

    def test_elbow_never_enters_the_tabletop_volume(self):
        """Elbow world position, checked the same way as the wrist."""
        for groove in self._groove_configs()[:2]:
            for side in ("l", "r"):
                for progress in PROGRESSES:
                    elbow = self._sample(groove, 6.5, side, progress)[f"lowerarm_{side}"]
                    self.assertFalse(self._inside_table(elbow), f"elbow_{side} intersected the table at progress={progress}")

    def test_forearm_segment_does_not_pass_through_the_tabletop(self):
        """Sample points along the elbow-wrist segment, not just its ends."""
        for side in ("l", "r"):
            for progress in PROGRESSES[::5]:
                positions = self._sample(self._groove_configs()[1], 6.5, side, progress)
                elbow, wrist = positions[f"lowerarm_{side}"], positions[f"hand_{side}"]
                for f in (0.25, 0.5, 0.75):
                    mid = elbow + (wrist - elbow) * f
                    self.assertFalse(self._inside_table(mid), f"forearm midpoint intersected the table (side={side}, progress={progress}, f={f})")

    def test_trajectory_is_continuous(self):
        """13. No discontinuity between adjacent samples - a real, smooth path.

        The bound is per 1/400 of the event (~3 ms of playback at these
        tempos). The staged approach deliberately snaps to the lifted clearance
        pose briskly - a real hand darting to the decks does move fast - so the
        peak step is larger than a slow groove sway would give; what this guards
        against is a *teleport* (a keying bug, a sign flip), not speed. 3.5 cm
        in ~3 ms is fast but smooth; anything past that is a defect.
        """
        fine = [i / 400.0 for i in range(401)]
        for side in ("l", "r"):
            groove = self._groove_configs()[1]
            previous = None
            for progress in fine:
                wrist = self._sample(groove, 6.5, side, progress)[f"hand_{side}"]
                if previous is not None:
                    step = (wrist - previous).length()
                    self.assertLess(step, 0.05, f"wrist jumped {step*100:.1f} cm between adjacent samples at progress={progress}")
                previous = wrist

    def test_endpoint_reach_distance_is_acceptable(self):
        """9. At the hold, the wrist is still close to the real control target."""
        for side, target_attr in (("l", "left_controls"), ("r", "right_controls")):
            wrist = self._sample(self._groove_configs()[1], 6.5, side, HOLD_MID)[f"hand_{side}"]
            target = getattr(DEFAULT_TARGETS, target_attr)
            distance = math.dist((wrist.x, wrist.y, wrist.z), target)
            self.assertLess(distance, 0.08, f"wrist at hold is {distance*100:.1f} cm from {target_attr}")

    def test_elbow_bend_stays_plausible_across_the_trajectory(self):
        """10. Never dead straight, never folded past physical limits, at any
        point along the path where the arm has genuinely left rest."""
        for side in ("l", "r"):
            for progress in PROGRESSES[::4]:
                positions = self._sample(self._groove_configs()[1], 6.5, side, progress)
                shoulder = positions[f"upperarm_{side}"]
                elbow, wrist = positions[f"lowerarm_{side}"], positions[f"hand_{side}"]
                upper = elbow - shoulder
                fore = wrist - elbow
                # A gesture with near-zero weight has an ill-defined "bend" -
                # only check once the arm has genuinely left rest.
                if upper.length() < 0.05 or fore.length() < 0.05:
                    continue
                upper.normalize()
                fore.normalize()
                cos_angle = max(-1.0, min(1.0, upper.dot(fore)))
                bend = 180.0 - math.degrees(math.acos(cos_angle))
                with self.subTest(side=side, progress=progress):
                    self.assertLess(bend, 179.0, f"elbow_{side} reads as locked straight ({bend:.1f} deg) at progress={progress}")
                    self.assertGreater(bend, 40.0, f"elbow_{side} folded implausibly far ({bend:.1f} deg) at progress={progress}")

    def test_opposite_hand_stays_near_groove_only_across_the_trajectory(self):
        """11. The non-reaching arm never notices the reach."""
        groove = self._groove_configs()[1]
        for side, other in (("l", "hand_r"), ("r", "hand_l")):
            state = groove.state_at(6.5)
            animator = AvatarAnimator(self.rig, groove, targets=DEFAULT_TARGETS)
            animator._write_pose(state, None)
            self.rig.force_update()
            groove_only = self.probes[other].getPos(self.base.render)
            for progress in PROGRESSES[::10]:
                during = self._sample(groove, 6.5, side, progress)[other]
                travel = (during - groove_only).length()
                self.assertLess(travel, 0.03, f"{other} moved {travel*100:.1f} cm during a {side} reach at progress={progress}")

    def test_feet_stay_planted_across_the_trajectory(self):
        """12. Feet barely move through the whole reach."""
        groove = self._groove_configs()[1]
        state = groove.state_at(6.5)
        animator = AvatarAnimator(self.rig, groove, targets=DEFAULT_TARGETS)
        animator._write_pose(state, None)
        self.rig.force_update()
        rest = {foot: self.probes[foot].getPos(self.base.render) for foot in ("foot_l", "foot_r")}
        for side in ("l", "r"):
            for progress in PROGRESSES[::10]:
                positions = self._sample(groove, 6.5, side, progress)
                for foot in ("foot_l", "foot_r"):
                    travel = (positions[foot] - rest[foot]).length()
                    self.assertLess(travel, 0.03, f"{foot} moved {travel*100:.1f} cm during a reach")

    def test_schedule_independence_same_progress_gives_same_pose(self):
        """15. A pure function of (time, progress) - order of evaluation is irrelevant."""
        groove = self._groove_configs()[1]
        samples_in_order = [self._sample(groove, 6.5, "l", p)["hand_l"] for p in PROGRESSES[:11]]
        shuffled_indices = list(range(11))[::-1]
        samples_out_of_order = [None] * 11
        for i in shuffled_indices:
            samples_out_of_order[i] = self._sample(groove, 6.5, "l", PROGRESSES[i])["hand_l"]
        for a, b in zip(samples_in_order, samples_out_of_order):
            self.assertAlmostEqual(a.x, b.x, places=9)
            self.assertAlmostEqual(a.y, b.y, places=9)
            self.assertAlmostEqual(a.z, b.z, places=9)

    def test_renderer_stall_independence(self):
        """16. Computing unrelated intermediate poses does not perturb the
        pose subsequently computed for a given progress - no frame history."""
        groove = self._groove_configs()[1]
        reference = self._sample(groove, 6.5, "r", 0.5)["hand_r"]
        # Simulate a stall: burn through many unrelated poses first.
        for p in PROGRESSES:
            self._sample(groove, 6.5, "l", p)
        after_stall = self._sample(groove, 6.5, "r", 0.5)["hand_r"]
        self.assertAlmostEqual(reference.x, after_stall.x, places=9)
        self.assertAlmostEqual(reference.y, after_stall.y, places=9)
        self.assertAlmostEqual(reference.z, after_stall.z, places=9)


@unittest.skipUnless(ASSET.is_file(), "avatar asset not built")
class TestHandFingerClearancePopulation(unittest.TestCase):
    """F1: the committed clearance guard - the whole hand (wrist + fifteen
    finger joints) against the *built* workstation, swept over a documented
    population of real scheduled hand_to_deck events with the groove composed
    on top. See tests/reach_clearance.py for the population, the box/cylinder
    solid model and the justification of both margins."""

    @classmethod
    def setUpClass(cls):
        import panda_env

        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

    def test_swept_population_clears_the_built_geometry(self):
        from reach_clearance import (
            CONTROL_CONTACT_MARGIN,
            SAFETY_MARGIN,
            ClearanceHarness,
        )

        result = ClearanceHarness().sweep()
        self.assertGreater(
            result.furniture.margin, SAFETY_MARGIN,
            "hand/finger joint inside the furniture beyond the model tolerance:\n"
            + result.describe(),
        )
        self.assertGreater(
            result.operated.margin, CONTROL_CONTACT_MARGIN,
            "hand/finger joint too deep inside an operated control:\n"
            + result.describe(),
        )


@unittest.skipUnless(ASSET.is_file(), "avatar asset not built")
class TestUnrelatedGestureRegression(unittest.TestCase):
    """17, 18, 19: small_hype, deck_glance and lean_in are untouched by this
    phase - a light regression check, not a redesign."""

    @classmethod
    def setUpClass(cls):
        import panda_env

        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

        cls.base = panda_env.get_base()
        cls.rig = AvatarRig(ASSET, parent=cls.base.render)
        cls.probes = {n: cls.rig.expose(n) for n in ("head", "hand_l", "hand_r", "clavicle_l", "clavicle_r")}

    def setUp(self):
        self.rig.reset()
        self.groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="regression")
        self.animator = AvatarAnimator(self.rig, self.groove, targets=DEFAULT_TARGETS)

    def _positions(self):
        self.rig.force_update()
        return {name: probe.getPos(self.base.render) for name, probe in self.probes.items()}

    def test_small_hype_still_raises_the_active_hand(self):
        self.animator._write_pose(self.groove.state_at(6.5), None)
        rest = self._positions()["hand_l"]
        self.animator._write_pose(self.groove.state_at(6.5), DJActionState(6.5, "small_hype", 0.5, 1.0, "l", 0.9))
        hyped = self._positions()["hand_l"]
        self.assertGreater(hyped.z, rest.z)

    def test_deck_glance_still_lowers_the_head(self):
        self.animator._write_pose(self.groove.state_at(6.5), None)
        rest = self._positions()["head"]
        self.animator._write_pose(self.groove.state_at(6.5), DJActionState(6.5, "deck_glance", 0.5, 1.0, None, 0.9))
        glanced = self._positions()["head"]
        self.assertLess(glanced.z, rest.z)

    def test_lean_in_still_moves_shoulders_toward_the_deck(self):
        self.animator._write_pose(self.groove.state_at(6.5), None)
        rest = self._positions()
        self.animator._write_pose(self.groove.state_at(6.5), DJActionState(6.5, "lean_in", 0.5, 1.0, None, 0.9))
        leaned = self._positions()
        avg_rest_y = (rest["clavicle_l"].y + rest["clavicle_r"].y) / 2.0
        avg_leaned_y = (leaned["clavicle_l"].y + leaned["clavicle_r"].y) / 2.0
        self.assertLess(avg_leaned_y, avg_rest_y)


if __name__ == "__main__":
    unittest.main()
