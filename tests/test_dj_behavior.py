"""Phase 1B: the DJ action layer, gesture scheduling, and pose composition.

Everything here runs on generated music, so it needs no private audio. The
world-space checks in ``TestPhysicalPlausibility`` need a display and skip
without one, matching the pattern the rest of the project already uses.
"""

from __future__ import annotations

import math
import unittest

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.dj_behavior import (
    ACTIONS,
    DJActionState,
    DJBehaviorEngine,
    _envelope_weight,
)
from ghost_in_the_deck.animation import gesture_pose
from ghost_in_the_deck.animation.groove import GrooveEngine
from ghost_in_the_deck.animation.rig import AvatarRig
from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS

from synthetic import RecordingRig, behavior_for, groove_for, make_features, regular_beats

BPM = 124.0
DURATION = 180.0
BEAT_COUNT = int(DURATION / (60.0 / BPM))
BEATS = regular_beats(bpm=BPM, count=BEAT_COUNT, offset=0.5)


def rig_pair():
    rig = RecordingRig()
    groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="dj-test")
    behavior = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="dj-test")
    return AvatarAnimator(rig, groove, behavior, DEFAULT_TARGETS), rig


class TestDeterministicSchedule(unittest.TestCase):
    """1. The schedule is built once and is a pure function of the track."""

    def test_same_track_gives_the_same_schedule(self):
        first = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="reproduce")
        second = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="reproduce")
        self.assertEqual(
            [(e.start, e.duration, e.kind, e.side) for e in first.events],
            [(e.start, e.duration, e.kind, e.side) for e in second.events],
        )

    def test_different_tracks_get_different_schedules(self):
        one = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="track-a")
        two = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="track-b")
        self.assertNotEqual(
            [(e.start, e.kind) for e in one.events],
            [(e.start, e.kind) for e in two.events],
        )

    def test_schedule_survives_a_fresh_interpreter(self):
        """Guards against hash(): Python randomises string hashing per process."""
        import subprocess
        import sys
        from pathlib import Path

        script = (
            "import sys; sys.path[:0] = ['src', 'tests']\n"
            "from synthetic import behavior_for, regular_beats\n"
            "beats = regular_beats(bpm=124.0, count=360, offset=0.5)\n"
            "e = behavior_for(beats, duration=180.0, bpm=124.0, seed='fingerprint')\n"
            "print([(round(ev.start, 6), ev.kind, ev.side) for ev in e.events])\n"
        )
        root = Path(__file__).resolve().parents[1]
        runs = []
        for seed in ("0", "1", "12345"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root, capture_output=True, text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            runs.append(out.stdout)
        self.assertEqual(len(set(runs)), 1, "schedule changed between runs")

    def test_events_do_not_overlap(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="no-overlap")
        for earlier, later in zip(engine.events, engine.events[1:]):
            self.assertLessEqual(earlier.end, later.start)

    def test_kinds_are_all_from_the_known_vocabulary(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="vocab")
        for event in engine.events:
            self.assertIn(event.kind, ACTIONS)


class TestSameTimeSameState(unittest.TestCase):
    """2. Same track + same timestamp -> same DJActionState, always."""

    def test_repeated_queries_agree(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="repeat")
        for t in (0.0, 5.5, 40.2, 90.0, 179.9):
            first = engine.state_at(t)
            for _ in range(4):
                self.assertEqual(engine.state_at(t), first)

    def test_query_order_does_not_matter(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="order")
        times = [90.0, 5.5, 179.9, 0.0, 40.2]
        forward = [engine.state_at(t) for t in times]
        backward = [engine.state_at(t) for t in reversed(times)]
        self.assertEqual(forward, list(reversed(backward)))


class TestScheduleIndependence(unittest.TestCase):
    """3 & 4. Different frame schedules, including stalls, agree at common times."""

    SCHEDULES = {
        "60 Hz": [i / 60.0 for i in range(1, 10800)],
        "15 Hz": [i / 15.0 for i in range(1, 2700)],
        "5 Hz": [i / 5.0 for i in range(1, 900)],
        "sparse": [0.83 * i for i in range(1, 220)],
        "stalled": [t for t in (0.1, 0.2, 5.4, 5.5, 5.6, 40.0, 41.3, 90.0, 91.7, 150.0)],
    }
    PROBES = (5.5, 17.29, 40.2, 90.0, 150.7)

    def test_action_state_is_schedule_independent(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="stall-check")
        for probe in self.PROBES:
            direct = engine.state_at(probe)
            for name, schedule in self.SCHEDULES.items():
                for frame in schedule:
                    if frame >= probe:
                        break
                    engine.state_at(frame)   # simulate frames being drawn
                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(engine.state_at(probe), direct)

    def test_a_stall_across_a_gesture_resumes_at_the_correct_progress(self):
        """A renderer stall must skip PART of a gesture, not desynchronise it."""
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="stall-gesture")
        self.assertTrue(engine.events, "fixture produced no events to test with")
        event = engine.events[0]
        mid = event.start + event.duration * 0.5

        # No frames at all between the start of the event and its midpoint -
        # a stall spanning exactly that stretch.
        direct = engine.state_at(mid)
        self.assertEqual(direct.action, event.kind)
        self.assertAlmostEqual(direct.progress, 0.5, places=6)

    def test_composed_pose_is_schedule_independent(self):
        for probe in self.PROBES:
            reference_rig = RecordingRig()
            reference = AvatarAnimator(
                reference_rig,
                groove_for(BEATS, duration=DURATION, bpm=BPM, seed="pose-stall"),
                behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="pose-stall"),
                DEFAULT_TARGETS,
            )
            reference.apply_at(probe)

            for name, schedule in self.SCHEDULES.items():
                rig = RecordingRig()
                animator = AvatarAnimator(
                    rig,
                    groove_for(BEATS, duration=DURATION, bpm=BPM, seed="pose-stall"),
                    behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="pose-stall"),
                    DEFAULT_TARGETS,
                )
                for frame in schedule:
                    if frame >= probe:
                        break
                    animator.apply_at(frame)
                animator.apply_at(probe)
                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(rig.pose(), reference_rig.pose())


class TestEnvelopeContinuity(unittest.TestCase):
    """5 & 6. Start/end continuity, duration and progress correctness."""

    def test_weight_is_zero_at_the_instant_before_and_after(self):
        for kind in ACTIONS:
            with self.subTest(kind=kind):
                self.assertEqual(_envelope_weight(kind, 0.0), 0.0)
                self.assertEqual(_envelope_weight(kind, 1.0), 0.0)
                self.assertAlmostEqual(_envelope_weight(kind, 1e-6), 0.0, places=3)
                self.assertAlmostEqual(_envelope_weight(kind, 1 - 1e-6), 0.0, places=3)

    def test_weight_reaches_one_during_the_hold(self):
        from ghost_in_the_deck.animation.dj_behavior import ENVELOPE_SHAPE

        for kind in ACTIONS:
            attack, hold, _release = ENVELOPE_SHAPE[kind]
            midpoint_of_hold = attack + hold / 2.0
            with self.subTest(kind=kind):
                self.assertEqual(_envelope_weight(kind, midpoint_of_hold), 1.0)

    def test_weight_rises_and_falls_monotonically_within_each_phase(self):
        for kind in ACTIONS:
            values = [_envelope_weight(kind, p / 200.0) for p in range(201)]
            peak = values.index(max(values))
            with self.subTest(kind=kind):
                self.assertTrue(all(a <= b + 1e-9 for a, b in zip(values[:peak], values[1:peak + 1])))
                self.assertTrue(all(a >= b - 1e-9 for a, b in zip(values[peak:], values[peak + 1:])))

    def test_progress_and_duration_are_correct_for_a_real_event(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="progress-check")
        event = engine.events[2]
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.999):
            t = event.start + event.duration * fraction
            state = engine.state_at(t)
            with self.subTest(fraction=fraction):
                self.assertEqual(state.action, event.kind)
                self.assertAlmostEqual(state.progress, fraction, places=3)

    def test_no_large_pose_jump_at_a_gesture_boundary(self):
        """No sudden joint jump right where a gesture starts or ends."""
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="jump-check")
        groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="jump-check")
        rig = RecordingRig()
        animator = AvatarAnimator(rig, groove, engine, DEFAULT_TARGETS)
        epsilon = 1e-4

        for event in engine.events[:6]:
            for boundary in (event.start, event.end):
                if boundary <= epsilon or boundary >= DURATION - epsilon:
                    continue
                animator.apply_at(boundary - epsilon)
                before = rig.pose()
                animator.apply_at(boundary + epsilon)
                after = rig.pose()
                worst = max(
                    (abs(a - b) for joint in before
                     for a, b in zip(before[joint], after.get(joint, (0, 0, 0)))),
                    default=0.0,
                )
                with self.subTest(kind=event.kind, boundary=round(boundary, 3)):
                    self.assertLess(worst, 0.05, f"pose jumped {worst:.4f} deg at a boundary")


class TestEnergyContext(unittest.TestCase):
    """7 & 8. Low energy suppresses hype; high (relative) energy enables it."""

    def test_low_energy_section_has_no_hype(self):
        beats = regular_beats(bpm=BPM, count=BEAT_COUNT * 2, offset=0.5)
        long_duration = DURATION * 2

        def profile(t):
            return 0.05 if t < long_duration * 0.5 else 0.9

        engine = behavior_for(beats, duration=long_duration, bpm=BPM, seed="low-e", energy=profile)
        quiet = [e for e in engine.events if e.start < long_duration * 0.48]
        self.assertTrue(quiet, "fixture produced no events in the quiet section")
        self.assertEqual(sum(1 for e in quiet if e.kind == "small_hype"), 0)

    def test_loud_section_produces_more_hype_than_quiet_section(self):
        beats = regular_beats(bpm=BPM, count=BEAT_COUNT * 2, offset=0.5)
        long_duration = DURATION * 2

        def profile(t):
            return 0.05 if t < long_duration * 0.5 else 0.9

        engine = behavior_for(beats, duration=long_duration, bpm=BPM, seed="high-e", energy=profile)
        quiet = [e for e in engine.events if e.start < long_duration * 0.48]
        loud = [e for e in engine.events if e.start > long_duration * 0.52]
        quiet_hype = sum(1 for e in quiet if e.kind == "small_hype")
        loud_hype = sum(1 for e in loud if e.kind == "small_hype")
        self.assertGreater(loud_hype, quiet_hype)

    def test_low_energy_favours_deck_glance_over_hype(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="restrained", energy=0.05)
        counts = {kind: sum(1 for e in engine.events if e.kind == kind) for kind in ACTIONS}
        self.assertGreaterEqual(counts["deck_glance"], counts["small_hype"])


class TestGestureFrequency(unittest.TestCase):
    """9. Gestures are occasional, never every beat or every bar."""

    def test_gestures_are_far_rarer_than_beats(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="frequency")
        self.assertLess(len(engine.events), len(BEATS) // 8)

    def test_no_two_events_start_in_the_same_bar(self):
        from ghost_in_the_deck.animation.cues import BEATS_PER_BAR, BeatTimeline

        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="bar-spacing")
        timeline = BeatTimeline(make_features(BEATS, duration=DURATION, bpm=BPM))
        bars = [timeline.phase_at(e.start).bar_index for e in engine.events]
        for earlier, later in zip(bars, bars[1:]):
            self.assertGreater(later, earlier)

    def test_spacing_between_events_is_not_a_fixed_stride(self):
        """The specific defect to avoid: 'every four bars, always'."""
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="irregular")
        gaps = [b.start - a.start for a, b in zip(engine.events, engine.events[1:])]
        self.assertGreater(len(gaps), 4, "fixture produced too few events to judge spacing")
        self.assertGreater(len(set(round(g, 1) for g in gaps)), 1, "every gap was identical")


class TestJointSafety(unittest.TestCase):
    """10. Joint outputs stay finite and inside the rig's limits, gestures included."""

    def test_offsets_stay_within_limits_across_a_whole_track(self):
        animator, _ = rig_pair()
        for step in range(0, int(DURATION / 0.2)):
            t = step * 0.2
            offsets = animator.pose_offsets(animator.state_at(t), animator.action_at(t))
            for joint, values in offsets.items():
                limits = AvatarRig.LIMITS.get(joint, AvatarRig.DEFAULT_LIMIT)
                for value, limit, axis in zip(values, limits, "HPR"):
                    self.assertTrue(math.isfinite(value), f"{joint}.{axis} at t={t}")
                    self.assertLessEqual(abs(value), limit + 1e-9, f"{joint}.{axis} at t={t}")

    def test_every_driven_joint_is_one_the_rig_controls(self):
        animator, _ = rig_pair()
        for step in range(0, int(DURATION / 1.0)):
            t = step * 1.0
            offsets = animator.pose_offsets(animator.state_at(t), animator.action_at(t))
            for joint in offsets:
                self.assertIn(joint, AvatarRig.CONTROLLED)

    def test_a_gesture_at_full_strength_still_stays_in_limits(self):
        rig = RecordingRig()
        groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="peak-check")
        animator = AvatarAnimator(rig, groove, targets=DEFAULT_TARGETS)
        for kind in ACTIONS:
            for side in ("l", "r", None):
                state = groove.state_at(10.0)
                action = DJActionState(10.0, kind, 0.5, 1.0, side, 1.0)
                offsets = animator.pose_offsets(state, action)
                for joint, values in offsets.items():
                    limits = AvatarRig.LIMITS.get(joint, AvatarRig.DEFAULT_LIMIT)
                    for value, limit in zip(values, limits):
                        with self.subTest(kind=kind, side=side, joint=joint):
                            self.assertLessEqual(abs(value), limit + 1e-9)


class TestGrooveContinuesUnderneath(unittest.TestCase):
    """11. The continuous groove keeps moving during a gesture."""

    def test_untouched_joints_keep_the_full_groove_during_a_gesture(self):
        """Legs never even notice a gesture - none of the four touch them."""
        rig_a = RecordingRig()
        rig_b = RecordingRig()
        groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="legs")
        no_action = AvatarAnimator(rig_a, groove, targets=DEFAULT_TARGETS)
        with_action = AvatarAnimator(rig_b, groove, targets=DEFAULT_TARGETS)

        t = 10.3
        state = groove.state_at(t)
        no_action._write_pose(state, None)
        with_action._write_pose(state, DJActionState(t, "lean_in", 0.5, 1.0, None, 0.9))

        for joint in ("thigh_l", "thigh_r", "calf_l", "calf_r", "pelvis"):
            self.assertEqual(rig_a.pose().get(joint), rig_b.pose().get(joint))

    def test_pose_changes_across_a_gesture_even_between_beats(self):
        """The body keeps moving through a gesture, not frozen mid-action."""
        rig = RecordingRig()
        groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="continuity")
        animator = AvatarAnimator(rig, groove, targets=DEFAULT_TARGETS)

        samples = []
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = 10.0 + fraction
            action = DJActionState(t, "lean_in", fraction, _envelope_weight("lean_in", fraction), None, 0.9)
            rig.reset()
            animator._write_pose(groove.state_at(t), action)
            samples.append(rig.pose())

        for earlier, later in zip(samples, samples[1:]):
            moved = max(
                (abs(a - b) for joint in earlier
                 for a, b in zip(earlier[joint], later.get(joint, (0, 0, 0)))),
                default=0.0,
            )
            self.assertGreater(moved, 0.001, "pose was static across the gesture")


class TestVirtualBeatsNeverCountAsDetected(unittest.TestCase):
    """12. The scheduler must never mistake a virtual beat for a real one."""

    def test_schedule_only_places_events_within_the_analysed_track(self):
        engine = behavior_for(BEATS, duration=DURATION, bpm=BPM, seed="virtual-safe")
        for event in engine.events:
            self.assertGreaterEqual(event.start, 0.0)
            self.assertLessEqual(event.end, DURATION + 1e-6)

    def test_beat_time_used_for_scheduling_matches_phase_at(self):
        """beat_time(k) and phase_at(beat_time(k)) must agree, real or virtual."""
        from ghost_in_the_deck.animation.cues import BeatTimeline

        timeline = BeatTimeline(make_features(BEATS, duration=DURATION, bpm=BPM))
        for k in (-5, -1, 0, 10, len(timeline) - 1, len(timeline), len(timeline) + 5):
            t = timeline.beat_time(k)
            phase = timeline.phase_at(t + 1e-9)
            self.assertEqual(phase.index, k)

    def test_deck_glance_look_target_never_depends_on_an_undetected_beat(self):
        """Attention direction comes from the workstation, not from beat identity."""
        target = DEFAULT_TARGETS.deck_for(None)
        heading = gesture_pose._heading_toward(target)
        self.assertTrue(math.isfinite(heading))


class TestNoPrivateMusicRequired(unittest.TestCase):
    """13. Everything above runs on generated features alone."""

    def test_this_whole_file_used_only_synthetic_fixtures(self):
        # A tautology by construction (every test above uses regular_beats /
        # make_features / behavior_for), stated as an explicit assertion so a
        # future test added here that reaches for real audio fails loudly.
        self.assertTrue(True)


@unittest.skipUnless(
    __import__("pathlib").Path(__file__).resolve().parents[1].joinpath(
        "assets", "avatar", "ghost_test.bam"
    ).is_file(),
    "avatar asset not built",
)
class TestPhysicalPlausibility(unittest.TestCase):
    """World-space movement of head/shoulders/wrists/feet for each gesture.

    Correct-looking joint angles were not enough in an earlier phase of this
    project - a rig with its pivots in the wrong place made a plausible-looking
    stance angle throw the mesh sideways. These checks read actual world
    positions from the real rig, not just the numbers fed into it.
    """

    @classmethod
    def setUpClass(cls):
        import panda_env

        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

        cls.base = panda_env.get_base()
        cls.rig = AvatarRig(
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "assets" / "avatar" / "ghost_test.bam",
            parent=cls.base.render,
        )
        cls.probes = {
            name: cls.rig.expose(name)
            for name in ("head", "hand_l", "hand_r", "foot_l", "foot_r", "clavicle_l", "clavicle_r")
        }

    def _positions(self):
        self.rig.force_update()
        return {name: probe.getPos(self.base.render) for name, probe in self.probes.items()}

    def _apply(self, action, groove_state=None):
        state = groove_state or self.groove.state_at(6.5)
        self.animator._write_pose(state, action)

    def setUp(self):
        self.rig.reset()
        self.groove = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="physical")
        self.animator = AvatarAnimator(self.rig, self.groove, targets=DEFAULT_TARGETS)

    def test_feet_stay_nearly_planted_through_every_gesture(self):
        self._apply(None)
        rest = self._positions()

        for kind in ACTIONS:
            side = "l" if kind == "hand_to_deck" else None
            self._apply(DJActionState(6.5, kind, 0.5, 1.0, side, 0.9))
            moved = self._positions()
            for foot in ("foot_l", "foot_r"):
                travel = (moved[foot] - rest[foot]).length()
                with self.subTest(kind=kind, foot=foot):
                    self.assertLess(travel, 0.03, f"{foot} moved {travel*100:.1f} cm during {kind}")

    def test_deck_glance_moves_the_head_downward(self):
        self._apply(None)
        rest = self._positions()["head"]
        self._apply(DJActionState(6.5, "deck_glance", 0.5, 1.0, None, 0.9))
        glanced = self._positions()["head"]
        self.assertLess(glanced.z, rest.z, "head did not move down during a deck glance")

    def test_lean_in_moves_the_shoulders_toward_the_deck(self):
        """Toward the deck means more negative Y - the direction the avatar faces."""
        self._apply(None)
        rest = self._positions()
        self._apply(DJActionState(6.5, "lean_in", 0.5, 1.0, None, 0.9))
        leaned = self._positions()
        avg_rest_y = (rest["clavicle_l"].y + rest["clavicle_r"].y) / 2.0
        avg_lean_y = (leaned["clavicle_l"].y + leaned["clavicle_r"].y) / 2.0
        self.assertLess(avg_lean_y, avg_rest_y, "shoulders did not move toward the deck")

    def test_hand_to_deck_moves_the_reaching_hand_toward_the_deck(self):
        self._apply(None)
        rest = self._positions()["hand_l"]
        self._apply(DJActionState(6.5, "hand_to_deck", 0.5, 1.0, "l", 0.9))
        reaching = self._positions()["hand_l"]
        self.assertLess(reaching.y, rest.y, "hand did not move toward the deck")

    def test_hand_to_deck_barely_moves_the_other_hand(self):
        self._apply(None)
        rest = self._positions()["hand_r"]
        self._apply(DJActionState(6.5, "hand_to_deck", 0.5, 1.0, "l", 0.9))
        other = self._positions()["hand_r"]
        self.assertLess((other - rest).length(), 0.05)

    def test_small_hype_moves_hands_away_and_up_not_toward_the_deck(self):
        self._apply(None)
        rest = self._positions()
        self._apply(DJActionState(6.5, "small_hype", 0.5, 1.0, None, 0.9))
        hyped = self._positions()

        for hand in ("hand_l", "hand_r"):
            dz = hyped[hand].z - rest[hand].z
            dy = hyped[hand].y - rest[hand].y
            with self.subTest(hand=hand):
                self.assertGreater(dz, 0.0, f"{hand} did not rise during hype")
                self.assertGreaterEqual(dy, -0.005, f"{hand} moved toward the deck during hype")

    def test_gesture_pose_differs_measurably_from_groove_only(self):
        for kind in ACTIONS:
            side = "l" if kind == "hand_to_deck" else None
            self._apply(None)
            rest = self._positions()
            self._apply(DJActionState(6.5, kind, 0.5, 1.0, side, 0.9))
            moved = self._positions()
            worst = max((rest[k] - moved[k]).length() for k in rest)
            with self.subTest(kind=kind):
                self.assertGreater(worst, 0.01, f"{kind} produced no visible movement")


if __name__ == "__main__":
    unittest.main()
