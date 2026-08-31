"""The Phase 1A body language: rhythm, energy, variation and joint safety.

Everything here runs on generated music and a recording rig, so it needs no
display, no sound card and none of the user's own audio.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.cues import BEATS_PER_BAR, BeatTimeline
from ghost_in_the_deck.animation.energy import EnergyTrack
from ghost_in_the_deck.animation.groove import GrooveEngine
from ghost_in_the_deck.animation.rig import AvatarRig

from synthetic import RecordingRig, groove_for, make_features, regular_beats

BPM = 120.0
BEATS = regular_beats(bpm=BPM, count=32, offset=0.5)   # 0.5 s apart
DURATION = 20.0


def animator_for(**kwargs) -> tuple[AvatarAnimator, RecordingRig]:
    rig = RecordingRig()
    groove = groove_for(BEATS, duration=DURATION, bpm=BPM, **kwargs)
    return AvatarAnimator(rig, groove), rig


def peak_travel(animator: AvatarAnimator, times) -> float:
    """Largest joint rotation, in degrees, over a set of times."""
    worst = 0.0
    for time in times:
        offsets = animator.pose_offsets(animator.state_at(time))
        for values in offsets.values():
            worst = max(worst, max(abs(v) for v in values))
    return worst


class TestBeatPhase(unittest.TestCase):
    """Rhythm is a position between beats, not just an event at each one."""

    def setUp(self):
        self.timeline = BeatTimeline(make_features(BEATS, duration=DURATION, bpm=BPM))

    def test_phase_runs_zero_to_one_between_known_beats(self):
        self.assertAlmostEqual(self.timeline.phase_at(0.5).phase, 0.0, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(0.625).phase, 0.25, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(0.75).phase, 0.5, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(0.875).phase, 0.75, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(1.0).phase, 0.0, places=9)

    def test_phase_names_the_surrounding_beats(self):
        phase = self.timeline.phase_at(0.8)
        self.assertEqual(phase.index, 0)
        self.assertAlmostEqual(phase.previous_time, 0.5, places=9)
        self.assertAlmostEqual(phase.next_time, 1.0, places=9)
        self.assertAlmostEqual(phase.interval, 0.5, places=9)

    def test_bar_phase_spans_four_beats(self):
        self.assertEqual(BEATS_PER_BAR, 4)
        self.assertAlmostEqual(self.timeline.phase_at(0.5).bar_phase, 0.0, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(1.0).bar_phase, 0.25, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(2.0).bar_phase, 0.75, places=9)
        self.assertAlmostEqual(self.timeline.phase_at(2.5).bar_phase, 0.0, places=9)

    def test_phase_keeps_running_before_the_first_beat(self):
        """An intro should still groove rather than stand frozen."""
        early = [self.timeline.phase_at(t).phase for t in (0.05, 0.15, 0.30, 0.45)]
        self.assertEqual(len(set(round(p, 6) for p in early)), 4)
        for value in early:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_phase_keeps_running_after_the_last_beat(self):
        last = BEATS[-1]
        for offset in (0.1, 0.6, 1.4, 3.0):
            phase = self.timeline.phase_at(last + offset)
            self.assertGreaterEqual(phase.phase, 0.0)
            self.assertLessEqual(phase.phase, 1.0)
            self.assertGreaterEqual(phase.index, len(BEATS) - 1)

    def test_a_timeline_with_no_beats_still_has_a_pulse(self):
        empty = BeatTimeline(make_features([], duration=DURATION, bpm=BPM))
        phases = [empty.phase_at(t).phase for t in (0.0, 0.1, 0.2, 0.3)]
        self.assertEqual(len(set(round(p, 6) for p in phases)), 4)


class TestContinuousMovement(unittest.TestCase):
    """The body must move between beats, not only on them."""

    def test_pose_changes_while_no_beat_is_firing(self):
        # A very short accent decay so the pulse is exactly zero between beats,
        # leaving only the continuous groove to explain any movement.
        rig = RecordingRig()
        groove = groove_for(BEATS, duration=DURATION, bpm=BPM)
        groove.pulse_decay = 0.02
        groove.pulse_attack = 0.005
        animator = AvatarAnimator(rig, groove)

        quiet = [3.30, 3.36, 3.42, 3.48]
        for time in quiet:
            self.assertEqual(animator.state_at(time).pulse, 0.0)

        poses = []
        for time in quiet:
            rig.reset()
            animator.apply_at(time)
            poses.append(rig.pose())

        for earlier, later in zip(poses, poses[1:]):
            moved = max(
                abs(a - b)
                for joint in earlier
                for a, b in zip(earlier[joint], later.get(joint, (0, 0, 0)))
            )
            self.assertGreater(moved, 0.01, "body was static between beats")

    def test_several_body_regions_move(self):
        animator, _ = animator_for()
        moving = set()
        for step in range(200):
            offsets = animator.pose_offsets(animator.state_at(step * 0.05))
            for joint, values in offsets.items():
                if max(abs(v) for v in values) > 0.25:
                    moving.add(joint)

        for region in ("pelvis", "spine_02", "head", "clavicle_l", "calf_l"):
            self.assertIn(region, moving, f"{region} never moved")
        self.assertGreaterEqual(len(moving), 10)

    def test_movement_is_not_a_single_rigid_frequency(self):
        """Layers run at different rates, so the body is never one sine wave."""
        animator, _ = animator_for()
        states = [animator.state_at(t * 0.05) for t in range(400)]
        bounce = [s.bounce for s in states]
        sway = [s.sway for s in states]
        weight = [s.weight_shift for s in states]

        def zero_crossings(values, centre):
            return sum(
                1
                for a, b in zip(values, values[1:])
                if (a - centre) * (b - centre) < 0
            )

        fast = zero_crossings(bounce, 0.5)
        medium = zero_crossings(sway, 0.0)
        slow = zero_crossings(weight, 0.0)
        self.assertGreater(fast, medium, "bounce should cycle faster than sway")
        self.assertGreater(medium, slow, "sway should cycle faster than weight shift")


class TestEnergyDrivesIntensity(unittest.TestCase):
    def test_quiet_music_moves_less_than_loud_music(self):
        times = [t * 0.05 for t in range(int(DURATION / 0.05))]

        # A flat track has no dynamic range to normalise against, so both ends
        # land mid-scale; the useful comparison is a track that actually varies.
        ramp, _ = animator_for(energy=lambda t: t / DURATION)
        early = peak_travel(ramp, [t for t in times if t < 4.0])
        late = peak_travel(ramp, [t for t in times if t > 15.0])
        self.assertGreater(late, early * 1.4, "loud passage did not move more")

    def test_intensity_tracks_energy_monotonically(self):
        animator, _ = animator_for(energy=lambda t: t / DURATION)
        samples = [(animator.state_at(t).energy, animator.state_at(t).intensity)
                   for t in (1.0, 5.0, 10.0, 15.0, 19.0)]
        energies = [e for e, _ in samples]
        intensities = [i for _, i in samples]
        self.assertEqual(energies, sorted(energies))
        self.assertEqual(intensities, sorted(intensities))

    def test_movement_never_stops_entirely_when_quiet(self):
        animator, _ = animator_for(energy=lambda t: t / DURATION)
        self.assertGreater(peak_travel(animator, [t * 0.05 for t in range(40)]), 0.5)

    def test_energy_is_smoothed_not_twitchy(self):
        """One loud analysis frame must not spike the whole body."""
        def spiky(t):
            return 1.0 if abs(t - 10.0) < 0.03 else 0.1

        engine = groove_for(BEATS, duration=DURATION, bpm=BPM, energy=spiky)
        around = [engine.state_at(10.0 + d).energy for d in
                  (-0.2, -0.1, 0.0, 0.1, 0.2)]
        self.assertLess(max(around) - min(around), 0.25, f"energy jumped: {around}")

    def test_energy_uses_the_tracks_own_range(self):
        """A quiet recording still reaches full intensity in its loud passage."""
        loud = EnergyTrack(make_features(BEATS, duration=DURATION,
                                         energy=lambda t: 0.2 + 0.6 * (t / DURATION)))
        soft = EnergyTrack(make_features(BEATS, duration=DURATION,
                                         energy=lambda t: 0.02 + 0.06 * (t / DURATION)))
        self.assertGreater(loud.summary()["max"], 0.9)
        self.assertGreater(soft.summary()["max"], 0.9)
        self.assertLess(loud.summary()["min"], 0.1)
        self.assertLess(soft.summary()["min"], 0.1)


class TestDeterministicVariation(unittest.TestCase):
    def test_same_track_and_time_give_the_same_pose(self):
        first, rig_a = animator_for()
        second, rig_b = animator_for()
        for time in (2.2, 5.7, 9.1, 13.4):
            rig_a.reset(); rig_b.reset()
            first.apply_at(time)
            second.apply_at(time)
            self.assertEqual(rig_a.pose(), rig_b.pose())

    def test_variation_survives_a_fresh_interpreter(self):
        """Guards against hash(): Python randomises string hashing per process."""
        script = (
            "import sys; sys.path[:0] = ['src', 'tests']\n"
            "from synthetic import groove_for\n"
            "from ghost_in_the_deck.animation.groove import _unit\n"
            "g = groove_for(seed='fingerprint')\n"
            "v = [g.variation_for(b) for b in range(6)]\n"
            "print([round(x.emphasis, 9) for x in v])\n"
            "print([round(_unit(g.seed, b, 1), 9) for b in range(6)])\n"
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
        self.assertEqual(len(set(runs)), 1, f"variation changed between runs:\n{runs}")

    def test_different_bars_have_different_character(self):
        engine = groove_for(BEATS, duration=DURATION, bpm=BPM)
        emphases = {round(engine.variation_for(bar).emphasis, 6) for bar in range(16)}
        biases = {round(engine.variation_for(bar).head_bias, 6) for bar in range(16)}
        self.assertGreater(len(emphases), 8, "every bar had the same emphasis")
        self.assertGreater(len(biases), 8, "every bar had the same head bias")

    def test_different_tracks_dance_differently(self):
        one = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="track-one")
        two = groove_for(BEATS, duration=DURATION, bpm=BPM, seed="track-two")
        self.assertNotEqual(
            [one.variation_for(b).emphasis for b in range(8)],
            [two.variation_for(b).emphasis for b in range(8)],
        )

    def test_variation_stays_within_its_documented_range(self):
        engine = groove_for(BEATS, duration=DURATION, bpm=BPM)
        for bar in range(64):
            v = engine.variation_for(bar)
            self.assertGreaterEqual(v.emphasis, 0.85)
            self.assertLessEqual(v.emphasis, 1.15)
            for value in (v.head_bias, v.shoulder_bias, v.lead_side):
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)


class TestAsymmetry(unittest.TestCase):
    def test_the_two_sides_do_not_move_identically(self):
        animator, _ = animator_for()
        differences = []
        for step in range(400):
            offsets = animator.pose_offsets(animator.state_at(step * 0.05))
            left = offsets.get("clavicle_l", (0, 0, 0))[1]
            right = offsets.get("clavicle_r", (0, 0, 0))[1]
            differences.append(abs(left - right))
        self.assertGreater(max(differences), 0.2, "shoulders moved as a mirror pair")

    def test_weight_favours_one_leg_at_a_time(self):
        animator, _ = animator_for()
        knees = [
            animator.pose_offsets(animator.state_at(t * 0.05))
            for t in range(400)
        ]
        gaps = [abs(o.get("calf_l", (0, 0, 0))[1] - o.get("calf_r", (0, 0, 0))[1])
                for o in knees]
        self.assertGreater(max(gaps), 0.3, "both knees behaved identically")


class TestJointSafety(unittest.TestCase):
    """Whatever the music does, the skeleton stays inside its limits."""

    def _sweep(self, animator):
        for step in range(600):
            yield animator.pose_offsets(animator.state_at(step * 0.05))

    def test_offsets_are_finite_and_within_limits(self):
        for energy in (lambda t: 0.05, lambda t: t / DURATION, lambda t: 1.0):
            animator, _ = animator_for(energy=energy)
            for offsets in self._sweep(animator):
                for joint, values in offsets.items():
                    limits = AvatarRig.LIMITS.get(joint, AvatarRig.DEFAULT_LIMIT)
                    for value, limit, axis in zip(values, limits, "HPR"):
                        self.assertTrue(math.isfinite(value), f"{joint}.{axis}")
                        self.assertLessEqual(
                            abs(value), limit + 1e-9,
                            f"{joint}.{axis} = {value:.2f} exceeds limit {limit}",
                        )

    def test_every_driven_joint_is_one_the_rig_controls(self):
        animator, _ = animator_for()
        for offsets in self._sweep(animator):
            for joint in offsets:
                self.assertIn(joint, AvatarRig.CONTROLLED)

    def test_pose_changes_smoothly_between_close_samples(self):
        """No sudden jumps that would read as a glitch."""
        animator, _ = animator_for(energy=lambda t: t / DURATION)
        previous = None
        worst = 0.0
        for step in range(2000):
            offsets = animator.pose_offsets(animator.state_at(step * 0.01))
            if previous is not None:
                for joint, values in offsets.items():
                    for a, b in zip(values, previous.get(joint, (0, 0, 0))):
                        worst = max(worst, abs(a - b))
            previous = offsets
        self.assertLess(worst, 1.5, f"pose jumped {worst:.2f} deg in 10 ms")


class TestGrooveIsScheduleIndependent(unittest.TestCase):
    """Phase 0's guarantee must survive the new behaviour layer."""

    SCHEDULES = {
        "60 Hz": [i / 60.0 for i in range(1, 900)],
        "30 Hz": [i / 30.0 for i in range(1, 450)],
        "15 Hz": [i / 15.0 for i in range(1, 225)],
        "5 Hz": [i / 5.0 for i in range(1, 75)],
        "sparse": [0.833 * i for i in range(1, 18)],
        "stalled": [t for t in
                    [0.1, 0.2, 0.3, 1.4, 1.5, 1.6, 3.7, 3.8, 5.9, 6.0, 9.2, 11.6]],
    }
    PROBES = [1.03, 2.5, 3.017, 4.44, 7.25, 11.5, 13.9]

    def test_pose_at_a_time_ignores_the_frames_before_it(self):
        for name, schedule in self.SCHEDULES.items():
            for probe in self.PROBES:
                reference, ref_rig = animator_for()
                reference.apply_at(probe)

                played, rig = animator_for()
                for frame in schedule:
                    if frame >= probe:
                        break
                    played.apply_at(frame)
                played.apply_at(probe)

                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(rig.pose(), ref_rig.pose())

    def test_groove_state_is_identical_across_schedules(self):
        animator, _ = animator_for()
        for probe in self.PROBES:
            direct = animator.state_at(probe)
            for name, schedule in self.SCHEDULES.items():
                for frame in schedule:
                    if frame >= probe:
                        break
                    animator.apply_at(frame)
                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(animator.state_at(probe), direct)


if __name__ == "__main__":
    unittest.main()
