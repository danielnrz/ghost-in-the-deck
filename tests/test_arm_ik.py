"""Two-bone arm IK geometry (animation/arm_ik.py), tested against independent
geometric facts rather than by re-deriving the same formula.

Pure Python math, no Panda3D, no display needed: every check here can run in
any environment, including a clean clone with no music and no GPU.
"""

from __future__ import annotations

import math
import unittest

from ghost_in_the_deck.animation.arm_ik import distance, solve_elbow

SHOULDER = (0.16, -0.01, 1.34)
UPPER_LEN = 0.24
FORE_LEN = 0.24
POLE = (0.3, -0.4, -0.8)


def within_reach(d: float) -> tuple[float, float, float]:
    """A point at distance `d` from SHOULDER, roughly toward the workstation."""
    direction = (0.15, -0.9, -0.5)
    length = math.sqrt(sum(c * c for c in direction))
    unit = tuple(c / length for c in direction)
    return tuple(SHOULDER[i] + unit[i] * d for i in range(3))


class TestReachableTarget(unittest.TestCase):
    """A target strictly inside [|L1-L2|, L1+L2] must be hit exactly."""

    def test_elbow_sits_exactly_upper_length_from_shoulder(self):
        for d in (0.05, 0.15, 0.30, 0.40, 0.47):
            target = within_reach(d)
            with self.subTest(d=d):
                solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
                self.assertAlmostEqual(distance(SHOULDER, solution.elbow), UPPER_LEN, places=9)

    def test_target_sits_exactly_fore_length_from_elbow(self):
        for d in (0.05, 0.15, 0.30, 0.40, 0.47):
            target = within_reach(d)
            with self.subTest(d=d):
                solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
                self.assertAlmostEqual(distance(solution.elbow, solution.aim_point), FORE_LEN, places=9)

    def test_reachable_target_is_not_clamped(self):
        target = within_reach(0.3)
        solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
        self.assertFalse(solution.was_clamped)
        for a, b in zip(solution.aim_point, target):
            self.assertAlmostEqual(a, b, places=9)

    def test_reachable_target_is_reproduced_exactly(self):
        target = within_reach(0.35)
        solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
        self.assertAlmostEqual(distance(solution.aim_point, target), 0.0, places=9)


class TestUnreachableTargetClamping(unittest.TestCase):
    """Handled gracefully - clamped to the reachable sphere, no invalid state."""

    def test_far_target_is_clamped_to_max_reach(self):
        far = within_reach(50.0)   # absurdly far, along the same direction
        solution = solve_elbow(SHOULDER, far, UPPER_LEN, FORE_LEN, POLE)
        self.assertTrue(solution.was_clamped)
        self.assertLess(solution.reach_distance, UPPER_LEN + FORE_LEN)
        self.assertGreater(solution.reach_distance, UPPER_LEN + FORE_LEN - 0.01)

    def test_far_target_still_satisfies_both_bone_lengths(self):
        far = within_reach(10.0)
        solution = solve_elbow(SHOULDER, far, UPPER_LEN, FORE_LEN, POLE)
        self.assertAlmostEqual(distance(SHOULDER, solution.elbow), UPPER_LEN, places=6)
        self.assertAlmostEqual(distance(solution.elbow, solution.aim_point), FORE_LEN, places=6)

    def test_too_close_target_is_clamped_to_min_reach(self):
        # Equal bone lengths give a min_reach of exactly zero, so nothing this
        # close to the shoulder can be "too close" to test the clamp with -
        # unequal lengths are what make min_reach nonzero.
        upper, fore = 0.30, 0.15
        very_close = (SHOULDER[0] + 1e-4, SHOULDER[1], SHOULDER[2])
        solution = solve_elbow(SHOULDER, very_close, upper, fore, POLE)
        self.assertTrue(solution.was_clamped)
        self.assertAlmostEqual(solution.reach_distance, abs(upper - fore), places=3)

    def test_target_exactly_on_the_shoulder_does_not_crash(self):
        solution = solve_elbow(SHOULDER, SHOULDER, UPPER_LEN, FORE_LEN, POLE)
        self.assertTrue(math.isfinite(solution.elbow[0]))
        self.assertAlmostEqual(distance(SHOULDER, solution.elbow), UPPER_LEN, places=6)

    def test_asymmetric_bone_lengths_clamp_correctly(self):
        """min_reach = |L1-L2| is nonzero when the bones differ noticeably."""
        upper, fore = 0.30, 0.10
        very_close = (SHOULDER[0] + 1e-4, SHOULDER[1], SHOULDER[2])
        solution = solve_elbow(SHOULDER, very_close, upper, fore, POLE)
        self.assertAlmostEqual(solution.reach_distance, upper - fore, places=3)
        self.assertAlmostEqual(distance(SHOULDER, solution.elbow), upper, places=6)
        self.assertAlmostEqual(distance(solution.elbow, solution.aim_point), fore, places=6)


class TestFiniteAndWellFormed(unittest.TestCase):
    def test_every_component_is_finite_across_many_targets(self):
        import random

        rng = random.Random(1234)
        for _ in range(200):
            target = tuple(rng.uniform(-2.0, 2.0) for _ in range(3))
            solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
            for value in (*solution.elbow, *solution.aim_point, solution.reach_distance):
                self.assertTrue(math.isfinite(value))

    def test_rejects_non_positive_bone_lengths(self):
        with self.assertRaises(ValueError):
            solve_elbow(SHOULDER, within_reach(0.2), 0.0, FORE_LEN, POLE)
        with self.assertRaises(ValueError):
            solve_elbow(SHOULDER, within_reach(0.2), UPPER_LEN, -0.1, POLE)


class TestBendPlanePreference(unittest.TestCase):
    """The pole vector should influence which side the elbow bends toward."""

    def test_different_poles_give_different_elbows_for_the_same_target(self):
        target = within_reach(0.30)
        pole_a = (1.0, 0.0, 0.0)
        pole_b = (-1.0, 0.0, 0.0)
        elbow_a = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, pole_a).elbow
        elbow_b = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, pole_b).elbow
        self.assertGreater(distance(elbow_a, elbow_b), 0.05)

    def test_a_pole_parallel_to_the_reach_direction_does_not_crash(self):
        target = within_reach(0.30)
        direction = tuple(target[i] - SHOULDER[i] for i in range(3))
        solution = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, direction)
        self.assertAlmostEqual(distance(SHOULDER, solution.elbow), UPPER_LEN, places=6)


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_give_the_same_solution(self):
        target = within_reach(0.28)
        first = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
        second = solve_elbow(SHOULDER, target, UPPER_LEN, FORE_LEN, POLE)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
