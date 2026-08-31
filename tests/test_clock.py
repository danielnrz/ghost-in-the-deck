"""The playback clock must keep musical time even while the loop is blocked.

Panda3D refreshes a sound's reported position from a task, so a blocked main
loop leaves that reading frozen while the sound card plays on. These tests use a
fake sound and a fake wall clock to pin down the behaviour exactly.
"""

from __future__ import annotations

import unittest

from ghost_in_the_deck.clock import PlaybackClock


class FakeWall:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSound:
    """Reports a position that only moves when refresh() is called."""

    PLAYING = "playing"
    STOPPED = "stopped"

    def __init__(self) -> None:
        self.position = 0.0
        self._status = self.PLAYING
        self.played = False

    def play(self) -> None:
        self.played = True

    def status(self) -> str:
        return self._status

    def getTime(self) -> float:
        return self.position

    def refresh(self, to: float) -> None:
        self.position = to

    def stop(self) -> None:
        self._status = self.STOPPED

    def resume(self) -> None:
        self._status = self.PLAYING


class TestPlaybackClock(unittest.TestCase):
    def setUp(self):
        self.wall = FakeWall()
        self.sound = FakeSound()
        self.clock = PlaybackClock(self.sound, wall=self.wall)
        self.clock.start()

    def test_start_plays_the_sound_from_zero(self):
        self.assertTrue(self.sound.played)
        self.assertAlmostEqual(self.clock.time(), 0.0, places=9)

    def test_time_advances_while_the_reading_is_frozen(self):
        """The whole point: a blocked loop must not freeze musical time."""
        self.wall.advance(1.0)
        self.assertAlmostEqual(self.clock.time(), 1.0, places=9)
        self.assertEqual(self.sound.position, 0.0, "fixture should stay frozen")

    def test_a_new_reading_re_anchors_the_estimate(self):
        self.wall.advance(0.5)
        self.assertAlmostEqual(self.clock.time(), 0.5, places=9)

        # The audio task catches up and reports a slightly different position.
        self.sound.refresh(0.62)
        self.assertAlmostEqual(self.clock.time(), 0.62, places=9)

        self.wall.advance(0.1)
        self.assertAlmostEqual(self.clock.time(), 0.72, places=9)

    def test_estimate_tracks_the_sound_rather_than_drifting(self):
        for step in range(1, 21):
            self.wall.advance(0.05)
            self.sound.refresh(step * 0.05)
            self.assertAlmostEqual(self.clock.time(), step * 0.05, places=9)

    def test_time_never_goes_backwards(self):
        self.wall.advance(2.0)
        ahead = self.clock.time()
        self.sound.refresh(0.25)          # a re-buffer reports an earlier spot
        self.assertGreaterEqual(self.clock.time(), ahead)

    def test_falls_back_to_the_wall_clock_without_a_sound(self):
        wall = FakeWall()
        clock = PlaybackClock(None, wall=wall)
        clock.start()
        wall.advance(3.25)
        self.assertAlmostEqual(clock.time(), 3.25, places=9)

    def test_a_brief_status_blip_does_not_stall_musical_time(self):
        """Panda3D cannot distinguish an underrun from a stop, so ride it out."""
        self.wall.advance(1.0)
        before = self.clock.time()

        self.sound.stop()
        self.wall.advance(0.2)                 # inside the grace period
        self.assertGreater(self.clock.time(), before)
        self.assertFalse(self.clock.holding)

        self.sound.resume()
        self.wall.advance(0.2)
        self.assertGreater(self.clock.time(), before + 0.3)

    def test_time_holds_once_playback_has_definitively_stopped(self):
        """The avatar must not dance on through a track nobody is hearing."""
        self.wall.advance(1.0)
        self.clock.time()

        self.sound.stop()
        self.wall.advance(self.clock.stopped_grace + 0.01)
        held = self.clock.time()
        self.assertTrue(self.clock.holding)

        self.wall.advance(30.0)
        self.assertEqual(self.clock.time(), held, "clock ran on past a stopped sound")

    def test_holding_is_false_during_normal_playback(self):
        self.wall.advance(5.0)
        self.clock.time()
        self.assertFalse(self.clock.holding)

    def test_a_silent_run_is_never_treated_as_stopped(self):
        """--no-audio has no sound to report a status; the wall clock rules."""
        wall = FakeWall()
        clock = PlaybackClock(None, wall=wall)
        clock.start()
        wall.advance(10.0)
        self.assertFalse(clock.holding)
        self.assertAlmostEqual(clock.time(), 10.0, places=9)

    def test_a_long_block_while_playing_still_advances(self):
        """The Phase 0.1 fix must survive the stopped-audio rule.

        The sound stays PLAYING through a blocked loop, so the grace period is
        never reached and extrapolation carries musical time across the freeze.
        """
        self.wall.advance(1.2)                 # loop blocked, status still PLAYING
        self.assertAlmostEqual(self.clock.time(), 1.2, places=9)
        self.assertFalse(self.clock.holding)

        self.sound.refresh(1.204)              # audio task finally catches up
        self.assertAlmostEqual(self.clock.time(), 1.204, places=9)

    def test_recovery_after_holding_resumes_from_the_held_position(self):
        self.wall.advance(1.0)
        self.clock.time()
        self.sound.stop()
        self.wall.advance(self.clock.stopped_grace + 5.0)
        held = self.clock.time()

        self.sound.resume()
        self.sound.refresh(1.05)
        self.assertGreaterEqual(self.clock.time(), held)

    def test_start_resets_the_position(self):
        self.wall.advance(5.0)
        self.assertGreater(self.clock.time(), 4.0)
        self.clock.start()
        self.assertAlmostEqual(self.clock.time(), 0.0, places=9)

    def test_a_long_block_yields_the_full_elapsed_time(self):
        """A one second freeze must show up as one second of music."""
        self.wall.advance(1.0)
        blocked = self.clock.time()
        self.assertAlmostEqual(blocked, 1.0, places=9)

        # When the audio task finally runs it confirms roughly that position.
        self.sound.refresh(1.004)
        self.assertAlmostEqual(self.clock.time(), 1.004, places=9)


if __name__ == "__main__":
    unittest.main()
