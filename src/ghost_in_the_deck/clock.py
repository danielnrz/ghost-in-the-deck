"""The authoritative clock for musical time.

Panda3D's OpenAL sound reports its own play position, which is the honest
reference for synchronisation: it follows the audio the listener is actually
hearing rather than how fast frames happen to be drawn.

It cannot be read naively, though. ``getTime()`` is *sampled*, not free-running:
Panda3D refreshes it from a task, so while the main loop is blocked the reading
does not move at all, even though the sound card keeps playing. Reading it
directly would freeze musical time for exactly as long as the renderer stalls -
the opposite of what an authoritative clock is for.

So the reading is used as an anchor and the wall clock carries time forward
between refreshes. Every time the sound reports a new position the anchor is
reset to it, which keeps the estimate locked to real playback and bounds any
error to a single refresh interval.

Extrapolating forward is only correct while the sound is actually playing. If
playback stops the clock would otherwise run on for the length of the track with
nothing audible, so once the sound has reported a non-playing status for longer
than ``stopped_grace`` the clock holds its position instead.

**Known limitation.** Panda3D's ``AudioSound.status()`` has exactly three values:
BAD, READY and PLAYING. It cannot distinguish a sound that finished, one that was
stopped early, one that is paused, one whose device underran, and one that
failed. The grace-then-hold rule therefore treats every non-playing state the
same way. The grace period exists so that a momentary status blip during normal
playback does not stall musical time; it is not pause support, and this project
does not implement pause or resume.
"""

from __future__ import annotations

import time
from typing import Callable

# How long the sound may report a non-playing status before the clock stops
# extrapolating. Long enough to ride out a transient blip, short enough that a
# genuinely dead sound does not leave the avatar dancing to nothing.
STOPPED_GRACE = 0.5


class PlaybackClock:
    """Current position in the track.

    Without a sound the wall clock stands in, so the prototype can still be
    exercised silently and headlessly.
    """

    def __init__(
        self,
        sound=None,
        wall: Callable[[], float] | None = None,
        stopped_grace: float = STOPPED_GRACE,
    ):
        self.sound = sound
        self.stopped_grace = stopped_grace
        self._wall = wall or time.perf_counter
        self._anchor_audio = 0.0
        self._anchor_wall = self._wall()
        # Seeded rather than left unset: an unchanged reading must not re-anchor,
        # otherwise the first read after a block would throw away the time that
        # passed during it.
        self._last_reading = 0.0
        self._last_playing_wall = self._anchor_wall
        self._latest = 0.0

    def start(self) -> None:
        self._anchor_audio = 0.0
        self._anchor_wall = self._wall()
        self._last_reading = 0.0
        self._last_playing_wall = self._anchor_wall
        self._latest = 0.0
        if self.sound is not None:
            self.sound.play()

    def _playing(self) -> bool:
        return self.sound is not None and self.sound.status() == self.sound.PLAYING

    @property
    def holding(self) -> bool:
        """True once playback has stopped for longer than the grace period.

        The clock is no longer advancing, so the caller should wind the run up
        rather than keep animating against a frozen time.
        """
        if self.sound is None or self._playing():
            return False
        return (self._wall() - self._last_playing_wall) > self.stopped_grace

    def time(self) -> float:
        """Playback position in seconds. Never goes backwards.

        A sound that stops or is re-buffered can briefly report an earlier
        position, and letting that through would send the avatar back to an
        earlier part of the music.
        """
        wall = self._wall()

        if self._playing():
            reading = self.sound.getTime()
            if reading != self._last_reading:
                self._last_reading = reading
                self._anchor_audio = reading
                self._anchor_wall = wall
            self._last_playing_wall = wall
        elif self.sound is not None and (wall - self._last_playing_wall) > self.stopped_grace:
            # Playback has definitively stopped. Extrapolating further would
            # invent musical time that nobody can hear.
            return self._latest

        estimate = self._anchor_audio + (wall - self._anchor_wall)
        self._latest = max(self._latest, estimate)
        return self._latest
