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
"""

from __future__ import annotations

import time
from typing import Callable


class PlaybackClock:
    """Current position in the track.

    Without a sound the wall clock stands in, so the prototype can still be
    exercised silently and headlessly.
    """

    def __init__(self, sound=None, wall: Callable[[], float] | None = None):
        self.sound = sound
        self._wall = wall or time.perf_counter
        self._anchor_audio = 0.0
        self._anchor_wall = self._wall()
        # Seeded rather than left unset: an unchanged reading must not re-anchor,
        # otherwise the first read after a block would throw away the time that
        # passed during it.
        self._last_reading = 0.0
        self._latest = 0.0

    def start(self) -> None:
        self._anchor_audio = 0.0
        self._anchor_wall = self._wall()
        self._last_reading = 0.0
        self._latest = 0.0
        if self.sound is not None:
            self.sound.play()

    def _playing(self) -> bool:
        return self.sound is not None and self.sound.status() == self.sound.PLAYING

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

        estimate = self._anchor_audio + (wall - self._anchor_wall)
        self._latest = max(self._latest, estimate)
        return self._latest
