"""Measuring how closely the avatar's movement follows the music.

Two different things are measured here and they must not be confused.

**State currency** - when a frame is drawn, was the pose the one the music calls
for at that moment? This is what the timing architecture guarantees. It is
reported as *state lag*: the playback time that had elapsed by the end of the
update minus the playback time the pose was evaluated for. It stays near zero
however badly the renderer behaves, because the pose is a pure function of
playback time rather than something integrated frame by frame.

**Visual coverage** - was a frame drawn at all while a cue's response was on
screen? A renderer that freezes for a second cannot show anything during that
second, and no architecture can change that. Beats that pass with no frame are
counted as *never rendered*. That number is honest: it says the display missed
them, not that the music drifted.

Neither figure includes sound card output latency, which would need an external
recording to quantify.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeatResponse:
    """A beat, and the first frame that actually displayed its response."""

    index: int
    beat_time: float
    shown_at: float

    @property
    def latency_ms(self) -> float:
        return (self.shown_at - self.beat_time) * 1000.0

    def format(self) -> str:
        return (
            f"Beat {self.index}\n"
            f"  expected : {self.beat_time:.3f} s\n"
            f"  shown at : {self.shown_at:.3f} s\n"
            f"  latency  : {self.latency_ms:+.1f} ms"
        )


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    absolute = [abs(v) for v in values]
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(absolute), 3),
        "median_ms": round(statistics.median(absolute), 3),
        "max_ms": round(max(absolute), 3),
    }


class TimingRecorder:
    """Collects per-frame and per-beat timing during a run."""

    def __init__(self, visible_for: float = 0.48):
        self.visible_for = visible_for
        self.responses: list[BeatResponse] = []

        self._state_lags: list[float] = []
        self._frame_intervals: list[float] = []
        self._update_costs: list[float] = []
        self._clock_skews: list[float] = []

        self._frames = 0
        self._first_time: float | None = None
        self._last_time = 0.0
        self._last_beat_index = -1
        self._beats_seen: set[int] = set()
        self._first_beat_index: int | None = None

    # ---------------------------------------------------------------- capture
    def record_frame(
        self,
        state,
        observed_at: float,
        frame_interval: float | None = None,
        update_seconds: float = 0.0,
        wall_interval: float | None = None,
    ) -> BeatResponse | None:
        """Log one rendered frame.

        ``state`` is the MotionState the frame drew, ``observed_at`` is the
        playback time read once the update had finished. Returns the beat
        response if this frame was the first to show a new beat.
        """
        self._frames += 1
        if self._first_time is None:
            self._first_time = state.time
        self._last_time = max(self._last_time, observed_at)

        self._state_lags.append((observed_at - state.time) * 1000.0)
        self._update_costs.append(update_seconds * 1000.0)
        if frame_interval is not None:
            self._frame_intervals.append(frame_interval * 1000.0)
            if wall_interval is not None:
                self._clock_skews.append((frame_interval - wall_interval) * 1000.0)

        # A beat counts as displayed the first time a frame draws it while its
        # response is still visible.
        response = None
        if (
            state.has_beat
            and state.beat_index != self._last_beat_index
            and state.beat_index not in self._beats_seen
            and state.beat_age <= self.visible_for
        ):
            response = BeatResponse(
                index=state.beat_index,
                beat_time=state.time - state.beat_age,
                shown_at=state.time,
            )
            self.responses.append(response)
            self._beats_seen.add(state.beat_index)
            if self._first_beat_index is None:
                self._first_beat_index = state.beat_index
        if state.has_beat:
            self._last_beat_index = state.beat_index
        return response

    # ---------------------------------------------------------------- reports
    @property
    def beats_never_rendered(self) -> int:
        """Beats inside the observed span that no frame ever displayed."""
        if self._first_beat_index is None:
            return 0
        span = range(self._first_beat_index, self._last_beat_index + 1)
        return sum(1 for index in span if index not in self._beats_seen)

    def summary(self) -> dict:
        elapsed = 0.0
        if self._first_time is not None:
            elapsed = max(self._last_time - self._first_time, 0.0)
        fps = round(self._frames / elapsed, 1) if elapsed > 0 else 0.0

        latencies = [r.latency_ms for r in self.responses]
        return {
            "frames": self._frames,
            "elapsed_seconds": round(elapsed, 3),
            "average_fps": fps,
            "frame_interval": _stats(self._frame_intervals),
            "update_cost": _stats(self._update_costs),
            "state_lag": _stats(self._state_lags),
            "audio_vs_wall_skew": _stats(self._clock_skews),
            "beats_displayed": len(self.responses),
            "beats_never_rendered": self.beats_never_rendered,
            "beat_response_latency": _stats(latencies),
        }

    def format_summary(self) -> str:
        data = self.summary()
        interval = data["frame_interval"]
        lag = data["state_lag"]
        cost = data["update_cost"]
        latency = data["beat_response_latency"]

        lines = [
            f"Frames rendered       : {data['frames']}  ({data['average_fps']} fps"
            f" over {data['elapsed_seconds']:.1f} s)",
        ]
        if interval.get("count"):
            lines.append(
                f"Frame interval        : mean {interval['mean_ms']:.2f} ms, "
                f"median {interval['median_ms']:.2f} ms, worst {interval['max_ms']:.1f} ms"
            )
        if cost.get("count"):
            lines.append(
                f"Update cost (our code): mean {cost['mean_ms']:.3f} ms, "
                f"worst {cost['max_ms']:.2f} ms"
            )
        if lag.get("count"):
            lines.append(
                f"State lag             : mean {lag['mean_ms']:.3f} ms, "
                f"worst {lag['max_ms']:.2f} ms   (pose age when written)"
            )
        lines.append(
            f"Beats displayed       : {data['beats_displayed']}"
        )
        lines.append(
            f"Beats never rendered  : {data['beats_never_rendered']}"
            "   (no frame while the response was on screen)"
        )
        if latency.get("count"):
            lines.append(
                f"Beat response latency : mean {latency['mean_ms']:.2f} ms, "
                f"median {latency['median_ms']:.2f} ms, max {latency['max_ms']:.1f} ms"
            )
        return "\n".join(lines)

    def save(self, path: Path | str, track: str = "") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "track": track,
                    "metric_semantics": {
                        "state_lag": (
                            "Playback time elapsed during the update minus the time "
                            "the pose was evaluated for. Near zero by construction; "
                            "a large value would mean the musical state fell behind."
                        ),
                        "beats_never_rendered": (
                            "Beats no frame displayed because rendering stalled. A "
                            "display limitation, not a timeline error."
                        ),
                        "beat_response_latency": (
                            "For beats that were displayed, the delay before the "
                            "first frame showing them."
                        ),
                    },
                    "summary": self.summary(),
                    "beats": [
                        {
                            "index": r.index,
                            "expected": round(r.beat_time, 4),
                            "shown_at": round(r.shown_at, 4),
                            "latency_ms": round(r.latency_ms, 2),
                        }
                        for r in self.responses
                    ],
                },
                indent=1,
            )
        )
        return path
