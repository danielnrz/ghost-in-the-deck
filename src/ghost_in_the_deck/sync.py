"""Measuring how closely the avatar's movement follows the music.

Two different things are measured here and they must not be confused.

**State currency** - when the update ran, was the pose the one the music calls
for at that moment? This is what the timing architecture guarantees. It is
reported as *state lag*: the playback time that had elapsed by the end of the
update minus the playback time the pose was evaluated for. It stays near zero
however badly the renderer behaves, because the pose is a pure function of
playback time rather than something integrated frame by frame.

**Sample coverage** - did the update run at all while a cue's response was
current? An update loop that freezes for a second produces no samples during
that second, and no architecture can change that. Beats that pass with no sample
are counted as *missed*. That says the loop did not get to them; it does not say
the music drifted.

What "sampled" does and does not claim
--------------------------------------
A sample is recorded in the update task, straight after the pose is written. It
proves that the application evaluated and wrote the pose for that playback time.
It does **not** prove that the GPU and compositor put that frame in front of the
viewer, and it does not prove when. Measuring real presentation would need
GPU timer queries or compositor presentation feedback, neither of which this
project does, so the metrics are named for what they actually observe: update
samples, not presented frames.

Neither figure includes sound card output latency, which would need an external
recording to quantify.
"""

from __future__ import annotations

import json
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeatResponse:
    """A beat, and the first update sample that carried its response."""

    index: int
    beat_time: float
    sampled_at: float

    @property
    def latency_ms(self) -> float:
        return (self.sampled_at - self.beat_time) * 1000.0

    def format(self) -> str:
        return (
            f"Beat {self.index}\n"
            f"  expected   : {self.beat_time:.3f} s\n"
            f"  sampled at : {self.sampled_at:.3f} s\n"
            f"  latency    : {self.latency_ms:+.1f} ms"
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
    """Collects per-sample and per-beat timing during a run.

    The beat times are supplied up front rather than inferred from whatever
    happened to be observed. Coverage is then a comparison between the beats the
    run passed through and the beats a sample actually caught, which is the only
    way a beat missed before the first successful sample can be counted.
    """

    def __init__(self, beat_times=None, response_window: float = 0.48):
        self.response_window = response_window
        self._beat_times: list[float] = sorted(beat_times or [])
        self.responses: list[BeatResponse] = []

        self._state_lags: list[float] = []
        self._sample_intervals: list[float] = []
        self._update_costs: list[float] = []
        self._clock_skews: list[float] = []

        self._samples = 0
        self._first_time: float | None = None
        self._last_time = 0.0
        self._last_beat_index = -1
        self._beats_seen: set[int] = set()

        self._previous_audio: float | None = None
        self._previous_wall: float | None = None

    # ---------------------------------------------------------------- capture
    def record_sample(
        self,
        state,
        observed_at: float,
        wall_time: float | None = None,
        update_seconds: float = 0.0,
    ) -> BeatResponse | None:
        """Log one update sample.

        ``state`` is the pose state just written, ``observed_at`` is the
        playback time read once the update had finished, and ``wall_time`` is a
        wall-clock reading taken at the same moment as the playback time the
        pose was evaluated for.

        Both intervals are derived here from the readings this method is given,
        so the playback and wall endpoints are matched by construction. Deriving
        them separately at the call site is what previously let an injected
        stall land in one interval but not the other.

        Returns the beat response when this sample is the first to carry a beat.
        """
        self._samples += 1
        if self._first_time is None:
            self._first_time = state.time
        self._last_time = max(self._last_time, observed_at)

        self._state_lags.append((observed_at - state.time) * 1000.0)
        self._update_costs.append(update_seconds * 1000.0)

        if self._previous_audio is not None:
            audio_interval = state.time - self._previous_audio
            self._sample_intervals.append(audio_interval * 1000.0)
            if wall_time is not None and self._previous_wall is not None:
                wall_interval = wall_time - self._previous_wall
                self._clock_skews.append((audio_interval - wall_interval) * 1000.0)
        self._previous_audio = state.time
        self._previous_wall = wall_time

        # A beat is covered by the first sample that carries it while its
        # response is still within the reporting window.
        response = None
        if (
            state.has_beat
            and state.beat_index != self._last_beat_index
            and state.beat_index not in self._beats_seen
            and state.beat_age <= self.response_window
        ):
            response = BeatResponse(
                index=state.beat_index,
                beat_time=state.time - state.beat_age,
                sampled_at=state.time,
            )
            self.responses.append(response)
            self._beats_seen.add(state.beat_index)
        if state.has_beat:
            self._last_beat_index = state.beat_index
        return response

    # ---------------------------------------------------------------- reports
    def beats_in_scope(self) -> list[int]:
        """Indices of beats this run can be held responsible for.

        A beat counts when the run was already sampling before it happened and
        was still sampling once its whole response window had passed. Anything
        outside that cannot be adjudicated: the run either had not started or
        had already stopped, and blaming it for those would be wrong in both
        directions.

        A beat that was actually sampled is always in scope. Its window may have
        run past the end of the run, but there is nothing to decide - it was
        caught. Leaving it out is what made a full run report "65 of 64".
        """
        if self._first_time is None or not self._beat_times:
            return sorted(self._beats_seen)
        first = bisect_left(self._beat_times, self._first_time)
        last = bisect_right(self._beat_times, self._last_time - self.response_window)
        return sorted(set(range(first, max(first, last))) | self._beats_seen)

    @property
    def beats_missed(self) -> int:
        """In-scope beats that no update sample carried."""
        return sum(1 for index in self.beats_in_scope() if index not in self._beats_seen)

    def missed_indices(self) -> list[int]:
        return [i for i in self.beats_in_scope() if i not in self._beats_seen]

    def summary(self) -> dict:
        elapsed = 0.0
        if self._first_time is not None:
            elapsed = max(self._last_time - self._first_time, 0.0)
        rate = round(self._samples / elapsed, 1) if elapsed > 0 else 0.0

        latencies = [r.latency_ms for r in self.responses]
        return {
            "update_samples": self._samples,
            "elapsed_seconds": round(elapsed, 3),
            "update_rate_hz": rate,
            "sample_interval": _stats(self._sample_intervals),
            "update_cost": _stats(self._update_costs),
            "state_lag": _stats(self._state_lags),
            "playback_vs_wall_skew": _stats(self._clock_skews),
            "beats_in_scope": len(self.beats_in_scope()),
            "beats_sampled": len(self.responses),
            "beats_missed": self.beats_missed,
            "beat_response_latency": _stats(latencies),
        }

    def format_summary(self) -> str:
        data = self.summary()
        interval = data["sample_interval"]
        lag = data["state_lag"]
        cost = data["update_cost"]
        latency = data["beat_response_latency"]

        lines = [
            f"Update samples        : {data['update_samples']}  "
            f"({data['update_rate_hz']} Hz over {data['elapsed_seconds']:.1f} s)",
        ]
        if interval.get("count"):
            lines.append(
                f"Sample interval       : mean {interval['mean_ms']:.2f} ms, "
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
            f"Beats sampled         : {data['beats_sampled']} of "
            f"{data['beats_in_scope']} in scope"
        )
        lines.append(
            f"Beats missed          : {data['beats_missed']}"
            "   (no update sample inside the response window)"
        )
        if latency.get("count"):
            lines.append(
                f"Beat response latency : mean {latency['mean_ms']:.2f} ms, "
                f"median {latency['median_ms']:.2f} ms, max {latency['max_ms']:.1f} ms"
            )
        lines.append(
            "Samples are update-task observations, not verified monitor presentation."
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
                        "what_a_sample_is": (
                            "One update-task run, recorded straight after the pose "
                            "was written. It proves the application evaluated and "
                            "wrote that pose for that playback time. It does not "
                            "prove the GPU or compositor presented the frame, which "
                            "this project does not measure."
                        ),
                        "state_lag": (
                            "Playback time elapsed during the update minus the time "
                            "the pose was evaluated for. Near zero by construction; "
                            "a large value would mean the musical state fell behind."
                        ),
                        "beats_in_scope": (
                            "Beats the run was sampling before they happened and "
                            "still sampling once their response window had passed, "
                            "so coverage can be judged for them."
                        ),
                        "beats_missed": (
                            "In-scope beats no update sample carried, because the "
                            "loop did not run during their response window. A loop "
                            "limitation, not a timeline error."
                        ),
                        "beat_response_latency": (
                            "For beats that were sampled, the delay before the "
                            "first sample carrying them."
                        ),
                        "response_window": (
                            "Reporting threshold in seconds, shared with the "
                            "animator: how soon after a beat a sample must occur "
                            "for that beat to count as covered. A deliberate "
                            "diagnostic choice, not a physical visibility boundary."
                        ),
                        "playback_vs_wall_skew": (
                            "Playback interval minus wall interval between "
                            "consecutive samples, both endpoints taken together. "
                            "Near zero unless the two clocks genuinely diverge."
                        ),
                    },
                    "response_window_seconds": round(self.response_window, 4),
                    "summary": self.summary(),
                    "missed_beat_indices": self.missed_indices(),
                    "beats": [
                        {
                            "index": r.index,
                            "expected": round(r.beat_time, 4),
                            "sampled_at": round(r.sampled_at, 4),
                            "latency_ms": round(r.latency_ms, 2),
                        }
                        for r in self.responses
                    ],
                },
                indent=1,
            )
        )
        return path
