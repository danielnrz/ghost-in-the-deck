"""Measuring how closely the avatar's movement follows the music.

Every triggered beat is compared against the timestamp the analysis predicted,
using the playback clock as the reference. This measures scheduling accuracy
inside the application; it does not include the latency of the sound card
itself, which would need an external recording to quantify.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeatTiming:
    index: int
    expected: float
    triggered: float

    @property
    def error_ms(self) -> float:
        return (self.triggered - self.expected) * 1000.0

    def format(self) -> str:
        return (
            f"Beat {self.index}\n"
            f"  expected : {self.expected:.3f} s\n"
            f"  triggered: {self.triggered:.3f} s\n"
            f"  error    : {self.error_ms:+.1f} ms"
        )


class SyncRecorder:
    def __init__(self) -> None:
        self.timings: list[BeatTiming] = []

    def record(self, index: int, expected: float, triggered: float) -> BeatTiming:
        timing = BeatTiming(index=index, expected=expected, triggered=triggered)
        self.timings.append(timing)
        return timing

    def summary(self) -> dict:
        if not self.timings:
            return {"beats": 0}
        errors = [t.error_ms for t in self.timings]
        absolute = [abs(e) for e in errors]
        return {
            "beats": len(errors),
            "mean_abs_error_ms": round(statistics.mean(absolute), 2),
            "median_abs_error_ms": round(statistics.median(absolute), 2),
            "max_abs_error_ms": round(max(absolute), 2),
            "mean_signed_error_ms": round(statistics.mean(errors), 2),
            "stdev_error_ms": round(statistics.pstdev(errors), 2),
        }

    def format_summary(self) -> str:
        data = self.summary()
        if not data.get("beats"):
            return "No beats were triggered."
        return (
            f"Beats triggered      : {data['beats']}\n"
            f"Mean absolute error  : {data['mean_abs_error_ms']:.2f} ms\n"
            f"Median absolute error: {data['median_abs_error_ms']:.2f} ms\n"
            f"Maximum error        : {data['max_abs_error_ms']:.2f} ms\n"
            f"Mean signed error    : {data['mean_signed_error_ms']:+.2f} ms\n"
            f"Std deviation        : {data['stdev_error_ms']:.2f} ms"
        )

    def save(self, path: Path | str, track: str = "") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "track": track,
                    "summary": self.summary(),
                    "beats": [
                        {
                            "index": t.index,
                            "expected": round(t.expected, 4),
                            "triggered": round(t.triggered, 4),
                            "error_ms": round(t.error_ms, 2),
                        }
                        for t in self.timings
                    ],
                },
                indent=1,
            )
        )
        return path
