"""Buffered OpenAL set playback. Audio preparation never touches the scene."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import queue
import threading
import time

import numpy as np

from .set_engine import DeckSpan, SAMPLE_RATE, SetEngine, frame


@dataclass(frozen=True)
class AudioChunk:
    span: DeckSpan
    data: bytes
    frames: int


class SetBuffer:
    """Bounded producer queue; only the main thread calls Panda/OpenAL APIs."""
    def __init__(self, engine: SetEngine):
        self.engine = engine
        self.queue = queue.Queue(maxsize=8)
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._produce, name='set-preparation', daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            for segment in self.engine.segments():
                for start in range(0, len(segment.samples), SAMPLE_RATE):
                    if self.cancel.is_set():
                        return
                    samples = segment.samples[start:start+SAMPLE_RATE]
                    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
                        raise ValueError('refusing invalid or clipping stream PCM')
                    data = np.rint(samples*32767).astype('<i2').tobytes()
                    chunk = AudioChunk(segment.span, data, len(samples))
                    while not self.cancel.is_set():
                        try:
                            self.queue.put(chunk, timeout=.1)
                            break
                        except queue.Full:
                            pass
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.finished.set()

    def close(self):
        self.cancel.set()
        self.thread.join()


class StreamLedger:
    """Pure audio-sample ownership lookup, including skipped render frames."""
    def __init__(self):
        self.spans: list[DeckSpan] = []
        self.starts: list[int] = []
        self.frames = 0

    def append(self, chunk: AudioChunk):
        if not self.spans or self.spans[-1] is not chunk.span:
            if chunk.span.start != self.frames:
                raise ValueError('non-contiguous set segment')
            self.spans.append(chunk.span)
            self.starts.append(chunk.span.start)
        if self.frames + chunk.frames > chunk.span.end:
            raise ValueError('chunk exceeds its segment')
        self.frames += chunk.frames

    def at(self, seconds: float) -> DeckSpan | None:
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('playback time must be finite and non-negative')
        if not self.spans:
            return None
        index = bisect_right(self.starts, frame(seconds))-1
        return self.spans[max(0, index)]
