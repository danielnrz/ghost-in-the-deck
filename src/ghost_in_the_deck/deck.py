"""One analysed track, wrapped for two-deck planning.

``TrackDeck`` bundles a single track's ``MusicFeatures`` with the derived
objects the planning layer needs - the beat grid, the groove-scale energy
curve, and a second energy curve smoothed over the phrase timescale for the
broad-structure layer. It reuses the existing analysis machinery exactly as
``app.py`` does: no new librosa pass, no feature-cache access, no Panda3D.

``TrackDeck`` takes an already-analysed ``MusicFeatures``; building one never
calls ``audio.analysis.analyse`` or touches a cache file.
"""

from __future__ import annotations

from dataclasses import dataclass

from .animation import structure as _structure
from .animation.cues import BeatTimeline
from .animation.energy import EnergyTrack
from .animation.structure import MusicalStructure
from .audio.features import MusicFeatures


@dataclass(frozen=True)
class TrackDeck:
    """One analysed track and its planning-time derived views.

    A pure function of ``features``: ``TrackDeck.from_features`` given the same
    ``MusicFeatures`` twice produces decks with identical fields.
    """

    features: MusicFeatures
    timeline: BeatTimeline
    energy: EnergyTrack
    broad_energy: EnergyTrack

    @classmethod
    def from_features(cls, features: MusicFeatures) -> "TrackDeck":
        """Build a deck from an already-analysed ``MusicFeatures``.

        ``timeline`` and ``energy`` are constructed exactly as
        ``GhostApp.__init__`` builds them. ``broad_energy`` is the same
        ``EnergyTrack`` at the phrase smoothing window the broad-structure
        layer expects.
        """
        return cls(
            features=features,
            timeline=BeatTimeline(features),
            energy=EnergyTrack(features),
            broad_energy=EnergyTrack(
                features,
                smoothing_seconds=_structure.PHRASE_SMOOTHING_SECONDS,
            ),
        )

    @property
    def track(self) -> str:
        return self.features.track

    @property
    def duration(self) -> float:
        return self.features.duration_seconds

    @property
    def bpm(self) -> float:
        return self.features.bpm

    def structure_at(self, time: float) -> MusicalStructure:
        """The broad-structure reading at ``time`` - a thin wrapper."""
        return _structure.structure_at(self.broad_energy, self.timeline, time)
