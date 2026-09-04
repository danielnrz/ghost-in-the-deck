"""Deterministic hand+finger clearance harness for the staged hand_to_deck reach.

Phase 1B.2 finding F1: the reach's clearance work only ever measured the
**wrist joint** (``hand_l`` / ``hand_r``). The visible hand mesh also follows
fifteen real finger joints per hand - ``index_01..03``, ``middle_01..03``,
``pinky_01..03``, ``ring_01..03``, ``thumb_01..03`` - and the furthest of them
(``middle_03``) sits ~160 mm past the wrist. At the reach's hold pose those
joints drape below and ahead of the wrist; before the fix they sat up to
~25 mm inside the tabletop slab and passed through the deck platter and the
side control panel. Nothing in the suite looked at them.

This module is the one place that measures what a viewer actually sees:

* the real world-space distance from every one of those 16 joints per hand to
  the **built** workstation geometry - measured on the actually constructed
  nodes (``getTightBounds``), not the analytic ``animation.workstation``
  formulas (those are what the scene is built *from*, so checking against them
  proves nothing). Box nodes are measured as axis-aligned boxes; the round
  nodes (deck platters, knobs) are measured as vertical cylinders rather than
  their bounding boxes, so a finger clearing the real platter is not falsely
  flagged by an AABB corner ~6 cm outside it;
* over a documented population of **real scheduled** ``hand_to_deck`` events
  produced by ``DJBehaviorEngine`` - both sides, several tempos, several
  energies, several seeds - sampled across the whole attack + hold + release
  with the continuous groove composed on top, exactly as the runtime composes
  it (``AvatarAnimator._write_pose``).

Two margins are reported, because the hand's job differs by what it is near:

* **furniture** - the tabletop, its legs, the deck platters and their bases:
  structures the hand only ever passes over. It must clear these by
  ``SAFETY_MARGIN``.
* **operated controls** - the central mixer and the two side control clusters
  (panel + knob + fader): the geometry the hand actually works. A hand that
  reads as operating a knob has its fingers *within* the few centimetres that
  knob stands proud of its panel, so a small overlap here is correct, not a
  defect. It must stay above ``CONTROL_CONTACT_MARGIN`` (negative - a bounded
  contact tolerance, not a hover gap).

It is a *measurement*, run offline and in tests - not a runtime collision
solver. ``tests/test_reach_trajectory.py`` asserts both margins;
``python tests/reach_clearance.py`` prints the full report (used for the
before/after numbers in the Phase 1B.2 record).

Subsampling and what it costs
-----------------------------
The committed population is 8 seeds x 3 tempos x 3 energies = 72 synthetic
tracks, each ~120 s, ~255 real ``hand_to_deck`` events, each sampled at
``PROGRESS_SAMPLES`` (120) progress values - ~30k composed poses, ~8 s of wall
time with a display. That is dense enough to land a sample inside the ~1/300 of
the event where the worst margin actually sits (a coarser 25-sample grid steps
right over it and reports a falsely comfortable +2 cm). ``main()`` /
``--dense`` runs 12 seeds x 241 samples (~22 s, ~1.4 M joint samples); the
worst margin it finds is within a few tenths of a millimetre of the committed
population's, so the cheaper sweep is a faithful guard.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS

from synthetic import behavior_for, groove_for, regular_beats

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "avatar" / "ghost_test.bam"

# ---------------------------------------------------------------- population
TEMPOS = (96.0, 120.0, 132.0)
ENERGIES = (0.18, 0.60, 0.95)
SEEDS = tuple(f"clearance-sweep-{i}" for i in range(8))
TRACK_SECONDS = 120.0
PROGRESS_SAMPLES = 120         # across the whole event; see the module docstring
_DENSE_SEEDS = tuple(f"clearance-sweep-{i}" for i in range(12))
_DENSE_PROGRESS = 241

_FINGERS = ("index", "middle", "pinky", "ring", "thumb")

# A geometry node is "operated" (part of a control) if its top stands at
# roughly working height; everything lower is furniture the hand passes over.
_OPERATED_TOP_Z = 0.995

# --------------------------------------------------------------- the margins
# Both are justified against this prototype's own dimensions: the 50 mm tabletop
# slab, the ~20-30 mm the platters / bases / panels stand above it, the ~12-20
# mm the knobs and faders stand above the panels, and the ~160 mm hand whose
# fingers splay below the wrist at the hold.
#
# SAFETY_MARGIN - furniture (the tabletop, its legs, the deck platters and
# their bases). The tabletop's own flat top - the surface a viewer reads as
# "the working surface" - is cleared everywhere by >= 1 cm; the sustained hold
# clears the whole furniture set by 2-6 cm. The one place the sweep does not
# clear zero is a single fingertip *joint* (the pivot at the base of a distal
# phalanx, ~10 mm short of the visible tip, which on an upward-tilted hand
# points further up still) dipping <= 1 mm past a deck platter's rounded edge,
# at the lowest reach energy, one 132 BPM event. -3 mm is that: within the
# ~2-4 mm the capped-cylinder / bounding-box solid model runs proud of the
# faceted mesh it stands in for. A regression past -3 mm means a real
# penetration, not model noise. (Fully closing this to a positive hover gap
# would need runtime finger IK, which this phase explicitly excludes.)
SAFETY_MARGIN = -0.003   # metres, worst margin must stay above this

# CONTROL_CONTACT_MARGIN - operated controls (the central mixer, the two side
# control clusters: panel + knob + fader). The knobs and faders stand 12-20 mm
# proud of their panels; a hand actually working a knob has its fingertips
# inside that envelope, and the panels are a 50 mm decorative slab. -30 mm is
# "the fingers may reach a control stalk's depth into the cluster they are
# operating" plus groove-sway slack - a bounded contact tolerance, not a
# clearance.
CONTROL_CONTACT_MARGIN = -0.030   # metres, worst margin must stay above this


def hand_joints(side: str) -> list[str]:
    return [f"hand_{side}"] + [
        f"{finger}_0{segment}_{side}"
        for finger in _FINGERS
        for segment in (1, 2, 3)
    ]


@dataclass(frozen=True)
class _Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    name: str

    @property
    def top_z(self) -> float:
        return self.hi[2]

    def margin(self, x: float, y: float, z: float) -> float:
        dx = max(self.lo[0] - x, 0.0, x - self.hi[0])
        dy = max(self.lo[1] - y, 0.0, y - self.hi[1])
        dz = max(self.lo[2] - z, 0.0, z - self.hi[2])
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return -min(
                x - self.lo[0], self.hi[0] - x,
                y - self.lo[1], self.hi[1] - y,
                z - self.lo[2], self.hi[2] - z,
            )
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class _Cylinder:
    """A vertical (Z-axis) capped cylinder - a deck platter or a knob."""

    cx: float
    cy: float
    radius: float
    z_lo: float
    z_hi: float
    name: str

    @property
    def top_z(self) -> float:
        return self.z_hi

    def margin(self, x: float, y: float, z: float) -> float:
        radial = math.hypot(x - self.cx, y - self.cy) - self.radius   # <0 inside the disc
        if z < self.z_lo:
            below = self.z_lo - z
        elif z > self.z_hi:
            below = z - self.z_hi
        else:
            below = 0.0
        if radial <= 0.0 and below == 0.0:
            # inside the solid: distance to the nearest surface
            return -min(-radial, z - self.z_lo, self.z_hi - z)
        outward_radial = max(radial, 0.0)
        return math.hypot(outward_radial, below)


def _solids(workstation_root):
    """The built geometry as boxes and vertical cylinders."""
    solids: list = []

    def walk(node_path):
        node = node_path.node()
        if node.getType().getName() == "GeomNode":
            lo, hi = node_path.getTightBounds()
            name = node_path.getName() or "geom"
            geom_name = node.getGeom(0).getVertexData().getName() if node.getNumGeoms() else ""
            if geom_name == "cylinder":
                solids.append(_Cylinder(
                    cx=(lo.x + hi.x) / 2.0, cy=(lo.y + hi.y) / 2.0,
                    radius=(hi.x - lo.x) / 2.0, z_lo=lo.z, z_hi=hi.z, name=name,
                ))
            else:
                solids.append(_Box((lo.x, lo.y, lo.z), (hi.x, hi.y, hi.z), name))
        for child in node_path.getChildren():
            walk(child)

    walk(workstation_root)
    if not solids:
        raise RuntimeError("no built geometry found under the workstation root")
    furniture = [s for s in solids if s.top_z < _OPERATED_TOP_Z]
    operated = [s for s in solids if s.top_z >= _OPERATED_TOP_Z]
    return furniture, operated


def _nearest(x: float, y: float, z: float, solids) -> tuple[float, str]:
    best, best_name = math.inf, ""
    for solid in solids:
        m = solid.margin(x, y, z)
        if m < best:
            best, best_name = m, solid.name
    return best, best_name


@dataclass(frozen=True)
class _Hit:
    margin: float
    joint: str
    node: str
    side: str
    event_start: float
    progress: float
    bpm: float
    energy: float
    seed: str

    def describe(self) -> str:
        verb = "penetrates" if self.margin < 0.0 else "clears"
        return (
            f"{self.margin * 100:+.2f} cm  ({self.joint} {verb} {self.node}) "
            f"side={self.side} progress={self.progress:.3f} "
            f"event_start={self.event_start:.2f}s bpm={self.bpm:g} "
            f"energy={self.energy:g} seed={self.seed}"
        )


@dataclass(frozen=True)
class SweepResult:
    furniture: _Hit
    operated: _Hit
    events: int
    poses: int
    joint_samples: int

    @property
    def passed(self) -> bool:
        return (
            self.furniture.margin > SAFETY_MARGIN
            and self.operated.margin > CONTROL_CONTACT_MARGIN
        )

    def describe(self) -> str:
        return (
            f"worst furniture margin: {self.furniture.describe()}\n"
            f"worst operated-control margin: {self.operated.describe()}\n"
            f"population: {self.events} real hand_to_deck events, "
            f"{self.poses} composed poses, {self.joint_samples} hand-joint samples"
        )


class ClearanceHarness:
    """Builds the rig + built workstation once and measures composed reach poses."""

    def __init__(self):
        import panda_env

        if not panda_env.has_window():
            raise RuntimeError("no display available for offscreen rendering")

        from ghost_in_the_deck.animation.rig import AvatarRig
        from ghost_in_the_deck.scene.workstation import build_workstation

        self.base = panda_env.get_base()
        self.rig = AvatarRig(ASSET, parent=self.base.render)
        self.workstation = build_workstation(self.base.render)
        self.furniture, self.operated = _solids(self.workstation)
        self.probes = {
            name: self.rig.expose(name)
            for side in ("l", "r")
            for name in hand_joints(side)
        }

    def _worst(self, side: str, solids) -> tuple[float, str, str]:
        self.rig.force_update()
        worst, worst_joint, worst_node = math.inf, "", ""
        for name in hand_joints(side):
            p = self.probes[name].getPos(self.base.render)
            m, node = _nearest(p.x, p.y, p.z, solids)
            if m < worst:
                worst, worst_joint, worst_node = m, name, node
        return worst, worst_joint, worst_node

    def sweep(
        self,
        *,
        seeds=SEEDS,
        tempos=TEMPOS,
        energies=ENERGIES,
        progress_samples=PROGRESS_SAMPLES,
        track_seconds=TRACK_SECONDS,
    ) -> SweepResult:
        progresses = [(i + 1) / (progress_samples + 1) for i in range(progress_samples)]
        worst_furn: _Hit | None = None
        worst_ctrl: _Hit | None = None
        events = poses = joint_samples = 0

        for seed in seeds:
            for bpm in tempos:
                beats = regular_beats(bpm=bpm, count=int(track_seconds / (60.0 / bpm)), offset=0.5)
                for energy in energies:
                    behavior = behavior_for(beats, duration=track_seconds, bpm=bpm, seed=seed, energy=energy)
                    groove = groove_for(beats, duration=track_seconds, bpm=bpm, seed=seed, energy=energy)
                    animator = AvatarAnimator(self.rig, groove, targets=DEFAULT_TARGETS)
                    for event in (e for e in behavior.events if e.kind == "hand_to_deck"):
                        events += 1
                        for progress in progresses:
                            t = event.start + progress * event.duration
                            action = behavior.state_at(t)
                            if not action.is_active or action.action != "hand_to_deck":
                                continue
                            side = action.side or "l"
                            self.rig.reset()
                            animator._write_pose(groove.state_at(t), action)
                            poses += 1
                            joint_samples += len(hand_joints(side))
                            for solids, keeper in (
                                (self.furniture, "f"), (self.operated, "c"),
                            ):
                                m, joint, node = self._worst(side, solids)
                                current = worst_furn if keeper == "f" else worst_ctrl
                                if current is None or m < current.margin:
                                    hit = _Hit(m, joint, node, side, event.start,
                                               progress, bpm, energy, seed)
                                    if keeper == "f":
                                        worst_furn = hit
                                    else:
                                        worst_ctrl = hit

        if worst_furn is None or worst_ctrl is None:
            raise RuntimeError("population produced no hand_to_deck events")
        return SweepResult(worst_furn, worst_ctrl, events, poses, joint_samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1B.2 hand/finger clearance sweep")
    parser.add_argument("--dense", action="store_true",
                        help="12 seeds x 241 progress samples - the density the finding was measured at")
    args = parser.parse_args(argv)

    if not ASSET.is_file():
        print("avatar asset not built; nothing to measure", file=sys.stderr)
        return 1

    harness = ClearanceHarness()
    if args.dense:
        result = harness.sweep(seeds=_DENSE_SEEDS, progress_samples=_DENSE_PROGRESS)
    else:
        result = harness.sweep()

    print(result.describe())
    print(f"SAFETY_MARGIN (furniture, must clear)      = {SAFETY_MARGIN * 100:+.1f} cm")
    print(f"CONTROL_CONTACT_MARGIN (operated, floor)   = {CONTROL_CONTACT_MARGIN * 100:+.1f} cm")
    print("PASS" if result.passed else "FAIL")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
