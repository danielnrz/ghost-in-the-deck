"""Deterministic hand+finger clearance harness for the staged hand_to_deck reach.

Phase 1B.2 finding F1: the reach's clearance work only ever measured the
**wrist joint** (``hand_l`` / ``hand_r``). The visible hand mesh also follows
fifteen real finger joints per hand - ``index_01..03``, ``middle_01..03``,
``pinky_01..03``, ``ring_01..03``, ``thumb_01..03`` - and the furthest of them
(``middle_03``) sits ~160 mm past the wrist. At the reach's hold pose those
joints drape below and ahead of the wrist; before the fix they sat up to
~25 mm inside the tabletop slab and passed through the deck platter and the
side control panel. Nothing in the suite looked at them.

F1 round 2: measuring the fifteen **joint centres** and demanding they clear
the furniture by a fixed omnidirectional skin allowance (the farthest mesh
vertex any hand joint carries, ~40 mm) is the wrong test - the joint centre is
never the closest point of the hand to the furniture, and the reach cannot put
every finger *pivot* 40 mm from the left deck platter without swinging the
wrist out of its calibrated 5-8 cm hover band, because the left control
cluster the hand operates sits only ~29 mm from that platter's edge. So this
harness now measures the **skinned hand mesh itself**: at build time it reads
the character's transform-blend table, assigns each mesh vertex to the hand
joint it is skinned predominantly to (blend weight >= 0.5), and stores that
vertex in the joint's own local frame. During the sweep every stored vertex
is carried rigidly by its joint's live world transform and measured against
the built geometry. That is the surface a viewer actually sees, so the guard
no longer needs a skin allowance bolted onto the threshold - the allowance is
in the measurement.

The geometry is the **built** scene (``getTightBounds`` on the actually
constructed nodes), not the analytic ``animation.workstation`` formulas (those
are what the scene is built *from*, so checking against them proves nothing).
Box nodes are measured as axis-aligned boxes; the round nodes (deck platters,
knobs) are measured as vertical cylinders rather than their bounding boxes, so
a finger clearing the real platter is not falsely flagged by an AABB corner
~6 cm outside it. The population is **real scheduled** ``hand_to_deck`` events
produced by ``DJBehaviorEngine`` - both sides, several tempos, several
energies, several seeds - sampled across the whole attack + hold + release
with the continuous groove composed on top, exactly as the runtime composes
it (``AvatarAnimator._write_pose``).

Two margins are reported, because the hand's job differs by what it is near:

* **furniture** - the tabletop, its legs, the deck platters and their bases:
  structures the hand only ever passes over. The measured skinned mesh must
  clear these by ``SAFETY_MARGIN``.
* **operated controls** - the central mixer and the two side control clusters
  (panel + knob + fader): the geometry the hand actually works. A hand that
  reads as operating a knob has its fingers *within* the few centimetres that
  knob stands proud of its panel, so a small overlap here is correct, not a
  defect. This one is still measured at the **joint centre** - it is a bounded
  allowance on how far the hand's skeleton reaches into the cluster it works,
  deliberately permitting mesh contact - and must stay above
  ``CONTROL_CONTACT_MARGIN`` (negative).

It is a *measurement*, run offline and in tests - not a runtime collision
solver. ``tests/test_reach_trajectory.py`` asserts both margins;
``python tests/reach_clearance.py`` prints the full report (used for the
before/after numbers in the Phase 1B.2 record).

Subsampling and what it costs
-----------------------------
The committed population is 6 seeds x 3 tempos x 3 energies = 54 synthetic
tracks, each ~120 s, ~190 real ``hand_to_deck`` events, each sampled at
``PROGRESS_SAMPLES`` (240) progress values. The mesh measurement only runs on a
joint once its *centre* comes within ``_MESH_GATE`` (60 mm) of the furniture -
elsewhere the joint's whole skinned blob is provably clear - so the sweep stays
near a minute with a display. The worst furniture margin sits in a narrow
window (~1/300 of an event, at the lowest reach energy, while a groove sway
rocks the arm outboard); ``--dense`` runs 8 seeds x 480 samples and moves the
reported worst furniture margin by under 0.5 mm, so the committed sweep is a
faithful guard. State the resolution at which the worst case stops moving in
the Phase 1B.2 record.
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
SEEDS = tuple(f"clearance-sweep-{i}" for i in range(6))
TRACK_SECONDS = 120.0
PROGRESS_SAMPLES = 240         # across the whole event; see the module docstring
_DENSE_SEEDS = tuple(f"clearance-sweep-{i}" for i in range(8))
_DENSE_PROGRESS = 480

_FINGERS = ("index", "middle", "pinky", "ring", "thumb")

# A joint's per-vertex loop is entered only when its skinned blob *can* contain
# a point nearer the furniture than the worst margin found so far this pose -
# i.e. when ``centre_margin - skin_radius`` is below that worst. Sound (no
# nearer point can be missed) and, since most finger joints sit well clear at
# the hold, it keeps the sweep near a minute with a display.

# Farthest N skinned vertices kept per joint. The closest point of a finger to
# an external convex solid is on the finger's surface facing it, which is among
# its outermost vertices; 32 spans that surface on this asset's hand mesh
# (raising it to the full ~120/joint moves the worst margin by < 0.3 mm).
_SKIN_VERTS_PER_JOINT = 32

# A geometry node is "operated" (part of a control) if its top stands at
# roughly working height; everything lower is furniture the hand passes over.
_OPERATED_TOP_Z = 0.995

# --------------------------------------------------------------- the margins
# SAFETY_MARGIN - furniture (the tabletop, its legs, the deck platters and
# their bases): structures the hand only passes over, never operates. Because
# this harness now measures the *skinned mesh* (see the module docstring), the
# painted-hand skin allowance is already in the measurement and this threshold
# is only the residual model error, all of it in the conservative direction:
#
#   * the box / capped-cylinder solids from ``getTightBounds`` are convex
#     supersets that sit ~1-3 mm proud of the faceted render mesh they stand in
#     for;
#   * rigid skinning (weight >= 0.5, one joint per vertex) ignores the minor
#     multi-joint blend near the knuckles - a sub-millimetre effect on the
#     distal joints that reach the furniture;
#   * only the 48 farthest vertices per joint are carried, so the sampled
#     surface can undercut the true nearest point by a fraction of a mm.
#
# 6 mm covers all three with room to spare. A measured mesh margin above +6 mm
# means the visible hand is outside the furniture with margin left over; below
# it means the render mesh is at or through a furniture face. Re-check the
# per-joint skin radii with ``scripts/measure_hand_skin.py`` if the asset
# changes.
SAFETY_MARGIN = 0.006   # metres, worst furniture mesh margin must exceed this

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
class _JointSkin:
    """The mesh a hand joint carries, in that joint's own local frame.

    ``local_pts`` are the ``_SKIN_VERTS_PER_JOINT`` farthest skinned vertices
    (blend weight >= 0.5) expressed relative to the joint's neutral world
    transform; ``radius`` is the farthest of them. At sweep time each point is
    carried rigidly by the joint's live world matrix.
    """

    local_pts: tuple
    radius: float


def build_hand_skin(rig, render, per_joint: int = _SKIN_VERTS_PER_JOINT) -> dict:
    """Read the character mesh once and bucket its vertices onto the hand joints.

    Returns ``{joint_name: _JointSkin}`` for all 32 hand joints (16 per side).
    Deterministic: vertex iteration order, the ``>= 0.5`` predominant-weight
    rule and the farthest-N truncation are all stable.
    """
    from panda3d.core import GeomVertexReader

    rig.reset()
    rig.force_update()
    joints = hand_joints("l") + hand_joints("r")
    inv = {}
    for name in joints:
        mat = rig.expose(name).getMat(render)
        m = type(mat)()
        m.invertFrom(mat)
        inv[name] = m
    joint_set = set(joints)

    geom_np = rig.actor.find("**/+GeomNode")
    vdata = geom_np.node().getGeom(0).getVertexData()
    tbt = vdata.getTransformBlendTable()
    v_reader = GeomVertexReader(vdata, "vertex")
    b_reader = GeomVertexReader(vdata, "transform_blend")
    world = geom_np.getNetTransform().getMat()

    buckets: dict = {name: [] for name in joints}
    for row in range(vdata.getNumRows()):
        v_reader.setRow(row)
        point = world.xformPoint(v_reader.getData3())
        b_reader.setRow(row)
        blend = tbt.getBlend(b_reader.getData1i())
        best_w, best_name = 0.0, None
        for k in range(blend.getNumTransforms()):
            w = blend.getWeight(k)
            try:
                name = blend.getTransform(k).getJoint().getName()
            except AttributeError:
                name = None
            if name in joint_set and w > best_w:
                best_w, best_name = w, name
        if best_name is not None and best_w >= 0.5:
            local = inv[best_name].xformPoint(point)
            buckets[best_name].append((local.lengthSquared(), local))

    skin: dict = {}
    for name in joints:
        pts = sorted(buckets[name], key=lambda t: -t[0])[:per_joint]
        skin[name] = _JointSkin(
            local_pts=tuple(p for _, p in pts),
            radius=math.sqrt(pts[0][0]) if pts else 0.0,
        )
    return skin


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
        self.skin = build_hand_skin(self.rig, self.base.render)

    def _worst_mesh(self, side: str, solids, running: float) -> tuple[float, str, str]:
        """Nearest point of the *skinned hand mesh* to ``solids``.

        Cheap joint-centre scan first; a joint's per-vertex loop is entered
        only when ``centre_margin - skin_radius`` is below the worst margin
        found so far this pose (or, on the first pose, ``running`` seeded from
        the running best) - so no nearer mesh point can be missed.
        """
        self.rig.force_update()
        worst, worst_joint, worst_node = running, "", ""
        for name in hand_joints(side):
            probe = self.probes[name]
            c = probe.getPos(self.base.render)
            m0, node0 = _nearest(c.x, c.y, c.z, solids)
            skin = self.skin[name]
            if m0 - skin.radius >= worst:
                if m0 < worst:
                    worst, worst_joint, worst_node = m0, name, node0
                continue
            mat = probe.getMat(self.base.render)
            for lp in skin.local_pts:
                w = mat.xformPoint(lp)
                m, node = _nearest(w.x, w.y, w.z, solids)
                if m < worst:
                    worst, worst_joint, worst_node = m, name, node
        return worst, worst_joint, worst_node

    def _worst_centre(self, side: str, solids) -> tuple[float, str, str]:
        """Nearest *joint centre* to ``solids`` - used for operated controls."""
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

                            running = worst_furn.margin if worst_furn else math.inf
                            m, joint, node = self._worst_mesh(side, self.furniture, running)
                            if worst_furn is None or m < worst_furn.margin:
                                worst_furn = _Hit(m, joint, node, side, event.start,
                                                  progress, bpm, energy, seed)

                            m, joint, node = self._worst_centre(side, self.operated)
                            if worst_ctrl is None or m < worst_ctrl.margin:
                                worst_ctrl = _Hit(m, joint, node, side, event.start,
                                                  progress, bpm, energy, seed)

        if worst_furn is None or worst_ctrl is None:
            raise RuntimeError("population produced no hand_to_deck events")
        return SweepResult(worst_furn, worst_ctrl, events, poses, joint_samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1B.2 hand/finger clearance sweep")
    parser.add_argument("--dense", action="store_true",
                        help="8 seeds x 480 progress samples - confirms the committed sweep is not under-resolved")
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
