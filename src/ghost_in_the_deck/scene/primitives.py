"""Minimal procedural mesh builders: boxes and cylinders.

The pip-distributed panda3d package ships no primitive models (no "models/box"
egg the way the full SDK samples do), so the small amount of geometry the DJ
workstation needs - a tabletop, platters, a mixer body, a few knobs and faders
- is built here directly from vertex data. Kept to exactly the two shapes that
are needed; this is not a general modelling toolkit.
"""

from __future__ import annotations

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def _empty_geom(name: str) -> tuple[GeomVertexData, GeomVertexWriter, GeomVertexWriter, GeomTriangles]:
    fmt = GeomVertexFormat.getV3n3()
    data = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(data, "vertex")
    normal = GeomVertexWriter(data, "normal")
    tris = GeomTriangles(Geom.UHStatic)
    return data, vertex, normal, tris


def _quad(vertex, normal, tris, points, face_normal) -> None:
    """One flat quad from four corner points, wound consistently."""
    start = vertex.getWriteRow()
    for point in points:
        vertex.addData3(*point)
        normal.addData3(*face_normal)
    tris.addVertices(start, start + 1, start + 2)
    tris.addVertices(start, start + 2, start + 3)


def box(width: float, depth: float, height: float, color=(0.5, 0.5, 0.5, 1.0)) -> NodePath:
    """An axis-aligned box, ``width`` along X, ``depth`` along Y, ``height`` along Z.

    Centred on X and Y; sits on the local XY plane (bottom at Z=0, top at
    Z=height), so placing one is just a matter of positioning its origin where
    the bottom-centre should be.
    """
    x, y, z = width / 2.0, depth / 2.0, height
    data, vertex, normal, tris = _empty_geom("box")

    faces = [
        ([(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)], (0, 0, -1)),   # bottom
        ([(-x, -y, z), (-x, y, z), (x, y, z), (x, -y, z)], (0, 0, 1)),    # top
        ([(-x, -y, 0), (-x, y, 0), (-x, y, z), (-x, -y, z)], (-1, 0, 0)),  # -X
        ([(x, -y, 0), (x, -y, z), (x, y, z), (x, y, 0)], (1, 0, 0)),      # +X
        ([(-x, -y, 0), (-x, -y, z), (x, -y, z), (x, -y, 0)], (0, -1, 0)),  # -Y
        ([(-x, y, 0), (x, y, 0), (x, y, z), (-x, y, z)], (0, 1, 0)),      # +Y
    ]
    for points, face_normal in faces:
        _quad(vertex, normal, tris, points, face_normal)

    geom = Geom(data)
    geom.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(geom)
    node_path = NodePath(node)
    node_path.setColor(*color)
    return node_path


def cylinder(radius: float, height: float, color=(0.5, 0.5, 0.5, 1.0), segments: int = 20) -> NodePath:
    """A capped cylinder standing on the local XY plane, axis along Z.

    Low-poly on purpose (``segments`` defaults to 20) - this stands in for a
    turntable platter or a knob, seen from a few metres away, not a hero asset.
    """
    data, vertex, normal, tris = _empty_geom("cylinder")

    top_centre_row = None
    bottom_centre_row = None
    ring_bottom: list[int] = []
    ring_top: list[int] = []

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)
        side_normal = (math.cos(angle), math.sin(angle), 0.0)

        ring_bottom.append(vertex.getWriteRow())
        vertex.addData3(cx, cy, 0.0)
        normal.addData3(*side_normal)

        ring_top.append(vertex.getWriteRow())
        vertex.addData3(cx, cy, height)
        normal.addData3(*side_normal)

    for i in range(segments):
        j = (i + 1) % segments
        tris.addVertices(ring_bottom[i], ring_bottom[j], ring_top[j])
        tris.addVertices(ring_bottom[i], ring_top[j], ring_top[i])

    bottom_centre_row = vertex.getWriteRow()
    vertex.addData3(0.0, 0.0, 0.0)
    normal.addData3(0.0, 0.0, -1.0)
    for i in range(segments):
        j = (i + 1) % segments
        angle_i, angle_j = 2 * math.pi * i / segments, 2 * math.pi * j / segments
        row_i = vertex.getWriteRow()
        vertex.addData3(radius * math.cos(angle_i), radius * math.sin(angle_i), 0.0)
        normal.addData3(0.0, 0.0, -1.0)
        row_j = vertex.getWriteRow()
        vertex.addData3(radius * math.cos(angle_j), radius * math.sin(angle_j), 0.0)
        normal.addData3(0.0, 0.0, -1.0)
        tris.addVertices(bottom_centre_row, row_j, row_i)

    top_centre_row = vertex.getWriteRow()
    vertex.addData3(0.0, 0.0, height)
    normal.addData3(0.0, 0.0, 1.0)
    for i in range(segments):
        j = (i + 1) % segments
        angle_i, angle_j = 2 * math.pi * i / segments, 2 * math.pi * j / segments
        row_i = vertex.getWriteRow()
        vertex.addData3(radius * math.cos(angle_i), radius * math.sin(angle_i), height)
        normal.addData3(0.0, 0.0, 1.0)
        row_j = vertex.getWriteRow()
        vertex.addData3(radius * math.cos(angle_j), radius * math.sin(angle_j), height)
        normal.addData3(0.0, 0.0, 1.0)
        tris.addVertices(top_centre_row, row_i, row_j)

    geom = Geom(data)
    geom.addPrimitive(tris)
    node = GeomNode("cylinder")
    node.addGeom(geom)
    node_path = NodePath(node)
    node_path.setColor(*color)
    return node_path
