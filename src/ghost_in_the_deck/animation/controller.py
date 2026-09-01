"""Turns a GrooveState into joint rotations.

The neutral standing pose lives in the rig; this module produces only the
movement stacked on top of it, scaled by how energetic the music is.

It is a pure function of absolute playback time, so nothing is integrated
across frames and the sequence of frames drawn before a moment cannot change the
pose produced at it. A renderer that stalls for a second misses frames; when it
comes back it draws the pose the music calls for now.

This module knows nothing about audio files or Panda3D windows. It reads a
GrooveState and writes joint offsets to an AvatarRig, which clamps them.
"""

from __future__ import annotations

from .groove import GrooveEngine, GrooveState

# ---------------------------------------------------------------------- gains
# Peak degrees at full intensity. Kept together so the body can be retuned in
# one place; every one of them is multiplied by the music's intensity.
#
# Retuned in Phase 1A.1 after the skeleton was aligned with the mesh. Before
# that every joint pivoted most of a metre from the body part it drove, so the
# same angles threw the mesh around far more than they should have; with correct
# pivots the upper body needed more travel and the legs much less. Leg gains are
# deliberately small: the pelvis is the root, so bending a knee moves the foot
# rather than lowering the body, and without IK anything larger reads as
# sliding.
PELVIS_WEIGHT_ROLL = 1.5
PELVIS_SWAY_HEADING = 1.4
SPINE1_COUNTER_ROLL = 3.0          # keeps the torso over the feet
SPINE2_SWAY_ROLL = 3.4
SPINE2_SWAY_HEADING = 3.6
SPINE3_SWAY_ROLL = 2.4
SPINE3_BOUNCE_PITCH = 3.2
NECK_BOUNCE_PITCH = 3.2
NECK_PULSE_PITCH = 4.5
HEAD_BOUNCE_PITCH = 4.5
HEAD_PULSE_PITCH = 8.0
HEAD_BIAS_HEADING = 3.0
HEAD_SWAY_HEADING = 2.0
HEAD_SWAY_ROLL = 2.5
HEAD_BREATH_PITCH = 1.2
CLAVICLE_BOUNCE = 2.4
CLAVICLE_PULSE = 5.0
UPPERARM_BOUNCE_PITCH = 2.6
UPPERARM_SWAY_ROLL = 2.4
LOWERARM_BOUNCE_PITCH = 3.0
THIGH_WEIGHT_ROLL = 0.8
KNEE_BOUNCE_PITCH = 1.4
KNEE_WEIGHT_PITCH = 1.2

# How strongly the per-bar variation is allowed to unbalance the two sides.
ASYMMETRY = 0.25


class AvatarAnimator:
    """Evaluates the pose for a playback time and writes it to the rig."""

    def __init__(self, rig, groove: GrooveEngine | None = None):
        self.rig = rig
        self.groove = groove

    @property
    def response_window(self) -> float:
        """Reporting threshold shared with the timing instrumentation."""
        return self.groove.response_window if self.groove else 0.48

    def state_at(self, time: float) -> GrooveState | None:
        return self.groove.state_at(time) if self.groove else None

    def apply_at(self, time: float) -> GrooveState | None:
        """Evaluate the pose for ``time`` and write it to the rig."""
        state = self.state_at(time)
        if state is None:
            self.rig.reset()
            return None
        self._write_pose(state)
        return state

    def reset(self) -> None:
        """Neutral stance with no movement on top."""
        self.rig.reset()

    def pose_offsets(self, state: GrooveState) -> dict[str, tuple[float, float, float]]:
        """The movement as joint -> (heading, pitch, roll), measured from neutral.

        The neutral stance itself lives in the rig, so these are purely the
        groove. Separated from writing them so tests and the debug output can
        look at the numbers without needing a rig.
        """
        scale = state.intensity * state.variation.emphasis
        bounce = state.bounce
        pulse = state.pulse
        sway = state.sway
        weight = state.weight_shift
        breath = state.breath

        # Human movement is not mirror-symmetric; one side always works harder.
        bias = state.variation.shoulder_bias * ASYMMETRY
        left = 1.0 + bias
        right = 1.0 - bias
        lead = state.variation.lead_side

        offsets: dict[str, list[float]] = {}

        def add(joint: str, heading: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
            current = offsets.setdefault(joint, [0.0, 0.0, 0.0])
            current[0] += heading
            current[1] += pitch
            current[2] += roll

        # Hips lead the weight shift, but only slightly: the pelvis is the root,
        # so rolling it swings the legs and drags the feet across the floor. Most
        # of the visible weight shift is carried by the spine above it, which
        # costs the feet nothing.
        add("pelvis", heading=sway * PELVIS_SWAY_HEADING * scale,
            roll=weight * PELVIS_WEIGHT_ROLL * scale)
        add("spine_01", roll=-weight * SPINE1_COUNTER_ROLL * scale)

        add("spine_02", heading=sway * SPINE2_SWAY_HEADING * scale,
            roll=sway * SPINE2_SWAY_ROLL * scale)
        add("spine_03", pitch=bounce * SPINE3_BOUNCE_PITCH * scale,
            roll=sway * SPINE3_SWAY_ROLL * scale)

        add("neck_01", pitch=(bounce * NECK_BOUNCE_PITCH + pulse * NECK_PULSE_PITCH) * scale)
        add(
            "head",
            heading=(state.variation.head_bias * HEAD_BIAS_HEADING
                     + sway * HEAD_SWAY_HEADING) * scale,
            pitch=(bounce * HEAD_BOUNCE_PITCH + pulse * HEAD_PULSE_PITCH) * scale
                  + breath * HEAD_BREATH_PITCH,
            roll=-sway * HEAD_SWAY_ROLL * scale,
        )

        shoulder = (bounce * CLAVICLE_BOUNCE + pulse * CLAVICLE_PULSE) * scale
        add("clavicle_l", pitch=-shoulder * left)
        add("clavicle_r", pitch=-shoulder * right)

        arm = bounce * UPPERARM_BOUNCE_PITCH * scale
        add("upperarm_l", pitch=arm * left, roll=sway * UPPERARM_SWAY_ROLL * scale)
        add("upperarm_r", pitch=arm * right, roll=-sway * UPPERARM_SWAY_ROLL * scale)
        add("lowerarm_l", pitch=bounce * LOWERARM_BOUNCE_PITCH * scale * left)
        add("lowerarm_r", pitch=bounce * LOWERARM_BOUNCE_PITCH * scale * right)

        # The leg carrying the weight straightens; the other softens. Subtle on
        # purpose - the feet are planted, so anything larger reads as sliding.
        add("thigh_l", roll=weight * THIGH_WEIGHT_ROLL * scale)
        add("thigh_r", roll=weight * THIGH_WEIGHT_ROLL * scale)
        knee = bounce * KNEE_BOUNCE_PITCH * scale
        carry = weight * KNEE_WEIGHT_PITCH * scale * lead
        add("calf_l", pitch=knee - carry)
        add("calf_r", pitch=knee + carry)

        return {name: (v[0], v[1], v[2]) for name, v in offsets.items()}

    def _write_pose(self, state: GrooveState) -> None:
        for name, (heading, pitch, roll) in self.pose_offsets(state).items():
            self.rig.set_offset(name, heading=heading, pitch=pitch, roll=roll)
