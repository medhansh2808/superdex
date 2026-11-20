"""Right-hand LEAP model and a fixed-palm cube calibration scene."""

from pathlib import Path

import mujoco

LEAP_JOINTS = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)
LEAP_PALM_POSITION = (0.0, 0.0, 0.1)
LEAP_PALM_ROTATION = (0.0, 1.0, 0.0, 0.0)
CUBE_START_POSITION = (-0.035, 0.04, 0.18)

_LEAP_MJCF = Path(__file__).parent / "assets" / "leap_hand" / "right_hand.xml"


def leap_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_LEAP_MJCF))


def cube_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    cube = spec.worldbody.add_body(name="object", pos=CUBE_START_POSITION)
    cube.add_freejoint(name="free")
    cube.add_geom(
        name="surface",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.018, 0.018, 0.018),
        mass=0.05,
        friction=(0.8, 0.005, 0.0001),
        rgba=(0.16, 0.45, 0.66, 1.0),
    )
    cube.add_geom(
        name="mark",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.004, 0.003, 0.0002),
        pos=(0.008, -0.005, 0.018),
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=(0.95, 0.8, 0.3, 1.0),
    )
    return spec


def calibration_spec() -> mujoco.MjSpec:
    """Place a free cube above the neutral palm for contact and reset checks."""
    spec = leap_spec()
    spec.attach(cube_spec(), prefix="cube/", frame=spec.worldbody.add_frame())
    return spec


def grasp_spec() -> mujoco.MjSpec:
    """Mount the fixed palm sideways so opening the fingers lets the cube fall."""
    spec = calibration_spec()
    palm = spec.body("palm")
    palm.pos = (0.0, 0.0, 0.25)
    palm.quat = (2**-0.5, 2**-0.5, 0.0, 0.0)
    return spec
