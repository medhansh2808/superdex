"""mjlab integration for Superdex's LEAP Hand calibration scene."""

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from superdex.leap import (
    CUBE_START_POSITION,
    LEAP_JOINTS,
    LEAP_PALM_POSITION,
    LEAP_PALM_ROTATION,
    cube_spec,
    leap_spec,
)


def _hand_entity_spec() -> mujoco.MjSpec:
    spec = leap_spec()
    # mjlab applies these options on the parent scene when it builds the simulation.
    defaults = mujoco.MjSpec().option
    spec.option.integrator = defaults.integrator
    spec.option.cone = defaults.cone
    spec.option.impratio = defaults.impratio
    return spec


def leap_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_hand_entity_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=LEAP_PALM_POSITION, rot=LEAP_PALM_ROTATION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=LEAP_JOINTS, command_field="position"),
            ),
        ),
    )


def calibration_scene_cfg(num_envs: int) -> SceneCfg:
    return SceneCfg(
        num_envs=num_envs,
        entities={
            "hand": leap_entity_cfg(),
            "cube": EntityCfg(
                spec_fn=cube_spec,
                init_state=EntityCfg.InitialStateCfg(pos=CUBE_START_POSITION),
            ),
        },
    )


def leap_simulation_cfg() -> SimulationCfg:
    # MjSpec attachment does not propagate the hand's Menagerie <option> block.
    return SimulationCfg(
        mujoco=MujocoCfg(integrator="implicitfast", cone="elliptic", impratio=100)
    )
