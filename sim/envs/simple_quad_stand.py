"""Standing/balance environment for the simple Onshape quadruped."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "sim/robots/simple_quad_v0/mjcf/simple_quad_v0.xml"
CONTROL_RATE_HZ = 50
EPISODE_SECONDS = 8.0
TARGET_BASE_HEIGHT = 0.17
STABLE_BASE_HEIGHT = 0.12
ACTION_SCALE_RAD = 0.35
SERVO_VELOCITY_LIMIT_RAD_S = 8.0

ACTUATED_JOINTS = [
    "revolute_back_left_hip",
    "revolute_back_left_knee",
    "revolute_back_right_hip",
    "revolute_back_right_knee",
    "revolute_front_left_hip",
    "revolute_front_left_knee",
    "revolute_front_right_hip",
    "revolute_front_right_knee",
]

STAND_POSE = {
    "revolute_back_left_hip": 0.0,
    "revolute_back_left_knee": 0.0,
    "revolute_back_right_hip": 0.0,
    "revolute_back_right_knee": 0.0,
    "revolute_front_left_hip": 0.0,
    "revolute_front_left_knee": 0.0,
    "revolute_front_right_hip": 0.0,
    "revolute_front_right_knee": 0.0,
}

# Measured from short MuJoCo rollouts of this generated simple_quad_v0 model.
# Each primitive is (dx, dy, amp_h, amp_k, knee_bias, frequency_hz, steer).
TARGET_GAIT_PRIMITIVES = [
    (-0.075, -0.011, 0.3, 0.9, -0.2, 1.3, 0.6),
    (-0.189, -0.300, 0.3, 0.5, 0.1, 0.7, 0.6),
    (-0.075, -0.342, 0.7, 0.5, 0.1, 0.7, 1.2),
    (0.034, -0.279, 0.7, 0.7, 0.1, 0.7, 1.2),
    (0.084, -0.111, 0.9, 0.7, -0.2, 1.3, 1.2),
    (0.159, -0.064, 0.7, 0.7, -0.5, 1.3, 1.2),
    (0.329, 0.186, 0.9, 0.7, -0.2, 0.7, 0.0),
    (0.263, 0.378, 0.5, 0.9, -0.5, 0.7, 0.0),
    (0.122, 0.356, 0.9, 0.9, 0.1, 1.0, -1.2),
    (-0.161, 0.364, 0.5, 0.7, 0.1, 1.3, -1.2),
    (-0.162, 0.176, 0.5, 0.9, -0.2, 0.7, 1.2),
    (-0.149, 0.048, 0.5, 0.5, -0.5, 1.0, 1.2),
    # Extra hard-direction primitives from direct MuJoCo gait search.
    (-0.116, -0.100, 0.746, 0.620, 0.064, 0.738, 1.000),
    (-0.153, -0.186, 0.923, 0.918, 0.163, 0.745, 1.496),
    (-0.065, -0.227, 0.838, 1.098, 0.296, 0.792, 1.675),
    (0.100, -0.234, 0.783, 0.329, -0.243, 0.534, -0.912),
    # Extended form adds knee_phase, side_gain, and hip_steer_gain.
    (-0.227, -0.213, 0.995, 0.890, 0.296, 0.678, 1.163, 0.894, 0.318, 0.087),
    (-0.205, -0.355, 0.711, 0.871, 0.237, 0.798, 1.007, 0.463, 0.301, 0.085),
]


def ensure_model(model_path: Path = DEFAULT_MODEL) -> Path:
    if model_path.exists():
        return model_path
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from sim.tools.build_simple_quad_mjcf import build_mjcf

    return build_mjcf(output_path=model_path)


def import_optional_deps():
    missing: list[str] = []
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        gym = None
        spaces = None
        missing.append("gymnasium")
    try:
        import mujoco
    except ImportError:
        mujoco = None
        missing.append("mujoco")
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            f"Missing simulation dependencies: {joined}. "
            "Install them with `python -m pip install -r sim/requirements.txt`."
        )
    return gym, spaces, mujoco, np


try:
    _gym, _spaces, _mujoco, _np = import_optional_deps()
    _BaseEnv = _gym.Env
except ImportError:
    _gym = _spaces = _mujoco = _np = None
    _BaseEnv = object


def ensure_optional_deps_loaded() -> None:
    global _gym, _spaces, _mujoco, _np
    if _gym is None or _spaces is None or _mujoco is None or _np is None:
        _gym, _spaces, _mujoco, _np = import_optional_deps()


class SimpleQuadStandEnv(_BaseEnv):
    """MuJoCo standing task with normalized position-control actions."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_RATE_HZ}

    def __init__(
        self,
        model_path: str | Path | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        episode_seconds: float = EPISODE_SECONDS,
        randomize_actuators: bool = True,
        target_base_height: float = STABLE_BASE_HEIGHT,
        action_scale_rad: float = ACTION_SCALE_RAD,
        servo_velocity_limit_rad_s: float = SERVO_VELOCITY_LIMIT_RAD_S,
        terrain: str = "flat",
        terrain_seed: int | None = None,
        terrain_curriculum: str | tuple[str, ...] | bool | None = None,
        deterministic: bool = False,
    ) -> None:
        ensure_optional_deps_loaded()
        super().__init__()

        from sim.terrain import (
            apply_terrain_to_model,
            curriculum_terrain_name,
            get_terrain_config,
            normalize_terrain_name,
            terrain_xml,
        )

        self.model_path = ensure_model(Path(model_path).resolve() if model_path else DEFAULT_MODEL)
        self.requested_terrain = normalize_terrain_name(terrain)
        self.terrain_curriculum = terrain_curriculum
        self.deterministic_terrain = bool(deterministic)
        self.base_terrain_seed = int(terrain_seed if terrain_seed is not None else seed if seed is not None else 0)
        self.terrain_episode_index = 0
        self._apply_terrain_to_model = apply_terrain_to_model
        self._curriculum_terrain_name = curriculum_terrain_name
        self._get_terrain_config = get_terrain_config
        initial_terrain = self._select_terrain_name()
        self.terrain_config = self._get_terrain_config(initial_terrain)
        self._terrain_model_config = self._select_terrain_model_config(initial_terrain)
        self.terrain = self.terrain_config.name
        self.terrain_name = self.terrain_config.name
        self.terrain_level = self.terrain_config.name
        self.terrain_info: dict[str, Any] = {
            "terrain": self.terrain_config.name,
            "terrain_seed": float(self.base_terrain_seed),
            "floor_friction": float(self.terrain_config.nominal_friction),
        }

        if self._terrain_model_config.uses_hfield:
            model_xml = terrain_xml(self.model_path.read_text(encoding="utf-8"), self._terrain_model_config)
            self.model = _mujoco.MjModel.from_xml_string(model_xml)
        else:
            self.model = _mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = _mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.randomize_actuators = randomize_actuators
        self.target_base_height = target_base_height
        self.action_scale_rad = action_scale_rad
        self.servo_velocity_limit_rad_s = float(servo_velocity_limit_rad_s)
        self.control_dt = 1.0 / CONTROL_RATE_HZ
        self.frame_skip = max(1, int(round(self.control_dt / float(self.model.opt.timestep))))
        self.max_steps = int(round(episode_seconds * CONTROL_RATE_HZ))
        self.elapsed_steps = 0
        self.rng = _np.random.default_rng(seed)
        self.last_action = _np.zeros(len(ACTUATED_JOINTS), dtype=_np.float32)
        self.wrap_joint_observations = False
        self.renderer = None

        self.joint_ids = [_mujoco.mj_name2id(self.model, _mujoco.mjtObj.mjOBJ_JOINT, name) for name in ACTUATED_JOINTS]
        self.body_id = _mujoco.mj_name2id(self.model, _mujoco.mjtObj.mjOBJ_BODY, "body")
        if any(joint_id < 0 for joint_id in self.joint_ids) or self.body_id < 0:
            raise RuntimeError("Generated MJCF is missing expected simple_quad_v0 joints or body.")

        self.joint_qpos_addr = _np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.joint_dof_addr = _np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0].astype(_np.float32)
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1].astype(_np.float32)
        self.neutral_ctrl = _np.asarray([STAND_POSE[name] for name in ACTUATED_JOINTS], dtype=_np.float32)
        self.current_ctrl = self.neutral_ctrl.copy()
        self.episode_servo_velocity_limit = _np.full(
            self.model.nu,
            self.servo_velocity_limit_rad_s,
            dtype=_np.float32,
        )
        self.base_forcerange = self.model.actuator_forcerange.copy()
        self.base_joint_frictionloss = self.model.dof_frictionloss[self.joint_dof_addr].copy()

        self.action_space = _spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=_np.float32)
        obs_size = self.model.nq + self.model.nv + self.model.nu
        self.observation_space = _spaces.Box(low=-_np.inf, high=_np.inf, shape=(obs_size,), dtype=_np.float32)

    def _terrain_curriculum_active(self) -> bool:
        if self.requested_terrain == "curriculum":
            return True
        value = self.terrain_curriculum
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (tuple, list)):
            return len(value) > 0
        return str(value).strip().lower() not in {"", "none", "off", "false", "0"}

    def _select_terrain_name(self) -> str:
        if self.deterministic_terrain and isinstance(self.terrain_curriculum, (tuple, list)):
            levels = [str(value) for value in self.terrain_curriculum if str(value).strip()]
            if levels:
                return self._get_terrain_config(levels[self.terrain_episode_index % len(levels)]).name
        return self._curriculum_terrain_name(
            self.requested_terrain,
            self.terrain_curriculum,
            self.terrain_episode_index,
        )

    def _select_terrain_model_config(self, initial_terrain: str):
        if not self._terrain_curriculum_active():
            return self._get_terrain_config(initial_terrain)
        names: list[str] = []
        if isinstance(self.terrain_curriculum, (tuple, list)):
            names.extend(str(value) for value in self.terrain_curriculum)
        elif self.requested_terrain == "curriculum" or isinstance(self.terrain_curriculum, bool):
            names.extend(["flat", "mild", "rough"])
        else:
            names.extend([self.requested_terrain, initial_terrain])
        configs = [self._get_terrain_config(name) for name in names if str(name).strip()]
        configs.append(self._get_terrain_config(initial_terrain))
        return max(configs, key=lambda config: config.amplitude_m)

    def _terrain_seed_for_episode(self) -> int:
        if self.deterministic_terrain:
            return self.base_terrain_seed
        return self.base_terrain_seed + 9973 * self.terrain_episode_index

    def _set_episode_terrain(self) -> None:
        terrain_name = self._select_terrain_name()
        self.terrain_config = self._get_terrain_config(terrain_name)
        terrain_seed = self._terrain_seed_for_episode()
        self.terrain_info = self._apply_terrain_to_model(
            self.model,
            _mujoco,
            self.terrain_config,
            terrain_seed,
            randomize_friction=(self.randomize_actuators and not self.deterministic_terrain),
            hfield_config=self._terrain_model_config,
        )
        self.terrain = self.terrain_config.name
        self.terrain_name = self.terrain_config.name
        self.terrain_level = self.terrain_config.name

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        if self.wrap_joint_observations:
            joint_angles = qpos[self.joint_qpos_addr]
            qpos[self.joint_qpos_addr] = _np.arctan2(_np.sin(joint_angles), _np.cos(joint_angles))
        return _np.concatenate(
            [
                qpos,
                self.data.qvel.copy(),
                self.last_action.astype(_np.float64),
            ]
        ).astype(_np.float32)

    def _set_stand_state(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, self.target_base_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name, qpos_addr in zip(ACTUATED_JOINTS, self.joint_qpos_addr, strict=True):
            self.data.qpos[qpos_addr] = STAND_POSE[name]
        self.current_ctrl[:] = self.neutral_ctrl
        self.data.ctrl[:] = self.current_ctrl
        _mujoco.mj_forward(self.model, self.data)

    def _randomize_episode(self) -> None:
        if not self.randomize_actuators:
            self.model.actuator_forcerange[:] = self.base_forcerange
            self.model.dof_frictionloss[self.joint_dof_addr] = self.base_joint_frictionloss
            self.episode_servo_velocity_limit[:] = self.servo_velocity_limit_rad_s
            return
        warmup_episodes = float(getattr(self, "domain_randomization_warmup_episodes", 0.0))
        curriculum_scale = 1.0
        if warmup_episodes > 0.0:
            curriculum_scale = max(0.0, min(1.0, self.terrain_episode_index / warmup_episodes))
        torque_min = 1.0 - 0.30 * curriculum_scale
        velocity_min = 1.0 - 0.25 * curriculum_scale
        friction_min = 1.0 - 0.15 * curriculum_scale
        friction_max = 1.0 + 0.20 * curriculum_scale
        torque_scale = self.rng.uniform(torque_min, 1.0, size=self.model.nu)
        self.model.actuator_forcerange[:, 0] = self.base_forcerange[:, 0] * torque_scale
        self.model.actuator_forcerange[:, 1] = self.base_forcerange[:, 1] * torque_scale
        velocity_scale = self.rng.uniform(velocity_min, 1.0, size=self.model.nu)
        self.episode_servo_velocity_limit[:] = self.servo_velocity_limit_rad_s * velocity_scale
        friction_scale = self.rng.uniform(friction_min, friction_max, size=len(self.joint_dof_addr))
        self.model.dof_frictionloss[self.joint_dof_addr] = self.base_joint_frictionloss * friction_scale

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = _np.random.default_rng(seed)
        self.elapsed_steps = 0
        self.last_action[:] = 0.0
        self._set_episode_terrain()
        self._randomize_episode()
        self._set_stand_state()
        info = {"model_path": str(self.model_path), **self.terrain_info}
        self.terrain_episode_index += 1
        return self._get_obs(), info

    def _action_to_ctrl(self, action):
        action = _np.asarray(action, dtype=_np.float32)
        action = _np.clip(action, -1.0, 1.0)
        target = self.neutral_ctrl + action * self.action_scale_rad
        return _np.clip(target, self.ctrl_min, self.ctrl_max)

    def _apply_servo_velocity_limit(self, target_ctrl):
        if self.servo_velocity_limit_rad_s <= 0.0:
            self.current_ctrl[:] = target_ctrl
            return self.current_ctrl.copy()
        max_delta = self.episode_servo_velocity_limit * self.control_dt
        delta = _np.clip(target_ctrl - self.current_ctrl, -max_delta, max_delta)
        self.current_ctrl[:] = _np.clip(self.current_ctrl + delta, self.ctrl_min, self.ctrl_max)
        return self.current_ctrl.copy()

    def step(self, action):
        target_ctrl = self._action_to_ctrl(action)
        previous_ctrl = self.current_ctrl.copy()
        ctrl = self._apply_servo_velocity_limit(target_ctrl)
        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            _mujoco.mj_step(self.model, self.data)

        joint_vel = self.data.qvel[self.joint_dof_addr]
        base_height = float(self.data.qpos[2])
        body_xmat = self.data.xmat[self.body_id].reshape(3, 3)
        upright = float(body_xmat[2, 2])
        height_error = base_height - self.target_base_height
        action_delta = ctrl - previous_ctrl

        reward = (
            1.0
            + 1.5 * upright
            - 20.0 * height_error * height_error
            - 0.02 * float(_np.dot(joint_vel, joint_vel))
            - 0.05 * float(_np.dot(action_delta, action_delta))
        )

        self.last_action[:] = _np.clip(_np.asarray(action, dtype=_np.float32), -1.0, 1.0)
        self.elapsed_steps += 1

        terminated = bool(base_height < 0.045 or upright < math.cos(0.9))
        truncated = bool(self.elapsed_steps >= self.max_steps)
        info = {
            "base_height": base_height,
            "upright": upright,
            "sim_time": float(self.data.time),
            **self.terrain_info,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = _mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="track")
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class SimpleQuadWalkEnv(SimpleQuadStandEnv):
    """Forward walking task for the prototype quadruped."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        episode_seconds: float = 12.0,
        randomize_actuators: bool = True,
        target_base_height: float = STABLE_BASE_HEIGHT,
        target_velocity: float = 0.12,
        action_scale_rad: float = 0.8,
        gait_frequency_hz: float = 1.2,
        terrain: str = "flat",
        terrain_seed: int | None = None,
        terrain_curriculum: str | tuple[str, ...] | bool | None = None,
        deterministic: bool = False,
    ) -> None:
        self.target_velocity = target_velocity
        self.start_x = 0.0
        self.gait_frequency_hz = gait_frequency_hz
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            seed=seed,
            episode_seconds=episode_seconds,
            randomize_actuators=randomize_actuators,
            target_base_height=target_base_height,
            action_scale_rad=action_scale_rad,
            terrain=terrain,
            terrain_seed=terrain_seed,
            terrain_curriculum=terrain_curriculum,
            deterministic=deterministic,
        )
        self.observation_space = _spaces.Box(
            low=-_np.inf,
            high=_np.inf,
            shape=(self.model.nq + self.model.nv + self.model.nu + 2,),
            dtype=_np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.start_x = float(self.data.qpos[0])
        info["target_velocity"] = self.target_velocity
        return obs, info

    def _phase(self) -> float:
        return 2.0 * math.pi * self.gait_frequency_hz * self.elapsed_steps * self.control_dt

    def reference_action(self) -> Any:
        phase = self._phase()
        action = []
        for name in ACTUATED_JOINTS:
            leg_phase = 0.0
            if "front_right" in name or "back_left" in name:
                leg_phase = math.pi
            if "hip" in name:
                value = 0.5 * math.sin(phase + leg_phase)
            else:
                value = -0.4 + 0.8 * math.sin(phase + leg_phase + 0.7)
            action.append(max(-1.0, min(1.0, value)))
        return _np.asarray(action, dtype=_np.float32)

    def _get_obs(self):
        obs = super()._get_obs()
        phase = self._phase()
        return _np.concatenate(
            [obs, _np.asarray([math.sin(phase), math.cos(phase)], dtype=_np.float32)]
        ).astype(_np.float32)

    def step(self, action):
        previous_x = float(self.data.qpos[0])
        reference = self.reference_action()
        obs, stand_reward, terminated, truncated, info = super().step(action)
        current_x = float(self.data.qpos[0])
        forward_velocity = (current_x - previous_x) / self.control_dt
        velocity_error = forward_velocity - self.target_velocity
        joint_vel = self.data.qvel[self.joint_dof_addr]
        action_arr = _np.asarray(action, dtype=_np.float32)
        reference_error = action_arr - reference

        reward = (
            0.8
            + 2.0 * _np.exp(-18.0 * velocity_error * velocity_error)
            + 0.25 * max(0.0, min(forward_velocity, 0.35))
            + 0.35 * max(0.0, info["upright"])
            + 0.35 * float(_np.exp(-1.5 * _np.dot(reference_error, reference_error) / len(ACTUATED_JOINTS)))
            - 0.015 * float(_np.dot(joint_vel, joint_vel))
            - 0.01 * float(_np.dot(action_arr, action_arr))
        )
        if terminated:
            reward -= 1.0

        info["forward_velocity"] = float(forward_velocity)
        info["forward_distance"] = float(current_x - self.start_x)
        info["stand_reward"] = float(stand_reward)
        info["target_velocity"] = self.target_velocity
        info["reference_error"] = float(_np.sqrt(_np.mean(_np.square(reference_error))))
        return obs, float(reward), terminated, truncated, info


class SimpleQuadTargetEnv(SimpleQuadWalkEnv):
    """Goal-conditioned task: reach a random or keyboard-moved target quickly."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        episode_seconds: float = 14.0,
        randomize_actuators: bool = True,
        target_base_height: float = STABLE_BASE_HEIGHT,
        action_scale_rad: float = 0.9,
        gait_frequency_hz: float = 1.2,
        target_radius_min: float = 0.34,
        target_radius_max: float = 0.46,
        success_radius: float = 0.25,
        target_velocity: float = 0.16,
        terrain: str = "flat",
        terrain_seed: int | None = None,
        terrain_curriculum: str | tuple[str, ...] | bool | None = None,
        deterministic: bool = False,
        randomize_start_xy_m: float = 0.30,
        recovery_start_probability: float = 0.15,
        flipped_start_probability: float = 0.10,
        continuous_joints: bool = True,
    ) -> None:
        ensure_optional_deps_loaded()
        self.target_xy = _np.asarray([0.6, 0.0], dtype=_np.float64)
        self.initial_target_distance = 0.0
        self.previous_target_distance = 0.0
        self.closest_target_distance = float("inf")
        self.target_radius_min = target_radius_min
        self.target_radius_max = target_radius_max
        self.success_radius = success_radius
        self.randomize_start_xy_m = max(0.0, float(randomize_start_xy_m))
        self.recovery_start_probability = max(0.0, min(1.0, float(recovery_start_probability)))
        self.flipped_start_probability = max(0.0, min(1.0, float(flipped_start_probability)))
        self.continuous_joints = bool(continuous_joints)
        self.domain_randomization_warmup_episodes = 300
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            seed=seed,
            episode_seconds=episode_seconds,
            randomize_actuators=randomize_actuators,
            target_base_height=target_base_height,
            target_velocity=target_velocity,
            action_scale_rad=action_scale_rad,
            gait_frequency_hz=gait_frequency_hz,
            terrain=terrain,
            terrain_seed=terrain_seed,
            terrain_curriculum=terrain_curriculum,
            deterministic=deterministic,
        )
        if self.continuous_joints:
            # The source URDF declares every leg joint continuous.  Remove the
            # generated MJCF hard stops and expose a complete 2*pi target span.
            self.model.jnt_limited[self.joint_ids] = 0
            self.model.jnt_range[self.joint_ids, 0] = -math.pi
            self.model.jnt_range[self.joint_ids, 1] = math.pi
            self.model.actuator_ctrllimited[:] = 1
            self.model.actuator_ctrlrange[:, 0] = -math.pi
            self.model.actuator_ctrlrange[:, 1] = math.pi
            self.ctrl_min = self.model.actuator_ctrlrange[:, 0].astype(_np.float32)
            self.ctrl_max = self.model.actuator_ctrlrange[:, 1].astype(_np.float32)
            self.wrap_joint_observations = True
        self.target_body_id = _mujoco.mj_name2id(self.model, _mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self.target_mocap_id = -1
        if self.target_body_id >= 0:
            self.target_mocap_id = int(self.model.body_mocapid[self.target_body_id])
        self.observation_space = _spaces.Box(
            low=-_np.inf,
            high=_np.inf,
            shape=(self.model.nq + self.model.nv + self.model.nu + 2 + 11,),
            dtype=_np.float32,
        )

    def _base_yaw(self) -> float:
        body_xmat = self.data.xmat[self.body_id].reshape(3, 3)
        return math.atan2(float(body_xmat[1, 0]), float(body_xmat[0, 0]))

    def _action_to_ctrl(self, action):
        action_arr = _np.clip(_np.asarray(action, dtype=_np.float32), -1.0, 1.0)
        body_xmat = self.data.xmat[self.body_id].reshape(3, 3)
        upright = float(body_xmat[2, 2])
        recovery_range = bool(self.continuous_joints and upright < 0.35)
        scale = math.pi if recovery_range else float(self.action_scale_rad)
        target = self.neutral_ctrl + action_arr * scale
        return _np.clip(target, self.ctrl_min, self.ctrl_max)

    def _target_delta_world(self) -> Any:
        return self.target_xy - self.data.qpos[0:2]

    def _target_local(self) -> tuple[float, float, float, float]:
        dx, dy = [float(v) for v in self._target_delta_world()]
        yaw = self._base_yaw()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        distance = math.sqrt(dx * dx + dy * dy)
        heading_error = math.atan2(local_y, local_x)
        return local_x, local_y, distance, heading_error

    def set_target(self, x: float, y: float) -> None:
        self.target_xy[:] = [float(x), float(y)]
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = [float(x), float(y), 0.035]
            self.data.mocap_quat[self.target_mocap_id] = [1.0, 0.0, 0.0, 0.0]

    def sample_target(self) -> tuple[float, float]:
        min_radius = max(float(self.target_radius_min), float(self.success_radius) * 1.35)
        max_radius = max(float(self.target_radius_max), min_radius + 0.05)
        radius = math.sqrt(float(self.rng.uniform(min_radius * min_radius, max_radius * max_radius)))
        angle = float(self.rng.uniform(-math.pi, math.pi))
        return radius * math.cos(angle), radius * math.sin(angle)

    @staticmethod
    def _quat_from_euler(roll: float, pitch: float, yaw: float):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return _np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=_np.float64,
        )

    def _randomize_target_start(self, options: dict[str, Any]) -> str:
        radius = self.randomize_start_xy_m * math.sqrt(float(self.rng.uniform(0.0, 1.0)))
        angle = float(self.rng.uniform(-math.pi, math.pi))
        if "start_xy" in options:
            start_x, start_y = [float(value) for value in options["start_xy"]]
        else:
            start_x, start_y = radius * math.cos(angle), radius * math.sin(angle)

        # Begin with random XY/yaw but mostly upright locomotion, then blend in
        # side and flipped recovery over the first 160 episodes per worker.
        recovery_factor = 1.0 if self.deterministic_terrain else 0.5
        flip_probability = self.flipped_start_probability * recovery_factor
        recovery_probability = self.recovery_start_probability * recovery_factor
        pose_draw = float(self.rng.uniform(0.0, 1.0))
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        if pose_draw < flip_probability:
            pose_name = "flipped"
            roll = math.copysign(float(self.rng.uniform(2.65, math.pi)), float(self.rng.uniform(-1.0, 1.0)))
            pitch = float(self.rng.uniform(-0.35, 0.35))
            base_z = 0.075
            joint_angles = self.rng.uniform(-math.pi, math.pi, size=len(self.joint_qpos_addr))
        elif pose_draw < flip_probability + recovery_probability:
            pose_name = "tipped"
            if float(self.rng.uniform()) < 0.5:
                roll = math.copysign(float(self.rng.uniform(0.95, 1.65)), float(self.rng.uniform(-1.0, 1.0)))
                pitch = float(self.rng.uniform(-0.30, 0.30))
            else:
                roll = float(self.rng.uniform(-0.30, 0.30))
                pitch = math.copysign(float(self.rng.uniform(0.95, 1.55)), float(self.rng.uniform(-1.0, 1.0)))
            base_z = 0.085
            joint_angles = self.rng.uniform(-math.pi, math.pi, size=len(self.joint_qpos_addr))
        else:
            pose_name = "upright"
            roll = float(self.rng.normal(0.0, 0.10))
            pitch = float(self.rng.normal(0.0, 0.10))
            base_z = float(self.target_base_height + self.terrain_config.amplitude_m)
            joint_angles = self.rng.uniform(-0.22, 0.22, size=len(self.joint_qpos_addr))

        self.data.qpos[0:3] = [start_x, start_y, base_z]
        self.data.qpos[3:7] = self._quat_from_euler(roll, pitch, yaw)
        self.data.qpos[self.joint_qpos_addr] = joint_angles
        self.data.qvel[:] = 0.0
        self.data.qvel[0:6] = self.rng.uniform(-0.08, 0.08, size=6)
        self.current_ctrl[:] = _np.clip(joint_angles, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[:] = self.current_ctrl
        _mujoco.mj_forward(self.model, self.data)
        return pose_name

    def _select_target_primitive(self) -> tuple[float, float, float, float, float, float, float]:
        # Gait selection must be body-relative so random starting yaw does not
        # rotate the desired direction out from under the policy.
        dx, dy, _distance, _heading_error = self._target_local()
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1e-6:
            return TARGET_GAIT_PRIMITIVES[6]

        best = TARGET_GAIT_PRIMITIVES[0]
        best_score = float("inf")
        target_angle = math.atan2(dy, dx)
        for primitive in TARGET_GAIT_PRIMITIVES:
            px, py = primitive[0], primitive[1]
            primitive_distance = math.sqrt(px * px + py * py)
            primitive_angle = math.atan2(py, px)
            angle_error = math.atan2(math.sin(target_angle - primitive_angle), math.cos(target_angle - primitive_angle))
            distance_error = abs(distance - primitive_distance)
            too_small = max(0.0, min(0.16, distance * 0.45) - primitive_distance)
            score = abs(angle_error) + 0.75 * distance_error + 4.0 * too_small
            if score < best_score:
                best = primitive
                best_score = score
        return best

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        options = options or {}
        reset_pose = self._randomize_target_start(options)
        if "target_xy" in options:
            target_x, target_y = options["target_xy"]
        else:
            if reset_pose != "upright":
                recovery_progress = max(0.0, min(1.0, self.terrain_episode_index / 160.0))
                min_radius = max(float(self.success_radius) * 1.05, 0.23)
                max_radius = 0.32 + 1.20 * recovery_progress
                radius = math.sqrt(float(self.rng.uniform(min_radius**2, max_radius**2)))
                angle = float(self.rng.uniform(-math.pi, math.pi))
                target_dx, target_dy = radius * math.cos(angle), radius * math.sin(angle)
            else:
                target_dx, target_dy = self.sample_target()
            target_x = float(self.data.qpos[0]) + target_dx
            target_y = float(self.data.qpos[1]) + target_dy
        self.set_target(target_x, target_y)
        self.previous_target_distance = self._target_local()[2]
        self.initial_target_distance = self.previous_target_distance
        self.closest_target_distance = self.previous_target_distance
        _mujoco.mj_forward(self.model, self.data)
        info["target_x"] = float(self.target_xy[0])
        info["target_y"] = float(self.target_xy[1])
        info["target_distance"] = self.previous_target_distance
        info["initial_target_distance"] = self.initial_target_distance
        info["reset_pose"] = reset_pose
        info["start_x"] = float(self.data.qpos[0])
        info["start_y"] = float(self.data.qpos[1])
        info["continuous_joints"] = bool(self.continuous_joints)
        return self._get_obs(), info

    def _get_obs(self):
        obs = super()._get_obs()
        local_x, local_y, distance, heading_error = self._target_local()
        _px, _py, amp_h, amp_k, bias_k, freq, steer = self._select_target_primitive()[:7]
        selected_phase = 2.0 * math.pi * freq * self.elapsed_steps * self.control_dt
        target_obs = _np.asarray(
            [
                max(-1.5, min(1.5, local_x)),
                max(-1.5, min(1.5, local_y)),
                max(0.0, min(1.5, distance)),
                max(-math.pi, min(math.pi, heading_error)) / math.pi,
                amp_h,
                amp_k,
                bias_k,
                freq / 1.4,
                steer / 1.5,
                math.sin(selected_phase),
                math.cos(selected_phase),
            ],
            dtype=_np.float32,
        )
        return _np.concatenate([obs, target_obs]).astype(_np.float32)

    def reference_action(self) -> Any:
        primitive = self._select_target_primitive()
        _px, _py, amp_h, amp_k, bias_k, freq, steer = primitive[:7]
        knee_phase = primitive[7] if len(primitive) > 7 else 0.7
        side_gain = primitive[8] if len(primitive) > 8 else 0.25
        hip_steer_gain = primitive[9] if len(primitive) > 9 else 0.1
        side_scale_max = 1.8 if len(primitive) > 8 else 1.6
        phase = 2.0 * math.pi * freq * self.elapsed_steps * self.control_dt
        _local_x, _local_y, distance, _heading_error = self._target_local()
        forward_scale = 0.0 if distance < self.success_radius else 1.0

        action = []
        for name in ACTUATED_JOINTS:
            diagonal_phase = 0.0
            if "front_right" in name or "back_left" in name:
                diagonal_phase = math.pi
            side = -1.0 if "left" in name else 1.0
            side_scale = max(0.2, min(side_scale_max, 1.0 + side_gain * steer * side))
            if "hip" in name:
                value = forward_scale * (
                    side_scale * amp_h * math.sin(phase + diagonal_phase) + hip_steer_gain * steer * side
                )
            else:
                value = forward_scale * (
                    bias_k + side_scale * amp_k * math.sin(phase + diagonal_phase + knee_phase)
                )
            action.append(max(-1.0, min(1.0, value)))
        return _np.asarray(action, dtype=_np.float32)

    def step(self, action):
        previous_distance = self._target_local()[2]
        previous_xy = self.data.qpos[0:2].copy()
        reference = self.reference_action()
        obs, stand_reward, terminated, truncated, info = SimpleQuadStandEnv.step(self, action)
        local_x, local_y, distance, heading_error = self._target_local()
        progress = previous_distance - distance
        self.closest_target_distance = min(self.closest_target_distance, distance)
        planar_step = self.data.qpos[0:2] - previous_xy
        planar_speed = float(_np.linalg.norm(planar_step) / self.control_dt)
        directed_velocity = float(progress / self.control_dt)
        lateral_speed = math.sqrt(max(0.0, planar_speed * planar_speed - directed_velocity * directed_velocity))
        target_distance_reduction = self.initial_target_distance - distance
        action_arr = _np.asarray(action, dtype=_np.float32)
        reference_error = action_arr - reference

        # Target success is deliberately planar-only. Height and uprightness
        # remain diagnostics, never reward inputs or success gates.
        success = bool(distance <= self.success_radius)
        velocity_error = directed_velocity - float(self.target_velocity)
        normalized_reduction = target_distance_reduction / max(self.initial_target_distance, 1e-6)
        previous_near_target = math.exp(-5.0 * previous_distance)
        near_target = math.exp(-5.0 * distance)
        near_target_progress = near_target - previous_near_target

        # A potential-difference reward cannot be collected by standing still
        # or circling. Every dense term is derived only from XY target distance.
        reward = 60.0 * progress + 12.0 * near_target_progress
        if progress < 0.0:
            reward += 30.0 * progress
        if success:
            reward += 120.0 + max(0.0, self.max_steps - self.elapsed_steps) * 0.10
            terminated = True
        else:
            # Permit self-righting. Only invalid/escaped simulations terminate;
            # low height and upside-down posture do not.
            finite_state = bool(_np.all(_np.isfinite(self.data.qpos)) and _np.all(_np.isfinite(self.data.qvel)))
            escaped = bool(abs(float(self.data.qpos[0])) > 4.0 or abs(float(self.data.qpos[1])) > 4.0)
            terminated = bool(not finite_state or escaped or float(self.data.qpos[2]) < -0.25)

        info["target_x"] = float(self.target_xy[0])
        info["target_y"] = float(self.target_xy[1])
        info["target_distance"] = float(distance)
        info["initial_target_distance"] = float(self.initial_target_distance)
        info["target_distance_reduction"] = float(target_distance_reduction)
        info["target_normalized_reduction"] = float(normalized_reduction)
        info["closest_target_distance"] = float(self.closest_target_distance)
        info["target_local_x"] = float(local_x)
        info["target_local_y"] = float(local_y)
        info["target_heading_error"] = float(heading_error)
        info["target_progress"] = float(progress)
        info["target_directed_velocity"] = directed_velocity
        info["target_velocity"] = float(self.target_velocity)
        info["target_velocity_error"] = float(velocity_error)
        info["planar_speed"] = planar_speed
        info["lateral_speed"] = float(lateral_speed)
        info["stable_for_success"] = bool(success)
        info["success"] = bool(success)
        info["stand_reward"] = float(stand_reward)
        info["reference_error"] = float(_np.sqrt(_np.mean(_np.square(reference_error))))
        info["target_reward_planar_only"] = True
        info["target_reward_near_target"] = float(near_target)
        info["target_reward_planar_progress"] = float(60.0 * progress)
        info["target_reward_planar_potential_progress"] = float(12.0 * near_target_progress)
        info["target_reward_crash_penalty"] = 0.0
        self.previous_target_distance = distance
        return self._get_obs(), float(reward), terminated, truncated, info
