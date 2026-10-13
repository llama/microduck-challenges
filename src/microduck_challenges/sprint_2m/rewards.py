"""Race-mode reward terms for the 2 m Sprint.

The Arena clocks the trunk from the start line to 2 m and keeps the best of
five seeds; the leaders cross in under 1 s at 2.5 m/s, nothing like a walk.
`race_speed` pays the trunk's speed along the spawn heading, linearly, so
faster always pays and no commanded speed caps it.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def race_speed(
    env: ManagerBasedRlEnv,
    clip: float = 4.0,
    power: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """The trunk's speed (m/s) along the heading it spawned with, clipped to ±clip.

    `power` > 1 bends it: sign(v) * clip * (|v| / clip) ** power, equal to the speed
    at the clip and steeper near it, so a burst at the top pays more than a steady
    middle speed (A-H plateaued at a 1.8 m/s shuffle; the leaders reach 2.5).

    The spawn heading is captured on the first step after reset, like
    heading_hold_reward does, so a crab walk or a drift earns less than running
    straight down the lane the Arena measures along.
    """
    asset = env.scene[asset_cfg.name]
    yaw = _yaw(asset.data.root_link_quat_w)
    if not hasattr(env, "_race_heading_ref") or env._race_heading_ref.shape[0] != env.num_envs:
        env._race_heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._race_heading_ref = torch.where(just_reset, yaw, env._race_heading_ref)
    ref = env._race_heading_ref
    vel = asset.data.root_link_lin_vel_w
    forward = vel[:, 0] * torch.cos(ref) + vel[:, 1] * torch.sin(ref)
    forward = forward.clamp(-clip, clip)
    if power != 1.0:
        return torch.sign(forward) * clip * (forward.abs() / clip) ** power
    return forward


def trunk_low(
    env: ManagerBasedRlEnv,
    target: float = 0.125,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """How far the trunk sits below `target` (m), 0 when at or above it.

    A-C settled into a crouched shuffle (trunk at 7-12 cm, 1.8 m/s peak); the
    Arena's leaders run tall (11-14 cm) at 2.5 m/s. Priced as a penalty, this
    only objects to the crouch and says nothing above the target.
    """
    asset = env.scene[asset_cfg.name]
    return (target - asset.data.root_link_pos_w[:, 2]).clamp(min=0.0)


def _spawn_frame(env: ManagerBasedRlEnv, asset) -> tuple[torch.Tensor, torch.Tensor]:
    """Each env's spawn position and heading, captured on the first step after reset."""
    pos = asset.data.root_link_pos_w[:, :2]
    yaw = _yaw(asset.data.root_link_quat_w)
    if not hasattr(env, "_race_spawn_pos") or env._race_spawn_pos.shape[0] != env.num_envs:
        env._race_spawn_pos = pos.clone()
        env._race_spawn_yaw = yaw.clone()
    just_reset = (env.episode_length_buf <= 1).unsqueeze(-1)
    env._race_spawn_pos = torch.where(just_reset, pos, env._race_spawn_pos)
    env._race_spawn_yaw = torch.where(just_reset.squeeze(-1), yaw, env._race_spawn_yaw)
    return env._race_spawn_pos, env._race_spawn_yaw


def race_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Metres covered along the spawn heading since the spawn: the Arena's progress."""
    asset = env.scene[asset_cfg.name]
    spawn, yaw = _spawn_frame(env, asset)
    delta = asset.data.root_link_pos_w[:, :2] - spawn
    return delta[:, 0] * torch.cos(yaw) + delta[:, 1] * torch.sin(yaw)


def finished_race(env: ManagerBasedRlEnv, distance: float = 2.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Termination: the trunk has covered `distance` m. The race is over; nothing after
    the line is trained, exactly as the Arena never watches it (the leaders cross at
    54 deg of nose-down lean, a controlled fall timed to outlast the race)."""
    return race_progress(env, asset_cfg) >= distance


def finish_bonus(env: ManagerBasedRlEnv, distance: float = 2.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 on the step the line is crossed, so crossing beats a dive that stops short."""
    return finished_race(env, distance, asset_cfg).float()


def arena_fall(env: ManagerBasedRlEnv, hold_steps: int = 15, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """The Arena's own fall: the trunk tipped past 60 deg (gravity's body-frame z above -0.5)
    for 0.3 s running (15 control steps). An instant 60 deg cut-off kills the leaders' gait,
    which dips to 54 deg and past it for a few steps under action noise."""
    asset = env.scene[asset_cfg.name]
    tipped = asset.data.projected_gravity_b[:, 2] > -0.5
    if not hasattr(env, "_race_tipped") or env._race_tipped.shape[0] != env.num_envs:
        env._race_tipped = torch.zeros(env.num_envs, dtype=torch.long, device=tipped.device)
    env._race_tipped = torch.where(tipped, env._race_tipped + 1, torch.zeros_like(env._race_tipped))
    env._race_tipped = torch.where(env.episode_length_buf <= 1, torch.zeros_like(env._race_tipped), env._race_tipped)
    return env._race_tipped >= hold_steps
