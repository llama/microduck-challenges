"""Dizzy Duck: the spin command and the reward terms that teach a pirouette.

The Arena feeds the policy the 13 numbers a timeline keyframe carries. This
challenge's env reads `twist[0]` as the sit flag; Dizzy Duck adds a **spin
flag in `twist[2]`** (1 = spin counter-clockwise, 0 = don't). `twist[1]` and
`body` stay zero. The flag is sampled by `SpinSitStandCommand` (a seated duck
never spins), and while it is up the posture rewards that pay for resting
still are gated off (`unless_spinning`) so that spinning is the only income —
the header lesson: a hold annuity extinguishes discovery of the dynamic move.

Physics measured on the Arena's floor (2026-10-12, scratch harness): a
one-foot stance on the sole's toe is a single floor contact, but hull-vertex
wander still drags ~0.025 Nm of yaw torque, so a 6 rad/s spin coasts only
~60°. A full turn therefore needs push-offs: both feet down briefly, then a
one-foot coast. The rewards below pay yaw progress on ONE foot in full, on two
feet at a fraction (the push), airborne not at all, and nothing when tilted.

Sign convention (microduck_rl AGENTS.md): every term here returns ≥ 0; costs
take a NEGATIVE weight in env.py. Check `Episode_Reward/spin_drift` ≤ 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp as microduck_mdp

SPIN_SLOT = 2  # twist[2]
_ROBOT = SceneEntityCfg("robot")


# ── Command ──────────────────────────────────────────────────────────────────


class SpinSitStandCommand(microduck_mdp.SitStandCommand):
    """cmd = [sit_flag, 0, spin_flag], both flags ∈ {0, 1}, never both 1.

    Resampled together on the cfg's dwell time: each segment is one of stand
    still / sit / spin. The sit ramp (`alpha`) is inherited unchanged, so the
    posture rewards keep tracking the slewed sit target.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._spin_prob = float(getattr(cfg, "spin_prob", 0.5))

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)  # writes [sit, 0, 0]
        n = len(env_ids)
        if n == 0:
            return
        spin = (torch.rand(n, device=self.device) < self._spin_prob).float()
        sit = self.vel_command_b[env_ids, 0]
        self.vel_command_b[env_ids, SPIN_SLOT] = spin * (1.0 - sit)


@dataclass(kw_only=True)
class SpinSitStandCommandCfg(microduck_mdp.SitStandCommandCfg):
    class_type: type = SpinSitStandCommand
    # Probability a resample commands SPIN (applied after the sit draw; a sit wins).
    spin_prob: float = 0.5

    def build(self, env: ManagerBasedRlEnv) -> SpinSitStandCommand:
        return SpinSitStandCommand(self, env)


# ── Helpers ──────────────────────────────────────────────────────────────────


def spin_flag(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """The spin flag as the policy reads it, per env (0 or 1)."""
    return env.command_manager.get_command(command_name)[:, SPIN_SLOT]


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) of a [w, x, y, z] quaternion batch."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def feet_on_floor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """How many feet touch the floor, per env (0, 1 or 2)."""
    found = env.scene[sensor_name].data.found
    assert found is not None
    if found.dim() == 1:
        found = found.unsqueeze(-1)
    return (found > 0).float().sum(dim=-1)


def upright_factor(env: ManagerBasedRlEnv, min_cos: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 upright, 0 once the trunk tilts past acos(min_cos); linear between."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]  # -1 upright
    return torch.clamp((-g - min_cos) / max(1.0 - min_cos, 1e-6), 0.0, 1.0)


def _fresh(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf <= 1


# ── Rewards ──────────────────────────────────────────────────────────────────


def spin_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    max_step_rad: float = 0.3,
    double_support_scale: float = 0.15,
    min_upright_cos: float = 0.5,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Potential-based yaw progress while the spin flag is up.

    Pays Δyaw (counter-clockwise positive) each step — holding still pays
    zero, unwinding pays negative, so it cannot be farmed — scaled by the
    support state: ×1 on exactly one foot (the coast), ×double_support_scale
    on two (the push-off), ×0 airborne, and faded out as the trunk tilts
    (a toppling duck turns too). Capped per step so a solver glitch cannot
    pay a jackpot.
    """
    asset = env.scene[asset_cfg.name]
    yaw = yaw_of(asset.data.root_link_quat_w)
    if not hasattr(env, "_spin_prev_yaw") or env._spin_prev_yaw.shape[0] != env.num_envs:
        env._spin_prev_yaw = yaw.clone()
    delta = wrap_angle(yaw - env._spin_prev_yaw)
    env._spin_prev_yaw = yaw.clone()
    delta = torch.where(_fresh(env), torch.zeros_like(delta), delta)
    delta = torch.clamp(delta, -max_step_rad, max_step_rad)

    feet = feet_on_floor(env, sensor_name)
    support = torch.where(
        feet == 1,
        torch.ones_like(feet),
        torch.where(feet >= 2, torch.full_like(feet, double_support_scale), torch.zeros_like(feet)),
    )
    up = upright_factor(env, min_upright_cos, asset_cfg)
    return spin_flag(env, command_name) * delta * support * up


def spin_coast(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    min_rate: float = 1.5,
    full_rate: float = 6.0,
    min_upright_cos: float = 0.5,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Per-step pay for coasting on ONE foot at a real spin rate (bounded ≤ 1).

    Standing on one foot without turning pays nothing (rate below min_rate),
    turning on two feet pays nothing: only the pirouette itself is rewarded.
    """
    asset = env.scene[asset_cfg.name]
    rate = asset.data.root_link_ang_vel_w[:, 2]
    r = torch.clamp((rate - min_rate) / max(full_rate - min_rate, 1e-6), 0.0, 1.0)
    single = (feet_on_floor(env, sensor_name) == 1).float()
    up = upright_factor(env, min_upright_cos, asset_cfg)
    return spin_flag(env, command_name) * single * r * up


def spin_drift(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    radius: float = 0.03,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Distance (m) beyond `radius` from where the spin was commanded; ≥ 0, weight it negative.

    A pirouette stays on its spot. The anchor is captured when the flag rises
    (and at episode start), so a spin that walks off in a circle pays per step.
    """
    pos = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]
    flag = spin_flag(env, command_name)
    if not hasattr(env, "_spin_anchor") or env._spin_anchor.shape[0] != env.num_envs:
        env._spin_anchor = pos.clone()
        env._spin_prev_flag = torch.zeros_like(flag)
    rising = ((flag > 0.5) & (env._spin_prev_flag < 0.5)) | _fresh(env)
    env._spin_anchor = torch.where(rising.unsqueeze(-1), pos, env._spin_anchor)
    env._spin_prev_flag = flag.clone()
    dist = torch.linalg.norm(pos - env._spin_anchor, dim=-1)
    return flag * torch.clamp(dist - radius, min=0.0)


def unless_spinning(
    env: ManagerBasedRlEnv,
    inner: Callable[..., torch.Tensor],
    params: dict,
    command_name: str = "twist",
    floor: float = 0.0,
) -> torch.Tensor:
    """`inner(env, **params)` scaled to `floor` while the spin flag is up.

    Wraps the posture terms that pay for resting still so that, under a spin
    command, standing still earns nothing and the spin terms are the only
    income. `params` must not hold a SceneEntityCfg (the manager resolves
    only top-level ones); the posture terms wrapped in env.py hold none.
    """
    for v in params.values():  # nested SceneEntityCfgs are not resolved by the manager
        if isinstance(v, SceneEntityCfg) and getattr(v, "body_ids", None) in (None, slice(None)) and v.body_names:
            v.resolve(env.scene)
    value = inner(env, **params)
    gate = 1.0 - (1.0 - floor) * spin_flag(env, command_name)
    return value * gate


# ── Style (v2): a pirouette, not a whirl ─────────────────────────────────────
# Stage A v1 (run spin_a2, 2026-10-12) found a 15-20 rad/s one-foot whirl with
# the trunk tilted 30°+, the head whipping full range as a propeller and the
# lifted foot skimming the floor. These terms, all gated on the spin flag so a
# resting duck never pays them, price the style: upright trunk, quiet head, a
# visible foot lift and no reason to exceed about 1.3 turns per second.


def spin_tilt(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """1 − cos(trunk tilt) while spinning; ≥ 0, weight it negative (30° → 0.13)."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return spin_flag(env, command_name) * torch.clamp(1.0 + g, min=0.0)


def spin_head_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Σ head joint velocity² (neck_pitch, head_pitch, head_yaw, head_roll) while spinning; weight it negative."""
    asset = env.scene[asset_cfg.name]
    vel = microduck_mdp._servo_joint_vel(env, asset)[:, 5:9]
    return spin_flag(env, command_name) * torch.sum(vel * vel, dim=-1)


def spin_overspeed(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    max_rate: float = 9.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """(|yaw rate| − max_rate)₊² while spinning; weight it negative."""
    rate = env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2]
    over = torch.clamp(rate.abs() - max_rate, min=0.0)
    return spin_flag(env, command_name) * over * over


def spin_swing_lift(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    clearance_target: float = 0.03,
    min_upright_cos: float = 0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    """Smoothstep of the lifted foot's height above the stance foot, up to `clearance_target`,
    while spinning on exactly one foot and upright (≤ 1 per step: no jackpot for a violent kick)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    lift = torch.nan_to_num(z.max(dim=-1).values - z.min(dim=-1).values, nan=0.0)
    x = torch.clamp(lift / clearance_target, 0.0, 1.0)
    single = (feet_on_floor(env, sensor_name) == 1).float()
    up = upright_factor(env, min_upright_cos, asset_cfg)
    return spin_flag(env, command_name) * single * up * x * x * (3.0 - 2.0 * x)


def spin_head_pose(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Σ |head joint − (default + commanded delta)| over the 4 head joints while spinning (rad); weight it negative.

    v2 (spin_a3) kept the head slung out as the spin's counterweight: the Gaussian
    tracking term is nearly flat that far out. An L1 keeps a constant pull back.
    """
    asset = env.scene[asset_cfg.name]
    q = microduck_mdp._servo_joint_pos(env, asset)[:, 5:9]
    target = microduck_mdp._servo_default_joint_pos(env, asset)[:, 5:9] + env.command_manager.get_command(head_command_name)
    return spin_flag(env, command_name) * torch.sum(torch.abs(q - target), dim=-1)


# ── Wilder (Doug 2026-10-12): faster, longer on each foot, foot higher ───────


def _feet_prev(env: ManagerBasedRlEnv, feet: torch.Tensor) -> torch.Tensor:
    if not hasattr(env, "_spin_prev_feet") or env._spin_prev_feet.shape[0] != env.num_envs:
        env._spin_prev_feet = feet.clone()
    prev = env._spin_prev_feet
    env._spin_prev_feet = feet.clone()
    return torch.where(_fresh(env), feet, prev)


def spin_touchdowns(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """1 on the step the free foot comes back down (1 foot → 2) while spinning; weight it negative."""
    feet = feet_on_floor(env, sensor_name)
    prev = _feet_prev(env, feet)
    return spin_flag(env, command_name) * ((prev == 1) & (feet >= 2)).float()


def spin_streak(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    full_s: float = 2.0,
    min_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """min(unbroken one-foot time / full_s, 1) per step while spinning fast on one foot: long coasts pay more."""
    feet = feet_on_floor(env, sensor_name)
    single = (feet == 1).float()
    if not hasattr(env, "_spin_streak_s") or env._spin_streak_s.shape[0] != env.num_envs:
        env._spin_streak_s = torch.zeros_like(single)
    env._spin_streak_s = torch.where(_fresh(env), torch.zeros_like(single), (env._spin_streak_s + env.step_dt) * single)
    rate = env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2]
    fast = (rate.abs() >= min_rate).float()
    return spin_flag(env, command_name) * single * fast * torch.clamp(env._spin_streak_s / full_s, 0.0, 1.0)


def spin_head_drop(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    min_z: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("jaw_soft",)),
) -> torch.Tensor:
    """(min_z − head height)₊ in metres while spinning; weight it negative.

    Doug 2026-10-12 on B4@500: the from-scratch show policy planted its head on
    the floor and spun as a head-and-foot tripod. The skull body (`jaw_soft`)
    stands at 0.23 m; anything below `min_z` during the spin is priced so that a
    head on the floor (~0.05 m) costs more than the spin earns.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2] - env.scene.terrain.env_origins[:, 2]
    z = torch.nan_to_num(z, nan=min_z)
    return spin_flag(env, command_name) * torch.clamp(min_z - z, min=0.0)


# ── Curriculum: sit and stand first, then the spin ───────────────────────────


def spin_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "twist",
    stages: list | None = None,
) -> torch.Tensor:
    """Step-staged probability that a command resample starts a spin segment.

    From-scratch show runs that mixed spin segments in from iteration 0 (B4-B6,
    2026-10-12) never learned to stand or sit: the spin-phase costs dominated a
    policy that had no skill yet. Stages are `{"step": int, "prob": float}`;
    the latest elapsed stage applies (same convention as reward_weight). The
    stock sit/stand recipe trains alone until the first non-zero stage.
    """
    del env_ids
    term = env.command_manager.get_term(command_name)
    prob = float(getattr(term, "_spin_prob", 0.0))
    for stage in stages or []:
        if env.common_step_counter > stage["step"]:
            prob = float(stage["prob"])
    term._spin_prob = prob
    return torch.tensor([prob])


def spin_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    full_rate: float = 3.0,
    min_rate: float = 0.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """clamp((|yaw rate| − min_rate) / (full_rate − min_rate), 0, 1) while the spin flag is up.

    `min_rate` is a deadzone: B8 (min_rate 0) rewarded the yaw rate that action
    noise alone produces, and the policy's std climbed 0.36 → 1.2 in 500
    iterations; wobble below ~1 rad/s must earn nothing.

    B7 (2026-10-13) learned sit/stand, then stood still through every spin
    segment for 300 iterations: Δyaw pays nothing until the policy happens to
    rotate, and the always-on upright/height terms make standing still
    comfortable. This dense term starts the rotation; progress and coast take
    over once it turns.
    """
    rate = env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2]
    x = (rate.abs() - min_rate) / max(full_rate - min_rate, 1e-6)
    return spin_flag(env, command_name) * torch.clamp(x, 0.0, 1.0)
