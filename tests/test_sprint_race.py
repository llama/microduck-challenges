"""The 2 m Sprint's race mode: speed along the spawn heading pays, the walking regularizers do not."""

import math
from types import SimpleNamespace

import torch

from microduck_challenges.sprint_2m import env as sprint_env
from microduck_challenges.sprint_2m import rewards as sprint_rewards


def _quat(yaws):
    half = torch.tensor(yaws) / 2
    zero = torch.zeros_like(half)
    return torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)


def test_race_speed_is_the_trunk_speed_along_the_spawn_heading():
    robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=_quat([0.0, math.pi / 2, 0.0]),
                                                  root_link_lin_vel_w=torch.zeros(3, 3)))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=3, episode_length_buf=torch.tensor([1, 1, 1]))
    sprint_rewards.race_speed(env)  # captures the spawn headings
    env.episode_length_buf = torch.tensor([30, 30, 30])
    robot.data.root_link_quat_w = _quat([0.3, math.pi / 2, 0.0])  # the first drifted; speed is still judged on the spawn line
    robot.data.root_link_lin_vel_w = torch.tensor([[2.0, 0.5, 0.0], [0.0, 1.5, 0.3], [-1.0, 0.0, 0.0]])
    r = sprint_rewards.race_speed(env, clip=4.0)
    assert torch.allclose(r, torch.tensor([2.0, 1.5, -1.0]), atol=1e-6)
    robot.data.root_link_lin_vel_w = torch.tensor([[9.0, 0.0, 0.0]] * 3)
    assert sprint_rewards.race_speed(env, clip=4.0)[0].item() == 4.0


def test_race_mode_rewards_speed_and_drops_the_walking_regularizers():
    cfg = sprint_env.make_env_cfg()
    v = sprint_env.RACE_VARIANTS[sprint_env.RACE_VARIANT]
    assert cfg.rewards["race_speed"].func is sprint_rewards.race_speed and cfg.rewards["race_speed"].weight == v["speed"]
    for name in ("pose", "foot_clearance", "foot_swing_height", "foot_slip", "body_ang_vel", "angular_momentum"):
        assert cfg.rewards[name].weight == 0.0
    assert cfg.rewards["air_time"].weight == 0.0
    assert cfg.rewards["heading_hold"].weight == v["heading"]
    assert cfg.rewards["dof_pos_limits"].weight < 0  # joint limits are physical, not a style
    assert "standing_envs" not in cfg.curriculum and cfg.commands["twist"].rel_standing_envs == 0.0
    assert ("push_robot" in cfg.events) == v["pushes"]
    assert ("action_rate_weight" in cfg.curriculum) == v["action_rate_ramp"]
    assert cfg.episode_length_s == v["episode_s"]
    assert "fell_over" in cfg.terminations


def test_race_progress_and_finish_follow_the_spawn_frame():
    robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=_quat([0.0, math.pi / 2]),
                                                  root_link_pos_w=torch.tensor([[1.0, 1.0, 0.1], [5.0, 5.0, 0.1]])))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, episode_length_buf=torch.tensor([1, 1]))
    assert torch.allclose(sprint_rewards.race_progress(env), torch.zeros(2))
    env.episode_length_buf = torch.tensor([40, 40])
    robot.data.root_link_pos_w = torch.tensor([[3.1, 1.5, 0.1], [5.5, 6.9, 0.1]])  # +2.1 along x; +1.9 along y
    assert torch.allclose(sprint_rewards.race_progress(env), torch.tensor([2.1, 1.9]), atol=1e-6)
    assert sprint_rewards.finished_race(env, 2.0).tolist() == [True, False]
    assert sprint_rewards.finish_bonus(env, 2.0).tolist() == [1.0, 0.0]


def test_race_variant_g_ends_at_the_line_and_runs_a_clock():
    v = sprint_env.RACE_VARIANTS["G"]
    cfg = sprint_env.make_env_cfg()
    sprint_env._apply_race_mode(cfg, v, play=False)
    assert cfg.terminations["finished"].func is sprint_rewards.finished_race
    assert cfg.terminations["finished"].params["distance"] == 2.0
    assert cfg.rewards["clock"].weight < 0 and cfg.rewards["finish_bonus"].weight > 0
    assert cfg.episode_length_s == 3.0


def test_race_speed_power_bends_the_top_end():
    robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=_quat([0.0, 0.0, 0.0]), root_link_lin_vel_w=torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]])))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=3, episode_length_buf=torch.tensor([1, 1, 1]))
    r = sprint_rewards.race_speed(env, clip=4.0, power=2.0)
    assert torch.allclose(r, torch.tensor([0.25, 1.0, 4.0]), atol=1e-6)


def test_arena_fall_needs_the_tilt_held():
    robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0, 0, -0.2], [0, 0, -0.9]])))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, episode_length_buf=torch.tensor([5, 5]))
    for _ in range(14):
        assert not sprint_rewards.arena_fall(env).any()
    assert sprint_rewards.arena_fall(env).tolist() == [True, False]
    env.episode_length_buf = torch.tensor([0, 5])  # a reset clears the count
    assert not sprint_rewards.arena_fall(env).any()


def test_race_variant_x_narrows_randomization_and_judges_falls_like_the_arena():
    cfg = sprint_env.make_env_cfg()
    sprint_env._apply_race_mode(cfg, sprint_env.RACE_VARIANTS["X"], play=False)
    assert cfg.terminations["arena_fall"].func is sprint_rewards.arena_fall
    lo, hi = cfg.events["foot_friction"].params["ranges"]
    assert 0.96 < lo < 1.0 < hi < 1.04
    for term in cfg.events.values():
        for key in ("ranges", "scale_range", "bias_range"):
            r = term.params.get(key) if isinstance(term.params, dict) else None
            if isinstance(r, tuple) and len(r) == 2 and all(isinstance(x, float) for x in r):
                assert r[1] - r[0] < 0.1, (key, r)


def test_race_variant_y_feeds_the_actor_clean_observations():
    cfg = sprint_env.make_env_cfg()
    sprint_env._apply_race_mode(cfg, sprint_env.RACE_VARIANTS["Y"], play=False)
    assert all(term.noise is None for term in cfg.observations["actor"].terms.values())
