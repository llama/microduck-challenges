"""Dizzy Duck: the spin flag in twist[2] and the rewards that pay a pirouette, on synthetic states."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import list_tasks

import microduck_challenges  # noqa: F401  (registers every task)
from microduck_challenges.got_a_talent import rewards as dizzy


def _quat_yaw(yaws):
    half = torch.tensor(yaws) / 2
    zero = torch.zeros_like(half)
    return torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)


def _env(yaws, feet, flag, gravity_z=-1.0, rate=0.0, pos=None, step=10):
    n = len(yaws)
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_quat_w=_quat_yaw(yaws),
        projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]] * n),
        root_link_ang_vel_w=torch.tensor([[0.0, 0.0, rate]] * n),
        root_link_pos_w=torch.zeros(n, 3) if pos is None else torch.tensor(pos),
    ))
    sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor(feet, dtype=torch.float32)))
    command = torch.zeros(n, 3)
    command[:, 2] = torch.tensor(flag, dtype=torch.float32)
    manager = SimpleNamespace(get_command=lambda name: command)
    return SimpleNamespace(
        scene={"robot": robot, "feet_ground_contact": sensor},
        num_envs=n, episode_length_buf=torch.full((n,), step),
        command_manager=manager,
    )


ONE_FOOT = [[1.0, 0.0]]
TWO_FEET = [[1.0, 1.0]]
AIRBORNE = [[0.0, 0.0]]


def test_spin_progress_pays_counter_clockwise_yaw_on_one_foot():
    env = _env([0.0], ONE_FOOT, [1.0])
    dizzy.spin_progress(env)  # primes the previous yaw
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([0.1])
    assert dizzy.spin_progress(env).item() == pytest.approx(0.1, abs=1e-5)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([0.0])
    assert dizzy.spin_progress(env).item() == pytest.approx(-0.1, abs=1e-5)


def test_spin_progress_wraps_across_pi():
    env = _env([3.1], ONE_FOOT, [1.0])
    dizzy.spin_progress(env)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([-3.1])
    assert dizzy.spin_progress(env).item() == pytest.approx(2 * math.pi - 6.2, abs=1e-5)


def test_spin_progress_scales_with_support_and_needs_the_flag():
    for feet, expect in ((ONE_FOOT, 0.1), (TWO_FEET, 0.015), (AIRBORNE, 0.0)):
        env = _env([0.0], feet, [1.0])
        dizzy.spin_progress(env)
        env.scene["robot"].data.root_link_quat_w = _quat_yaw([0.1])
        assert dizzy.spin_progress(env).item() == pytest.approx(expect, abs=1e-5), feet
    env = _env([0.0], ONE_FOOT, [0.0])
    dizzy.spin_progress(env)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([0.1])
    assert dizzy.spin_progress(env).item() == 0.0


def test_spin_progress_pays_nothing_tilted_or_on_a_fresh_episode():
    env = _env([0.0], ONE_FOOT, [1.0], gravity_z=-0.3)
    dizzy.spin_progress(env)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([0.1])
    assert dizzy.spin_progress(env).item() == 0.0
    env = _env([0.0], ONE_FOOT, [1.0], step=1)
    dizzy.spin_progress(env)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([2.0])
    assert dizzy.spin_progress(env).item() == 0.0


def test_spin_progress_is_capped_per_step():
    env = _env([0.0], ONE_FOOT, [1.0])
    dizzy.spin_progress(env)
    env.scene["robot"].data.root_link_quat_w = _quat_yaw([1.0])
    assert dizzy.spin_progress(env).item() == pytest.approx(0.3, abs=1e-5)


def test_spin_coast_pays_only_a_fast_one_foot_turn():
    assert dizzy.spin_coast(_env([0.0], ONE_FOOT, [1.0], rate=6.0)).item() == pytest.approx(1.0)
    assert dizzy.spin_coast(_env([0.0], ONE_FOOT, [1.0], rate=3.75)).item() == pytest.approx(0.5)
    assert dizzy.spin_coast(_env([0.0], ONE_FOOT, [1.0], rate=1.0)).item() == 0.0
    assert dizzy.spin_coast(_env([0.0], TWO_FEET, [1.0], rate=6.0)).item() == 0.0
    assert dizzy.spin_coast(_env([0.0], ONE_FOOT, [0.0], rate=6.0)).item() == 0.0
    assert dizzy.spin_coast(_env([0.0], ONE_FOOT, [1.0], rate=-6.0)).item() == 0.0


def test_spin_drift_anchors_where_the_spin_was_commanded():
    env = _env([0.0], ONE_FOOT, [0.0], pos=[[1.0, 1.0, 0.1]])
    assert dizzy.spin_drift(env).item() == 0.0
    env.command_manager.get_command("twist")[:, 2] = 1.0  # flag rises here: anchor (1, 1)
    assert dizzy.spin_drift(env).item() == 0.0
    env.scene["robot"].data.root_link_pos_w = torch.tensor([[1.0, 1.08, 0.1]])
    assert dizzy.spin_drift(env).item() == pytest.approx(0.05, abs=1e-6)
    env.command_manager.get_command("twist")[:, 2] = 0.0
    assert dizzy.spin_drift(env).item() == 0.0


def test_unless_spinning_gates_the_inner_term():
    inner = lambda env, k: torch.full((env.num_envs,), k)  # noqa: E731
    assert dizzy.unless_spinning(_env([0.0], ONE_FOOT, [0.0]), inner, {"k": 2.0}).item() == 2.0
    assert dizzy.unless_spinning(_env([0.0], ONE_FOOT, [1.0]), inner, {"k": 2.0}).item() == 0.0
    assert dizzy.unless_spinning(_env([0.0], ONE_FOOT, [1.0]), inner, {"k": 2.0}, floor=0.3).item() == pytest.approx(0.6)


@pytest.fixture(scope="module")
def env_module():
    return importlib.import_module("microduck_challenges.got_a_talent.env")


def test_both_stages_read_the_spin_flag_and_keep_the_layout(env_module):
    for stage in env_module.SPIN_STAGES:
        cfg = env_module.make_env_cfg(stage=stage)
        twist = cfg.commands["twist"]
        assert isinstance(twist, dizzy.SpinSitStandCommandCfg)
        assert twist.spin_prob == env_module.SPIN_PROB[stage]
        assert twist.sit_prob == env_module.SIT_PROB_BY_STAGE[stage]
        assert list(cfg.observations["actor"].terms)[-3:] == ["command", "head_command", "body_command"]
        for name in ("spin_progress", "spin_coast", "spin_drift"):
            assert name in cfg.rewards
        assert cfg.rewards["spin_drift"].weight < 0
        assert cfg.rewards["spin_progress"].weight > 0
        for name in ("posture_pose_legs", "posture_pose_l1", "posture_stillness", "posture_composite", "head_pose_tracking"):
            assert cfg.rewards[name].func is dizzy.unless_spinning, name
        assert cfg.rewards["head_pose_tracking"].params["floor"] == env_module.SPIN_HEAD_FLOOR


def test_stage_a_spins_from_standing_only(env_module):
    cfg = env_module.make_env_cfg(stage="spin")
    assert cfg.events["set_ground_state"].params["sitting_prob"] == 0.0
    assert cfg.events["set_ground_state"].params["standing_prob"] == 1.0
    assert cfg.episode_length_s == 8.0
    full = env_module.make_env_cfg(stage="full")
    assert full.events["set_ground_state"].params["sitting_prob"] == 0.5
    assert full.commands["twist"].resampling_time_range == (3.5, 6.5)


def test_the_spin_task_is_registered():
    from microduck_challenges.got_a_talent.tasks import SPIN_TASK

    assert SPIN_TASK in list_tasks()
    assert "Mjlab-GotATalent-MicroDuck" in list_tasks()


def test_the_manifest_names_the_spin_flag():
    from mjlab_microduck import challenge as ch

    assert "spin flag" in ch.for_task("Mjlab-GotATalent-MicroDuck").command["twist"]


def test_style_costs_are_zero_off_the_flag_and_positive_on_it():
    env = _env([0.0], ONE_FOOT, [1.0], gravity_z=-0.866, rate=12.0)
    assert dizzy.spin_tilt(env).item() == pytest.approx(0.134, abs=1e-3)
    assert dizzy.spin_overspeed(env).item() == pytest.approx(9.0)
    env.command_manager.get_command("twist")[:, 2] = 0.0
    assert dizzy.spin_tilt(env).item() == 0.0
    assert dizzy.spin_overspeed(env).item() == 0.0


def test_the_style_terms_are_costs_or_off_in_both_stages(env_module):
    for stage in env_module.SPIN_STAGES:
        cfg = env_module.make_env_cfg(stage=stage)
        style = env_module.SPIN_STYLES[env_module.SPIN_STYLE]
        for name, key in (("spin_tilt", "tilt_w"), ("spin_head_vel", "head_vel_w"),
                          ("spin_overspeed", "overspeed_w"), ("spin_head_pose", "head_pose_w")):
            assert cfg.rewards[name].weight <= 0, name
            assert cfg.rewards[name].weight == style[key], name
        assert cfg.rewards["spin_swing_lift"].weight == style["lift_w"] >= 0
        assert cfg.rewards["spin_head_drop"].weight == style["head_drop_w"] <= 0
        assert cfg.rewards["spin_progress"].params["max_step_rad"] == env_module.SPIN_STEP_CAP


def test_the_fast_style_keeps_v1_speed_but_prices_tilt_and_lift(env_module):
    cfg = env_module.make_env_cfg(stage="spin", style="fast")
    assert cfg.rewards["spin_progress"].params["max_step_rad"] >= 0.3
    assert cfg.rewards["spin_overspeed"].weight == 0.0
    assert cfg.rewards["spin_tilt"].weight < 0 and cfg.rewards["spin_swing_lift"].weight > 0
    from microduck_challenges.got_a_talent.tasks import SPIN_FAST_TASK
    assert SPIN_FAST_TASK in list_tasks()


def test_touchdowns_and_streaks():
    env = _env([0.0], ONE_FOOT, [1.0], rate=6.0)
    env.step_dt = 0.02
    assert dizzy.spin_touchdowns(env).item() == 0.0            # primes: 1 foot
    env.scene["feet_ground_contact"].data.found = torch.tensor(TWO_FEET, dtype=torch.float32)
    assert dizzy.spin_touchdowns(env).item() == 1.0            # 1 -> 2: a touchdown
    assert dizzy.spin_touchdowns(env).item() == 0.0            # staying down is not a new one
    env = _env([0.0], ONE_FOOT, [1.0], rate=6.0); env.step_dt = 0.5
    assert dizzy.spin_streak(env).item() == pytest.approx(0.25)   # 0.5 s of 2 s
    assert dizzy.spin_streak(env).item() == pytest.approx(0.5)
    env.scene["feet_ground_contact"].data.found = torch.tensor(TWO_FEET, dtype=torch.float32)
    assert dizzy.spin_streak(env).item() == 0.0                   # streak broken


def test_the_wilder_style_is_the_show_style(env_module):
    assert env_module.SPIN_STYLE in ("wilder", "fastwilder")
    cfg = env_module.make_env_cfg(stage="full")
    assert cfg.rewards["spin_touchdowns"].weight < 0 and cfg.rewards["spin_streak"].weight > 0
    assert cfg.rewards["spin_swing_lift"].params["clearance_target"] == 0.04
    assert cfg.rewards["spin_overspeed"].weight == 0.0


def test_the_show_style_prices_a_dropped_head(env_module):
    cfg = env_module.make_env_cfg(stage="full")
    assert cfg.rewards["spin_head_drop"].weight <= -50
    assert cfg.rewards["spin_head_drop"].params["asset_cfg"].body_names == ("jaw_soft",)


def test_the_show_task_ramps_the_spin_in_after_the_sit(env_module):
    cfg = env_module.make_env_cfg(stage="full")
    assert cfg.commands["twist"].spin_prob == env_module.SPIN_PROB["full"]
    stages = cfg.curriculum["spin_prob"].params["stages"]
    assert stages[-1]["prob"] > 0.3
    stages = [{"step": 0, "prob": 0.0}, {"step": 1000 * 24, "prob": 0.15}, {"step": 1500 * 24, "prob": 0.35}]
    assert "spin_prob" not in env_module.make_env_cfg(stage="spin").curriculum
    term = SimpleNamespace(_spin_prob=0.0)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: term), common_step_counter=1200 * 24)
    assert dizzy.spin_prob_curriculum(env, None, stages=stages).item() == pytest.approx(0.15)
    assert term._spin_prob == pytest.approx(0.15)


def test_spin_bootstrap_pays_any_rotation_and_rest_terms_are_gated(env_module):
    assert dizzy.spin_bootstrap(_env([0.0], TWO_FEET, [1.0], rate=1.5)).item() == pytest.approx(0.5)
    assert dizzy.spin_bootstrap(_env([0.0], TWO_FEET, [1.0], rate=-9.0)).item() == 1.0
    assert dizzy.spin_bootstrap(_env([0.0], TWO_FEET, [1.0], rate=0.8), min_rate=1.0, full_rate=4.0).item() == 0.0
    assert dizzy.spin_bootstrap(_env([0.0], TWO_FEET, [1.0], rate=2.5), min_rate=1.0, full_rate=4.0).item() == pytest.approx(0.5)
    assert env_module.ENTROPY_COEF == 0.01 and env_module.RL_CFG.algorithm.entropy_coef == 0.01  # B8, the published run
    assert dizzy.spin_bootstrap(_env([0.0], TWO_FEET, [0.0], rate=9.0)).item() == 0.0
    cfg = env_module.make_env_cfg(stage="full")
    for name in ("upright_linear", "upright_while_tall", "posture_height", "posture_height_sharp"):
        assert cfg.rewards[name].func is dizzy.unless_spinning and cfg.rewards[name].params["floor"] == env_module.SPIN_REST_FLOOR
    assert cfg.rewards["spin_bootstrap"].weight > 0
