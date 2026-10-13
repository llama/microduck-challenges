"""Registers MicroDuck Got a Talent's task with mjlab, and its contract (challenge.toml) with publish."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck import challenge
from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from .env import RL_CFG, make_env_cfg

register_mjlab_task(
    task_id=challenge.register(__file__).task,
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=RL_CFG,
    runner_cls=MicroduckOnPolicyRunner,
)

# Dizzy Duck stage A: the spin alone (no sit commands, standing starts). Not a
# challenge task (publish never sees it); its runs land in the same
# logs/rsl_rl/got_a_talent/ so stage B can warm-start from them.
SPIN_TASK = "Mjlab-GotATalentSpin-MicroDuck"
register_mjlab_task(
    task_id=SPIN_TASK,
    env_cfg=make_env_cfg(stage="spin"),
    play_env_cfg=make_env_cfg(play=True, stage="spin"),
    rl_cfg=RL_CFG,
    runner_cls=MicroduckOnPolicyRunner,
)
# Stage A in the "fast" style (v1's speed, tidier form); see SPIN_STYLES in env.py.
SPIN_FAST_TASK = "Mjlab-GotATalentSpinFast-MicroDuck"
register_mjlab_task(
    task_id=SPIN_FAST_TASK,
    env_cfg=make_env_cfg(stage="spin", style="fast"),
    play_env_cfg=make_env_cfg(play=True, stage="spin", style="fast"),
    rl_cfg=RL_CFG,
    runner_cls=MicroduckOnPolicyRunner,
)
