"""How well does a policy pirouette on the Arena's stage?

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.eval_spin policy.onnx \
        [--spin-s 4.0] [--stand-s 1.5] [--seeds 1 2 3] [--timeline FILE]

Plays the Arena's stage exactly as `arena.runner.run_stage` does (same reset,
settle, servos and command delay), on a timeline that stands, raises the spin
flag (twist[2]) for `--spin-s`, then drops it, and records what the trace
alone cannot show: which feet touch the floor each control step. Reports, per
seed, the turn achieved while the flag was up, the share of it done on one
foot, the longest one-foot coast, the peak yaw rate, the drift from the spot,
whether the duck stood again within 1.5 s of the flag dropping, and the fall.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from arena import runner, stage
from arena.constants import CTRL_DT
from arena.manifest import load_event
from arena.policy import load_policy
from arena.timeline import Timeline, load_timeline

EVENT = "got-a-talent"
REACH_S = 1.5
COAST_MIN_RATE = 1.5  # rad/s: below this a one-foot stance is a stand, not a coast


def spin_timeline(stand_s: float, spin_s: float, rest_s: float = 4.0) -> Timeline:
    zero = (0.0,) * 13
    spin = (0.0, 0.0, 1.0) + (0.0,) * 10
    duration = stand_s + spin_s + rest_s
    return Timeline(duration, (0.0, stand_s, stand_s + spin_s), (zero, spin, zero))


def _yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def play(policy, timeline: Timeline, seed: int):
    """The stage run, plus per-step feet-on-floor counts and trunk yaw rate."""
    manifest = load_event(EVENT)
    sim, settle = runner._start(
        manifest, policy, seed, None, np.asarray(timeline.command_at(0.0), dtype=np.float64)
    )
    model, data = sim.model, sim.data
    soles = {model.geom("left_foot_collision").id, model.geom("right_foot_collision").id}
    floor = model.geom("floor").id
    trunk = model.body("trunk_base").id
    steps = round(timeline.duration_s / CTRL_DT)
    frames = np.empty((steps, runner.TRACE_WIDTH), dtype=np.float32)
    feet = np.zeros(steps, dtype=np.int8)
    rate = np.zeros(steps, dtype=np.float32)
    written = 0
    for step in range(steps):
        sim.step(np.asarray(timeline.command_at(step * CTRL_DT), dtype=np.float64))
        if runner._reset_by_mujoco(data):
            break
        sim.record(frames[written])
        touching = set()
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if floor in pair:
                touching |= pair & soles
        feet[written] = len(touching)
        rate[written] = data.cvel[trunk][2]  # world-frame yaw rate of the trunk
        written += 1
    return frames[:written], feet[:written], rate[:written], written == steps


def analyse(trace, feet, rate, timeline: Timeline, finished: bool, termination) -> dict:
    t = np.arange(len(trace)) * CTRL_DT
    flag = np.array([timeline.command_at(x)[2] >= 0.5 for x in t])
    yaw = np.unwrap(_yaw(trace[:, 3:7].astype(np.float64)))
    dyaw = np.diff(yaw, prepend=yaw[0])
    on = flag
    turn = float(np.degrees(yaw[on][-1] - yaw[on][0])) if on.any() else 0.0
    one = (feet == 1) & on
    turn_one_foot = float(np.degrees(dyaw[one].sum()))
    # longest run of consecutive one-foot steps with a real yaw rate, in degrees turned
    best, cur = 0.0, 0.0
    for k in range(len(trace)):
        if on[k] and feet[k] == 1 and abs(rate[k]) >= COAST_MIN_RATE:
            cur += abs(np.degrees(dyaw[k]))
            best = max(best, cur)
        else:
            cur = 0.0
    peak_rate = float(np.abs(rate[on]).max()) if on.any() else 0.0
    start = trace[on][0, :2] if on.any() else trace[0, :2]
    drift = float(np.linalg.norm(trace[on][:, :2] - start, axis=1).max()) if on.any() else 0.0
    # recovery: standing (Arena's reading) within REACH_S after the flag drops, and held
    off_after = np.where(~on & (t > (t[on][0] if on.any() else 0.0)))[0]
    recovered, stood_s, held = None, None, None
    if len(off_after):
        standing = stage.standing(trace)[off_after]
        if standing.any():
            first = int(np.argmax(standing))
            stood_s = first * CTRL_DT                       # first standing reading after the flag drops
            held = bool(standing[first:].all())             # and never left it
            recovered = stood_s if held and stood_s <= REACH_S else None
    fell = stage.fell_at(trace, termination)
    return dict(
        turn_deg=round(turn, 1),
        one_foot_deg=round(turn_one_foot, 1),
        one_foot_share=round(float(one.sum() / max(on.sum(), 1)), 2),
        longest_coast_deg=round(best, 1),
        peak_rate=round(peak_rate, 2),
        drift_m=round(drift, 3),
        recovered_s=None if recovered is None else round(recovered, 2),
        stood_after_s=None if stood_s is None else round(stood_s, 2),
        held_stand=held,
        fell_at=None if fell is None else round(float(fell), 2),
        finished=finished,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a Dizzy Duck policy's pirouette on the Arena's stage.")
    ap.add_argument("onnx")
    ap.add_argument("--spin-s", type=float, default=4.0)
    ap.add_argument("--stand-s", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--timeline", default=None, help="play this timeline instead of the synthetic one")
    args = ap.parse_args(argv)
    event = load_event(EVENT)
    timeline = (
        load_timeline(Path(args.timeline), event.time_limit_s)
        if args.timeline
        else spin_timeline(args.stand_s, args.spin_s)
    )
    policy = load_policy(Path(args.onnx))
    rows = []
    for seed in args.seeds:
        trace, feet, rate, finished = play(policy, timeline, seed)
        r = analyse(trace, feet, rate, timeline, finished, event.termination)
        rows.append(r)
        print(f"seed {seed}: " + "  ".join(f"{k}={v}" for k, v in r.items()))
    turns = [r["turn_deg"] for r in rows]
    falls = sum(r["fell_at"] is not None for r in rows)
    print(f"summary: turn median {np.median(turns):.0f} deg (min {min(turns):.0f}, max {max(turns):.0f}), "
          f"one-foot share {np.median([r['one_foot_share'] for r in rows]):.2f}, "
          f"longest coast {np.median([r['longest_coast_deg'] for r in rows]):.0f} deg, falls {falls}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
