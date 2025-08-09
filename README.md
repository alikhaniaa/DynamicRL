# DynamicRL: A Framework for Interactive Reinforcement Learning

This repository contains the framework for `DynamicRL`, a toolkit for dynamically and interactively tuning Reinforcement Learning agents during training.

## Project Structure:

```text
DynamicRL/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── configs/
│   ├── defaults.yaml
│   ├── algo/
│   │   └── ppo.yaml
│   ├── env/
│   │   ├── mujoco_ant.yaml
│   │   └── mujoco_humanoid.yaml
│   └── run/
│       └── local_debug.yaml
├── dynamicrl/                   # installable package: `dynamicrl`
│   ├── __init__.py
│   ├── core/
│   │   ├── trainer.py           # data-plane coordinator (safe pause points)
│   │   ├── algorithm.py         # base Algo API + registry
│   │   ├── policies.py          # base Policy/Value interfaces
│   │   ├── data_buffer.py       # rollout storage (on-policy) + samplers
│   │   ├── advantage.py         # GAE/returns
│   │   ├── optim.py             # PPO updater loops
│   │   ├── environment.py       # unified env runner (vec/async) + seeding
│   │   ├── control.py           # control-plane event loop + HyperparamServer
│   │   ├── checkpoint.py        # atomic save/restore of full state
│   │   ├── logging.py           # Rich console, CSV/TensorBoard, sys metrics
│   │   ├── video.py             # record/stream episodes; pause previews
│   │   ├── events.py            # typed events (pause/resume/patch/etc.)
│   │   └── utils.py             # RNG, determinism, typing helpers
│   ├── algorithms/
│   │   └── ppo/
│   │       ├── __init__.py
│   │       ├── ppo_algorithm.py # implements Algorithm API using PPO
│   │       └── nets.py          # MuJoCo-ready MLP actor-critic
│   ├── envs/
│   │   ├── gymnasium_adapter.py # Gymnasium+MuJoCo adapter + wrappers
│   │   ├── vec/
│   │   │   ├── sync_vec.py
│   │   │   └── async_vec.py
│   │   └── wrappers/
│   │       └── normalize.py
│   ├── external/                # placeholder for future BCI streams
│   │   └── provider.py          # interface + mock provider
│   └── ui/
│       ├── cli.py               # Typer-based CLI for runs + live tweaks
│       └── api.py               # optional REST for remote control
├── scripts/
│   ├── train_ppo.py             # minimal runnable entrypoint
│   └── inspect_checkpoint.py
├── tests/
│   ├── test_repro.py            # pause/resume bitwise tests
│   ├── test_ppo_cartpole.py
│   ├── test_pause_resume.py
│   └── test_alg_plugin.py
└── experiments/
    └── mujoco_ant_interactive.yaml
