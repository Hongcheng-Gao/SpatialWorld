# Environment Management with uv

This directory contains the `uv` environment definitions for SpatialWorld.
The source code lives at the repository root, while this directory only manages
Python dependencies and lock files.

There are two supported ways to install dependencies:

1. Use one isolated project per simulator, for example `envs/ai2thor/`.
2. Use the grouped project at `envs/pyproject.toml` and select dependency groups
   with `uv sync --group <name>`.

The per-simulator projects are the recommended default because they keep heavy
or conflicting simulator dependencies separated.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, make sure `uv` is available in your shell:

```bash
uv --version
```

## Recommended Setup

Create a virtual environment for the simulator you want to run:

```bash
cd envs/ai2thor
uv sync
source .venv/bin/activate
```

Run commands from the repository root after activation:

```bash
cd ../..
python -m scripts.ai2thor.run_benchmark \
  --csv "experiments/csv/ai2thor/Spatial-Annotation-ai2thor-gpt-5.csv" \
  --config "experiments/configs/ai2thor/config_close_gpt-5.yaml"
```

You can also avoid activation by calling the environment Python directly:

```bash
envs/ai2thor/.venv/bin/python -m scripts.ai2thor.run_benchmark \
  --csv "experiments/csv/ai2thor/Spatial-Annotation-ai2thor-gpt-5.csv" \
  --config "experiments/configs/ai2thor/config_close_gpt-5.yaml"
```

## Unified Grouped Setup

The top-level `envs/pyproject.toml` provides a single `uv` project with
dependency groups. This is useful when you want one environment that can run
multiple simulators.

```bash
cd envs
uv sync --group ai2thor
source .venv/bin/activate
```

Multiple groups can be installed together:

```bash
uv sync --group ai2thor --group procthor --group dev
```

Use grouped installs carefully: some simulators pin old transitive dependencies,
so installing all groups together may be heavier than using isolated
`envs/<environment>/` projects.

## Available Environments

| Group | Simulator | Main extra dependencies |
| --- | --- | --- |
| `ai2thor` | AI2-THOR | ai2thor, pygame, opencv |
| `procthor` | ProcTHOR | ai2thor, prior, matplotlib |
| `virtualhome` | VirtualHome | virtualhome, scipy, ikpy |
| `carla` | CARLA | pygame, opencv |
| `embodiedcity` | EmbodiedCity | airsim, networkx, scipy |
| `game` | Game | pygame, PyOpenGL, opencv |
| `dev` | Development and tests | pytest |

## Project Layout

- `mllm_base_agent/` contains agent logic, LLM provider code, prompts, and
  environment wrappers.
- `evaluation/` contains environment-specific evaluators, metrics, getters, and
  semantic mappings.
- `actions/` contains shared action parsing and max-step resolution utilities.
- `configs/` contains model, simulator, and task configuration files.
- `data/` contains task definitions and environment data.
- `experiments/` contains benchmark CSV results.
- `scripts/` contains the runnable entry points.

The root `pyproject.toml` exposes the source packages in editable mode. Each
`envs/<environment>/pyproject.toml` depends on that root package through a local
path dependency, so source edits are visible immediately after `uv sync`.

## Running Benchmarks

Install the matching environment first, then run the script from the repository
root. Examples:

```bash
envs/ai2thor/.venv/bin/python -m scripts.ai2thor.run_benchmark \
  --csv "experiments/csv/ai2thor/Spatial-Annotation-ai2thor-gpt-5.csv" \
  --config "experiments/configs/ai2thor/config_close_gpt-5.yaml"
```

```bash
envs/procthor/.venv/bin/python -m scripts.procthor.run_benchmark \
  --config "configs/procthor/config.yaml"
```

```bash
envs/game/.venv/bin/python -m scripts.game.run_benchmark \
  --config "configs/game/maze_config.py"
```

## Maintenance

When you change dependencies in a per-simulator project, update its lock file:

```bash
cd envs/ai2thor
uv lock
uv sync
```

When you change grouped dependencies, update the top-level environment lock:

```bash
cd envs
uv lock
uv sync
```

Keep generated virtual environments out of git. The `.venv/` directories are
local artifacts and should not be committed.

## Troubleshooting

- If `uv` cannot resolve dependencies, try the isolated project for the target
  simulator first.
- If a simulator import fails, confirm you are using the matching environment
  Python, for example `envs/procthor/.venv/bin/python`.
- If a lock file changes unexpectedly, rerun `uv lock` from the corresponding
  environment directory and review the diff before committing.
