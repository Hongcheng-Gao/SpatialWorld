# 3D Maze Game MLLM Integration Guide

## Overview

This guide explains how to connect the existing 3D maze game (`maze3d_pygame.py`) to the MLLM evaluation framework so multimodal large language models can be tested and benchmarked in the game.

## File Structure

```
games/maze3d/
├── maze3d_pygame.py          # Original 3D maze game
├── maze3d_adapter.py         # Game adapter class
├── maze3d_demo.py            # Full demo script
├── run_maze_evaluation.py    # Evaluation runner
├── test_adapter.py           # Adapter tests
└── README.md                 # This guide
```

## Core Components

### 1. `Maze3DGame` Adapter

`Maze3DGame` implements the interface required by the MLLM framework:

- `init()`: Initialize the game
- `update()`: Update game state
- `render()`: Render the game view
- `get_state()`: Return game state
- `reset()`: Reset the game

### 2. AI Model Functions

Several AI models are provided for testing:

- `random_maze_ai_model()`: Random policy
- `heuristic_maze_ai_model()`: Heuristic policy
- `intelligent_maze_ai_model()`: Stronger heuristic policy

### 3. Callbacks

- `frame_callback()`: Frame data callback
- `state_callback()`: Game state callback
- `action_callback()`: Action execution callback

## Usage

### Basic Integration Test

```bash
# Run from the project root
conda run -n pygame python -c "
from games.maze3d.maze3d_adapter import Maze3DGame
from input_sources.pygame_input_source import PygameInputSource

game = Maze3DGame()
input_source = PygameInputSource()

success = input_source.initialize(game_module=game, headless=True, screen_size=(800, 600))
print(f'Initialization: {success}')

state = input_source.get_game_state()
print(f'Game state: {state.normalized_state}')

input_source.close()
"
```

### Run Full Evaluation

```bash
# Run from the project root
conda run -n pygame python games/maze3d/run_maze_evaluation.py
```

### Run Adapter Tests

```bash
# Run from the maze3d directory
conda run -n pygame python test_adapter.py
```

## Game State Fields

The adapter exposes:

- **Player**: position and heading
- **Exit**: position
- **Game status**: win flag, step count, max steps
- **Distance**: Manhattan distance to the exit
- **Performance**: animation time, remaining steps

## Supported Actions

- `key_press "w"`: move forward
- `key_press "a"`: turn left
- `key_press "d"`: turn right

## Configuration

### Evaluation Parameters

- `max_steps`: maximum steps (default 300)
- `decision_frequency`: decision rate in Hz (default 2.0)
- `headless`: headless mode (default True)

### Game Parameters

- `maze_file`: optional maze file path
- `max_steps`: in-game step limit (default 1000)

## Logs and Output

Evaluation results are written under `logs/` and include:

- performance stats (FPS, inference time, etc.)
- game outcome (success, step count, etc.)
- detailed state transition logs

## Extension Guide

### Add a New AI Model

1. Implement a new AI model function
2. Register it in `run_maze_evaluation.py`
3. Tune decision logic and inference latency

### Customize Game State

Modify `Maze3DGame.get_state()` to expose additional state fields.

### Support New Actions

Add action handlers in the adapter and call them from the AI model.

## Troubleshooting

### Common Issues

1. **Import errors**: verify Python path setup
2. **Pygame init failure**: check headless mode settings
3. **State read failure**: verify adapter interface implementation

### Debugging Tips

1. Use `log_level=logging.DEBUG` for verbose logs
2. Run without headless mode to inspect the game window
3. Inspect callback output for state transitions

## Performance Tips

- Lower decision frequency to reduce compute load
- Use headless mode for faster runs
- Optimize AI inference time
- Set a reasonable `max_steps` to avoid infinite loops

## Summary

The adapter connects the 3D maze game to the MLLM evaluation framework with a complete interface, multiple example AI policies, and a full evaluation workflow.
