# Task Configuration Example: Move TeddyBear from Bedroom to Bathroom

## Task Description

Move the TeddyBear from the bed in the bedroom to the bathroom.

## Success Condition Setup

### Method 1: Check Target Room Only (Recommended)

```json
{
  "task_id": "procthor00003",
  "task_name": "Move TeddyBear from Bedroom to Bathroom",
  "instruction": "Move the TeddyBear from the bed in the bedroom to the bathroom.",
  "scene_index": 100,
  "target_object_types": ["TeddyBear"],
  "success_conditions": [
    {
      "type": "object_in_room",
      "object_type": "TeddyBear",
      "room_type": "Bathroom"
    }
  ],
  "success_logic": "AND",
  "max_steps": 100
}
```

**Notes:**
- The task succeeds as soon as the TeddyBear is in the bathroom
- There is no need to check that it is no longer in the bedroom

### Method 2: Combined Conditions (Stricter)

If you need to ensure the TeddyBear is no longer in the bedroom, add a reverse condition:

```json
{
  "success_conditions": [
    {
      "type": "object_in_room",
      "object_type": "TeddyBear",
      "room_type": "Bathroom"
    },
    {
      "type": "object_state",
      "object_type": "TeddyBear",
      "state": "isPickedUp",
      "value": false
    }
  ],
  "success_logic": "AND"
}
```

**Notes:**
- First condition: TeddyBear must be in the bathroom
- Second condition: TeddyBear must be placed down (not held)

## Code Support

The evaluator already supports the `object_in_room` condition.

### Implementation

Room membership is determined using **floorPolygon (exact polygon)**:

1. **Prefer floorPolygon**:
   - Read the object's (x, z) position
   - Use `_point_in_polygon()` to test whether the point lies inside a room polygon
   - This is the most accurate method and comes directly from ProcTHOR scene data

2. **Fallback**:
   - If floorPolygon is unavailable, use the object's `roomType` attribute when present

### Code Locations

- **Condition check**: `evaluators/base.py` lines 390-419
- **Room inference**: `_build_room_boundaries_from_house_scene()` and `_point_in_polygon()` in `evaluators/getters.py`

## Usage Examples

### Run Task Evaluation

```bash
python scripts/evaluate_action_sequence.py \
  --task-config tasks/procthor00003/task.json \
  --action-sequence actions.json
```

### Use in Code

```python
from envs.procthor_wrapper import ProcTHOREnvWrapper
import json

with open('tasks/procthor00003/task.json', 'r') as f:
    task_config = json.load(f)

env = ProcTHOREnvWrapper(
    scene_index=task_config['scene_index'],
    config={'task': task_config}
)

observation = env.reset(task_config['instruction'])

from evaluators.base import create_evaluator_from_config
evaluator = create_evaluator_from_config(task_config)
score = evaluator.evaluate(env, env.controller.last_event.metadata)
print(f"Task score: {score}")
```

## Notes

1. **Room name casing**: Matching is case-insensitive (`"Bathroom"` and `"bathroom"` both work)
2. **Object type matching**: Semantic variants are supported (e.g. `"Apple"` also matches `"AppleSliced"`)
3. **Multiple objects**: If several objects share the same type, success is returned when any one satisfies the condition
4. **Object position**: Room membership uses the object's `position.x` and `position.z` coordinates

## Other Success Condition Types

- `object_state`: Check object state (e.g. `isOpen`, `isPickedUp`)
- `object_in_hand`: Check whether an object is held
- `object_in_receptacle`: Check whether an object is inside a receptacle
- `agent_in_room`: Check whether the agent is in a specified room
