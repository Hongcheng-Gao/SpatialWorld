# Agent-Near-Object Success Condition Guide

## Basic Format

Add an `agent_near_object` condition to `success_conditions` in `task.json`:

```json
{
  "type": "agent_near_object",
  "object_type": "Bed",
  "distance": 1.0
}
```

## Parameters

- **`type`**: Must be `"agent_near_object"`
- **`object_type`**: Target object type (e.g. `"Bed"`, `"Fridge"`, `"Sofa"`)
- **`distance`**: Maximum distance in meters (default `1.0` when omitted)

## Full Examples

### Example 1: Move Near Bed (within 1 m)

```json
{
  "task_id": "procthor00004",
  "task_name": "Move to bed",
  "instruction": "Move the agent to near the bed in the bedroom.",
  "scene_index": 100,
  "target_object_types": ["Bed"],
  "success_conditions": [
    {
      "type": "agent_near_object",
      "object_type": "Bed",
      "distance": 1.0
    }
  ],
  "success_logic": "AND",
  "max_steps": 100
}
```

### Example 2: Move Close to Fridge (within 0.5 m)

```json
{
  "task_id": "procthor00005",
  "task_name": "Move close to fridge",
  "instruction": "Move very close to the refrigerator.",
  "success_conditions": [
    {
      "type": "agent_near_object",
      "object_type": "Fridge",
      "distance": 0.5
    }
  ],
  "success_logic": "AND"
}
```

### Example 3: Near Bed + In Bedroom

```json
{
  "task_id": "procthor00006",
  "task_name": "Move to bed in bedroom",
  "instruction": "Move to the bed in the bedroom.",
  "success_conditions": [
    {
      "type": "agent_near_object",
      "object_type": "Bed",
      "distance": 1.0
    },
    {
      "type": "agent_in_room",
      "room_type": "Bedroom"
    }
  ],
  "success_logic": "AND"
}
```

### Example 4: Near Sofa (within 2 m)

```json
{
  "task_id": "procthor00007",
  "task_name": "Move near sofa",
  "instruction": "Move near the sofa in the living room.",
  "success_conditions": [
    {
      "type": "agent_near_object",
      "object_type": "Sofa",
      "distance": 2.0
    }
  ],
  "success_logic": "AND"
}
```

## Distance Calculation

- **2D distance**: Uses X and Z only; Y (height) is ignored
- **Formula**: `distance = sqrt((agent_x - object_x)² + (agent_z - object_z)²)`
- **Unit**: meters

## Supported Object Types

All ProcTHOR object types are supported, for example:

- **Furniture**: `Bed`, `Sofa`, `Chair`, `Table`, `CounterTop`
- **Appliances**: `Fridge`, `Microwave`, `Stove`, `TV`
- **Containers**: `Cabinet`, `Drawer`, `Sink`
- **Other**: Any object type present in the scene

**Note:** Semantic variants are supported (e.g. `"Apple"` also matches `"AppleSliced"`).

## Implementation

1. Read the agent's (x, z) position
2. Find all scene objects matching `object_type` (including semantic variants)
3. Compute 2D distance from the agent to each match
4. Return success (1.0) if the shortest distance is ≤ `distance`, otherwise failure (0.0)

## Code Locations

- **Condition check**: `evaluators/base.py` lines 445-490
- **Distance check**: `check_agent_near_object()` in `evaluators/metrics.py`
- **Position lookup**: `get_agent_position()` in `evaluators/getters.py`

## Debug Output

During evaluation, detailed debug logs are printed:

```
Checking distance from agent to Bed:
Agent position: (6.75, 5.25)
Nearest object: Bed|1|2|3, distance: 0.85m, threshold: 1.00m
Agent is near Bed (distance: 0.85m <= 1.00m)
```

## Combining with Other Conditions

`agent_near_object` can be combined with other success conditions:

```json
{
  "success_conditions": [
    {
      "type": "agent_near_object",
      "object_type": "Bed",
      "distance": 1.0
    },
    {
      "type": "agent_in_room",
      "room_type": "Bedroom"
    },
    {
      "type": "object_state",
      "object_type": "Bed",
      "state": "isOpen",
      "value": false
    }
  ],
  "success_logic": "AND"
}
```
