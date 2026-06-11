# Agent-in-Room Success Condition Guide

## Basic Format

Add an `agent_in_room` condition to `success_conditions` in `task.json`:

```json
{
  "type": "agent_in_room",
  "room_type": "Kitchen"
}
```

## Comparison with `object_in_room`

### Object in Room (requires `object_type`)

```json
{
  "type": "object_in_room",
  "object_type": "TeddyBear",
  "room_type": "Bedroom"
}
```

### Agent in Room (no `object_type` required)

```json
{
  "type": "agent_in_room",
  "room_type": "Kitchen"
}
```

## Full Examples

### Example 1: Agent Position Only

```json
{
  "task_id": "procthor00004",
  "task_name": "Go to kitchen",
  "success_conditions": [
    {
      "type": "agent_in_room",
      "room_type": "Kitchen"
    }
  ],
  "success_logic": "AND"
}
```

### Example 2: Combined Agent + Object Conditions

```json
{
  "task_id": "procthor00005",
  "task_name": "Move TeddyBear to bedroom and stay in kitchen",
  "success_conditions": [
    {
      "type": "object_in_room",
      "object_type": "TeddyBear",
      "room_type": "Bedroom"
    },
    {
      "type": "agent_in_room",
      "room_type": "Kitchen"
    }
  ],
  "success_logic": "AND"
}
```

### Example 3: Multiple Combined Conditions

```json
{
  "task_id": "procthor00006",
  "task_name": "Complete task in bedroom",
  "success_conditions": [
    {
      "type": "agent_in_room",
      "room_type": "Bedroom"
    },
    {
      "type": "object_state",
      "object_type": "Fridge",
      "state": "isOpen",
      "value": true
    },
    {
      "type": "object_in_room",
      "object_type": "Apple",
      "room_type": "Kitchen"
    }
  ],
  "success_logic": "AND"
}
```

## Supported Room Types

- `Kitchen`
- `Bathroom`
- `Bedroom`
- `LivingRoom`
- `DiningRoom`
- Other room types defined by the scene

**Note:** Room names are matched case-insensitively (`"Kitchen"` and `"kitchen"` both work).

## Implementation

The `agent_in_room` condition uses **floorPolygon (exact polygon)** to determine the agent's room:

1. Read the agent's (x, z) position
2. Use `_point_in_polygon()` to test whether the point lies inside a room polygon
3. This is the most accurate method and comes directly from ProcTHOR scene data

## Code Locations

- **Condition check**: `evaluators/base.py` lines 321-388
- **Room inference**: `get_agent_room()` in `evaluators/getters.py`
