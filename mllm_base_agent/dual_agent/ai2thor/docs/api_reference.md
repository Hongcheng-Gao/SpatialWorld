# Dual-Agent API Reference

## Core Functions

### `create_dual_agent_graph()`

Create the dual-agent state machine graph.

```python
from .core.agent.graph import create_dual_agent_graph

app = create_dual_agent_graph()
final_state = app.invoke(initial_state)
```

**Returns:** compiled `StateGraph` instance

---

### `initialize_dual_agent_state()`

Initialize dual-agent state.

```python
from .core.agent.state import initialize_dual_agent_state

state = initialize_dual_agent_state(
    task_prompt="Open the fridge",
    env=env,
    vlm=vlm,
    config=config_dict,
    max_global_steps=60,
    collaboration_mode="alternating",
    run_output_dir="outputs/task_xxx"
)
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `task_prompt` | `str` | Task instruction |
| `env` | `Any` | Environment instance |
| `vlm` | `BaseChatModel` | VLM instance |
| `config` | `Dict` | Config dictionary |
| `max_global_steps` | `int` | Global step budget |
| `collaboration_mode` | `str` | Collaboration mode |
| `run_output_dir` | `str` | Output directory |

**Returns:** initialized `DualAgentState`

---

### `parse_action_string()`

Parse an action string into a normalized action dictionary.

```python
from .core.agent.graph import parse_action_string

action = parse_action_string("OpenObject(Fridge)")
# {"action_type": "interaction", "action_name": "OpenObject", "object_type": "Fridge"}

action = parse_action_string("MoveAhead")
# {"action_type": "navigation", "action_name": "MoveAhead"}

action = parse_action_string("DONE")
# {"action_type": "task_completion", "action_name": "DONE"}

action = parse_action_string("Pass()")
# {"action_type": "pass", "action_name": "Pass"}
```

**Supported action families:**
- Navigation: `MoveAhead`, `MoveBack`, `MoveLeft`, `MoveRight`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`
- Interaction: `PickupObject(X)`, `OpenObject(X)`, `CloseObject(X)`, `ToggleObjectOn(X)`, `ToggleObjectOff(X)`, ...
- Task completion: `DONE`, `FAIL`
- Skip: `Pass()`

---

### `get_vlm()`

Create a VLM instance.

```python
from .core.llm import get_vlm

vlm = get_vlm(
    model_name="gpt-4o",
    temperature=0.2,
    max_tokens=2000
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `model_name` | `str` | `None` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `2000` | Max output tokens |
| `agent_config` | `dict` | `None` | Agent-specific config |

**Returns:** `BaseChatModel`

---

### `build_shared_context()`

Build shared context from communication history only.

```python
from .core.agent.state import build_shared_context

context = build_shared_context(state)
```

**Returns:** formatted shared-context string

---

## State Types

### `DualAgentState`

```python
class DualAgentState(TypedDict, total=False):
    task_prompt: str
    subtasks: Dict[str, str]
    agent_1: AgentState
    agent_2: AgentState
    current_agent: str
    env: Any
    vlm: Any
    shared_memory: Dict[str, Any]
    communication_history: List[Dict[str, Any]]
    message_queue: List[Dict[str, Any]]
    global_step_count: int
    max_global_steps: int
    global_success: bool
    global_fail_reason: Optional[str]
    collaboration_mode: str
    turn_count: int
    current_turn_steps: int
    force_agent_switch: bool
    config: Optional[Dict[str, Any]]
    run_output_dir: Optional[str]
```

### `AgentState`

```python
class AgentState(TypedDict, total=False):
    agent_id: str
    agent_role: str
    task_prompt: str
    observation: Optional[EnvObservation]
    step_count: int
    max_steps: int
    success: bool
    fail_reason: Optional[str]
    short_term_history: List[Dict[str, Any]]
    long_term_summary: str
    structured_trajectory: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    next_action: Optional[dict]
    should_continue: bool
    task_done_by_model: bool
    task_fail_by_model: bool
    vlm: Any
    env: Any
    config: Optional[Dict[str, Any]]
```

---

## Prompt Templates

### `get_collaborative_system_prompt()`

Return the equal-collaboration system prompt.

```python
from .dual_agent import get_collaborative_system_prompt

prompt = get_collaborative_system_prompt(enable_summary=False)
```

**Template variables:**
- `{task_prompt}`
- `{shared_context}`

---

## Evaluator

The dual-agent system reuses the main-repo evaluator in `evaluators/base.py`.

### `create_evaluator_from_config()`

```python
from evaluators.base import create_evaluator_from_config

evaluator = create_evaluator_from_config(task_config)
score = evaluator.evaluate(env, metadata)
```

---

## Utility Functions

### `load_task_info()`

```python
from mllm_base_agent.dual_agent.ai2thor.main import load_task_info

task_data = load_task_info("ai2thor05002")
```

### `execute_init_actions()`

```python
from mllm_base_agent.dual_agent.ai2thor.main import execute_init_actions

init_count = execute_init_actions(env, init_actions)
```
