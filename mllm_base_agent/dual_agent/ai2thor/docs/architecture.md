# Dual-Agent System Architecture

## 1. Overall Architecture

The dual-agent system extends the single-agent framework. The core idea is that **two equal agents collaborate through explicit communication**.

### 1.1 Architecture Comparison

```
Single-agent:
┌──────────────────────────────────────┐
│           main.py                    │
│              ↓                       │
│  ┌────────────────────────────────┐  │
│  │       AgentState               │  │
│  │  ┌─────┐ ┌─────┐ ┌──────────┐ │  │
│  │  │Think│→│ Act │→│ Evaluate │ │  │
│  │  └─────┘ └─────┘ └──────────┘ │  │
│  └────────────────────────────────┘  │
│              ↓                       │
│           AI2-THOR                   │
└──────────────────────────────────────┘

Dual-agent:
┌────────────────────────────────────────────────────────┐
│                     main.py                            │
│                        ↓                               │
│  ┌──────────────────────────────────────────────────┐ │
│  │              DualAgentState                       │ │
│  │  ┌─────────────┐         ┌─────────────┐        │ │
│  │  │  Agent 1    │◄───────►│  Agent 2    │        │ │
│  │  │ AgentState  │ COMM    │ AgentState  │        │ │
│  │  └─────────────┘         └─────────────┘        │ │
│  │         │                       │               │ │
│  │         └───────────┬───────────┘               │ │
│  │                     ↓                           │ │
│  │  ┌───────────┐ ┌───────┐ ┌─────┐ ┌──────────┐  │ │
│  │  │Coordinator│→│ Think │→│ Act │→│ Evaluate │  │ │
│  │  └───────────┘ └───────┘ └─────┘ └──────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
│                        ↓                               │
│                    AI2-THOR                            │
└────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Single-Agent | Dual-Agent | Notes |
|------|----------|----------|------|
| State | `AgentState` | `DualAgentState` | Dual-agent state contains two `AgentState` objects |
| Graph | 3 nodes | 4 nodes | Adds Coordinator |
| Prompt | Single prompt | Collaboration prompt | Includes communication instructions |
| Evaluation | Single trajectory | Joint trajectory | Aggregates both agents' behavior |

## 2. State Design

### 2.1 `DualAgentState`

```python
class DualAgentState(TypedDict, total=False):
    task_prompt: str
    subtasks: Dict[str, str]

    agent_1: AgentState
    agent_2: AgentState
    current_agent: str

    env: Any
    vlm: Any

    shared_memory: Dict
    communication_history: List
    message_queue: List

    global_step_count: int
    max_global_steps: int
    global_success: bool
    global_fail_reason: str

    collaboration_mode: str
    turn_count: int
    current_turn_steps: int
```

### 2.2 `AgentState`

```python
class AgentState(TypedDict, total=False):
    agent_id: str
    agent_role: str

    observation: EnvObservation

    short_term_history: List
    long_term_summary: str
    structured_trajectory: List

    step_count: int
    max_steps: int
    should_continue: bool
    next_action: dict

    task_done_by_model: bool
    task_fail_by_model: bool
```

## 3. Graph Nodes

### 3.1 Node Flow

```
┌─────────────┐
│ Coordinator │ ← entry
└──────┬──────┘
       ↓
┌─────────────┐
│    Think    │
└──────┬──────┘
       ↓
┌─────────────┐
│     Act     │
└──────┬──────┘
       ↓
┌─────────────┐
│  Evaluate   │
└──────┬──────┘
       ↓
   ┌───┴───┐
   │ branch│
   └───┬───┘
       ├─────────→ Coordinator (continue)
       └─────────→ Final (end)
```

### 3.2 Coordinator Node

Decides which agent acts next.

```python
def coordinator_node(state: DualAgentState) -> DualAgentState:
  if collaboration_mode == "alternating":
    if current_turn_steps >= switch_interval:
      state["current_agent"] = other_agent_id
      state["current_turn_steps"] = 0
      state["turn_count"] += 1
  return state
```

### 3.3 Think Node

Calls the VLM and parses `<THINK>`, `<ACTION>`, and `<COMMUNICATE>`.

```python
def think_node(state: DualAgentState) -> DualAgentState:
  shared_context = build_shared_context(state)  # communication history only
  if communicate_block:
    state["message_queue"].append({...})
    state["communication_history"].append(...)
  return state
```

### 3.4 Act Node

Executes the selected action and updates observations.

```python
def act_node(state: DualAgentState) -> DualAgentState:
  observation, error = env.step_with_action_dict(action_dict)
  agent_state["observation"] = observation
  state[other_agent_id]["observation"] = observation
  return state
```

### 3.5 Evaluate Node

Checks terminal conditions and decides whether to continue.

```python
def evaluate_node(state: DualAgentState) -> DualAgentState:
  if task_done_by_model:
    success, score = perform_final_evaluation(state)
    if success:
      state["global_success"] = True
    else:
      state["current_agent"] = other_agent_id
  return state
```

## 4. Communication

### 4.1 Message Flow

```
Agent 1                    Message Queue                Agent 2
   │                            │                          │
   │ <COMMUNICATE>              │                          │
   │ "I found a book..."        │                          │
   │ ─────────────────────────►│                          │
   │                            │    next Think step       │
   │                            │ ─────────────────────────►
   │                            │    message delivered     │
```

### 4.2 Data Structures

```python
{
  "sender": "agent_1",
  "receiver": "agent_2",
  "message": "I found a book on the desk...",
  "step": 5
}

shared_memory = {
  "key_findings": [
    "[agent_1] I found a book on the desk...",
    "[agent_2] Got it, I'll open it"
  ],
  "discovered_objects": {},
}
```

### 4.3 Information Isolation

Agents cannot directly access each other's observations.

```python
# Correct: share only through communication
shared_context = build_shared_context(state)

# Forbidden: auto-sharing observations
# state["shared_memory"]["discovered_objects"][obj] = position
```

## 5. Collaboration Modes

### 5.1 Alternating

```
Step 1   Step 2   Step 3   Step 4
Agent1   Agent2   Agent1   Agent2
switch_interval = 1
```

### 5.2 Sequential

```
Step 1 ... Step N   Step N+1 ... Step M
│    Agent 1     │  │     Agent 2      │
│  exploration   │  │    execution     │
```

## 6. Compatibility with Single-Agent Code

| Component | Reuse |
|------|----------|
| `envs/ai2thor/wrapper.py` | Shared environment |
| `evaluators/base.py` | Evaluator |
| `config/load_config.py` | Config loader |
| `core/llm/schemas.py` | Schemas |
| `tasks/` | Task definitions |

Extended components:

| Component | Purpose |
|------|------|
| `dual_agent/core/agent/graph.py` | 4-node state machine |
| `dual_agent/core/agent/state.py` | Dual-agent state |
| `dual_agent/core/prompts/dual_agent.py` | Collaboration prompts |
| `dual_agent/config.yaml` | Dual-agent config |

## 7. Extension Guide

### 7.1 Add a Collaboration Mode

1. Extend `coordinator_node`
2. Add a config option
3. Update docs

### 7.2 Change Communication

1. Update parsing in `think_node`
2. Update `build_shared_context`
3. Update prompt templates
