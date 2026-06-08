# 双智能体 API 参考

## 核心函数

### `create_dual_agent_graph()`

创建双智能体状态机图。

```python
from .core.agent.graph import create_dual_agent_graph

app = create_dual_agent_graph()
final_state = app.invoke(initial_state)
```

**返回值**: `StateGraph` 编译后的状态机实例

---

### `initialize_dual_agent_state()`

初始化双智能体状态。

```python
from .core.agent.state import initialize_dual_agent_state

state = initialize_dual_agent_state(
    task_prompt="打开冰箱",
    env=env,
    vlm=vlm,
    config=config_dict,
    max_global_steps=60,
    collaboration_mode="alternating",
    run_output_dir="outputs/task_xxx"
)
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `task_prompt` | `str` | 任务描述 |
| `env` | `Any` | 环境实例 |
| `vlm` | `BaseChatModel` | VLM 模型实例 |
| `config` | `Dict` | 配置字典 |
| `max_global_steps` | `int` | 最大总步数 |
| `collaboration_mode` | `str` | 协作模式 |
| `run_output_dir` | `str` | 输出目录 |

**返回值**: `DualAgentState` 初始化后的状态

---

### `parse_action_string()`

解析动作字符串为标准动作字典。

```python
from .core.agent.graph import parse_action_string

action = parse_action_string("OpenObject(Fridge)")
# 返回: {"action_type": "interaction", "action_name": "OpenObject", "object_type": "Fridge"}

action = parse_action_string("MoveAhead")
# 返回: {"action_type": "navigation", "action_name": "MoveAhead"}

action = parse_action_string("DONE")
# 返回: {"action_type": "task_completion", "action_name": "DONE"}

action = parse_action_string("Pass()")
# 返回: {"action_type": "pass", "action_name": "Pass"}
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `action_string` | `str` | 动作字符串 |

**返回值**: `dict` 标准化的动作字典

**支持的动作类型**:
- 导航动作: `MoveAhead`, `MoveBack`, `MoveLeft`, `MoveRight`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`
- 交互动作: `PickupObject(X)`, `OpenObject(X)`, `CloseObject(X)`, `ToggleObjectOn(X)`, `ToggleObjectOff(X)`, ...
- 任务完成: `DONE`, `FAIL`
- 跳过动作: `Pass()`

---

### `get_vlm()`

获取 VLM 模型实例。

```python
from .core.llm import get_vlm

vlm = get_vlm(
    model_name="gpt-4o",
    temperature=0.2,
    max_tokens=2000
)
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | `None` | 模型名称 |
| `temperature` | `float` | `0.7` | 采样温度 |
| `max_tokens` | `int` | `2000` | 最大 token 数 |
| `agent_config` | `dict` | `None` | 智能体特定配置 |

**返回值**: `BaseChatModel` VLM 实例

---

### `build_shared_context()`

构建共享上下文（仅包含通信历史）。

```python
from .core.agent.state import build_shared_context

context = build_shared_context(state)
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `state` | `DualAgentState` | 双智能体状态 |

**返回值**: `str` 格式化的共享上下文字符串

---

## 状态类型

### `DualAgentState`

双智能体协作状态的完整类型定义。

```python
class DualAgentState(TypedDict, total=False):
    # 任务相关
    task_prompt: str
    subtasks: Dict[str, str]
    
    # 智能体状态
    agent_1: AgentState
    agent_2: AgentState
    current_agent: str  # "agent_1" | "agent_2"
    
    # 共享环境
    env: Any
    vlm: Any
    
    # 通信
    shared_memory: Dict[str, Any]
    communication_history: List[Dict[str, Any]]
    message_queue: List[Dict[str, Any]]
    
    # 全局控制
    global_step_count: int
    max_global_steps: int
    global_success: bool
    global_fail_reason: Optional[str]
    
    # 协作配置
    collaboration_mode: str  # "alternating" | "sequential"
    turn_count: int
    current_turn_steps: int
    force_agent_switch: bool
    
    # 输出
    config: Optional[Dict[str, Any]]
    run_output_dir: Optional[str]
```

### `AgentState`

单个智能体的状态类型定义。

```python
class AgentState(TypedDict, total=False):
    # 身份
    agent_id: str
    agent_role: str
    task_prompt: str
    
    # 观察
    observation: Optional[EnvObservation]
    
    # 执行控制
    step_count: int
    max_steps: int
    
    # 任务状态
    success: bool
    fail_reason: Optional[str]
    
    # 记忆
    short_term_history: List[Dict[str, Any]]
    long_term_summary: str
    structured_trajectory: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    
    # 节点间通信
    next_action: Optional[dict]
    should_continue: bool
    task_done_by_model: bool
    task_fail_by_model: bool
    
    # 实例
    vlm: Any
    env: Any
    config: Optional[Dict[str, Any]]
```

---

## Prompt 模板

### `get_collaborative_system_prompt()`

获取平等协作模式的系统 Prompt。

```python
from .dual_agent import get_collaborative_system_prompt

prompt = get_collaborative_system_prompt(enable_summary=False)
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_summary` | `bool` | `False` | 是否启用长期摘要 |

**返回值**: `str` 系统 Prompt 模板

**模板变量**:
- `{task_prompt}`: 任务描述
- `{shared_context}`: 共享上下文（通信历史）

---

## 评估器

使用主目录的评估器（`evaluators/base.py`）：

### `create_evaluator_from_config()`

从任务配置创建评估器。

```python
from evaluators.base import create_evaluator_from_config

evaluator = create_evaluator_from_config(task_config)
score = evaluator.evaluate(env, metadata)
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `task_config` | `dict` | 任务配置字典 |

**返回值**: `Evaluator` 评估器实例

---

## 工具函数

### `load_task_info()`

加载任务信息。

```python
from mllm_base_agent.dual_agent.ai2thor.main import load_task_info

task_data = load_task_info("ai2thor05002")
# 返回:
# {
#     "task_id": "ai2thor05002",
#     "instruction": "打开书并关闭台灯",
#     "scene": "FloorPlan302",
#     "init_actions": ["MoveAhead", "RotateRight"],
#     "task_info": {...}
# }
```

---

### `execute_init_actions()`

执行初始化动作序列。

```python
from mllm_base_agent.dual_agent.ai2thor.main import execute_init_actions

init_count = execute_init_actions(env, init_actions)
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `env` | `Any` | 环境实例 |
| `init_actions` | `List[str]` | 初始化动作列表 |

**返回值**: `int` 执行的动作数量
