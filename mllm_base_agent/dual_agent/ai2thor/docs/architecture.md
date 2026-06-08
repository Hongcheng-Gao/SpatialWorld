# 双智能体系统架构设计

## 1. 整体架构

双智能体系统基于单智能体框架扩展，核心思想是**两个平等的智能体通过显式通信协作完成任务**。

### 1.1 架构对比

```
单智能体架构:
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

双智能体架构:
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

### 1.2 核心组件

| 组件 | 单智能体 | 双智能体 | 说明 |
|------|----------|----------|------|
| 状态 | `AgentState` | `DualAgentState` | 双智能体包含两个 AgentState |
| 状态机 | 3 节点 | 4 节点 | 新增 Coordinator 节点 |
| Prompt | 单一 | 协作导向 | 包含通信指令 |
| 评估 | 单一轨迹 | 联合轨迹 | 综合两个智能体的表现 |

## 2. 状态设计

### 2.1 DualAgentState

```python
class DualAgentState(TypedDict, total=False):
    # ===== 任务相关 =====
    task_prompt: str              # 整体任务描述
    subtasks: Dict[str, str]      # 各智能体的子任务描述
    
    # ===== 智能体状态 =====
    agent_1: AgentState           # 智能体1的完整状态
    agent_2: AgentState           # 智能体2的完整状态
    current_agent: str            # 当前活跃的智能体 ID
    
    # ===== 共享环境 =====
    env: Any                      # 共享的环境实例
    vlm: Any                      # 共享的 VLM 模型
    
    # ===== 通信通道 =====
    shared_memory: Dict           # 共享记忆（仅存储通信内容）
    communication_history: List   # 通信历史记录
    message_queue: List           # 待处理的消息队列
    
    # ===== 全局控制 =====
    global_step_count: int        # 全局步数计数
    max_global_steps: int         # 最大步数限制
    global_success: bool          # 全局任务成功标志
    global_fail_reason: str       # 失败原因
    
    # ===== 协作配置 =====
    collaboration_mode: str       # 协作模式
    turn_count: int               # 智能体切换次数
    current_turn_steps: int       # 当前轮已执行步数
```

### 2.2 AgentState（单个智能体）

```python
class AgentState(TypedDict, total=False):
    # 身份
    agent_id: str                 # "agent_1" 或 "agent_2"
    agent_role: str               # 角色描述
    
    # 观察
    observation: EnvObservation   # 当前观察
    
    # 记忆
    short_term_history: List      # 短期历史（滑动窗口）
    long_term_summary: str        # 长期语义摘要
    structured_trajectory: List   # 结构化轨迹
    
    # 控制
    step_count: int               # 步数计数
    max_steps: int                # 单个智能体最大步数
    should_continue: bool         # 是否继续
    next_action: dict             # 下一个动作
    
    # 任务状态
    task_done_by_model: bool      # 模型输出 DONE
    task_fail_by_model: bool      # 模型输出 FAIL
```

## 3. 状态机节点

### 3.1 节点流程

```
┌─────────────┐
│ Coordinator │ ← 入口点
└──────┬──────┘
       ↓
┌─────────────┐
│    Think    │ ← VLM 推理
└──────┬──────┘
       ↓
┌─────────────┐
│     Act     │ ← 执行动作
└──────┬──────┘
       ↓
┌─────────────┐
│  Evaluate   │ ← 评估状态
└──────┬──────┘
       ↓
   ┌───┴───┐
   │ 条件  │
   └───┬───┘
       ├─────────→ Coordinator（继续）
       └─────────→ Final（结束）
```

### 3.2 Coordinator 节点

**职责**：决定哪个智能体执行下一步动作

```python
def coordinator_node(state: DualAgentState) -> DualAgentState:
    """
    协调逻辑:
    1. 检查当前智能体是否被阻止
    2. 检查是否需要切换（基于 switch_interval）
    3. 检查消息队列
    4. 更新 current_agent
    """
    
    # 交替模式下的切换逻辑
    if collaboration_mode == "alternating":
        if current_turn_steps >= switch_interval:
            # 切换到另一个智能体
            state["current_agent"] = other_agent_id
            state["current_turn_steps"] = 0
            state["turn_count"] += 1
    
    return state
```

### 3.3 Think 节点

**职责**：调用 VLM 进行推理，解析动作和通信

```python
def think_node(state: DualAgentState) -> DualAgentState:
    """
    思考流程:
    1. 构建上下文（任务 + 通信历史 + 短期记忆）
    2. 编码当前图像
    3. 构建多轮对话消息
    4. 调用 VLM
    5. 解析 <THINK>, <ACTION>, <COMMUNICATE> 标签
    6. 处理通信消息（加入队列）
    """
    
    # 关键：只通过通信获取伙伴信息
    shared_context = build_shared_context(state)  # 仅包含通信历史
    
    # 解析通信
    if communicate_block:
        state["message_queue"].append({
            "sender": current_agent_id,
            "receiver": other_agent_id,
            "message": communication_message,
        })
        state["communication_history"].append(...)
    
    return state
```

### 3.4 Act 节点

**职责**：执行动作，更新观察

```python
def act_node(state: DualAgentState) -> DualAgentState:
    """
    执行流程:
    1. 获取当前智能体的 next_action
    2. 处理特殊动作（DONE, FAIL, Pass）
    3. 调用 env.step_with_action_dict()
    4. 更新双方智能体的 observation（共享环境）
    5. 更新步数计数器
    """
    
    # 执行动作
    observation, error = env.step_with_action_dict(action_dict)
    
    # 更新双方观察（共享环境）
    agent_state["observation"] = observation
    state[other_agent_id]["observation"] = observation
    
    # 注意：不自动共享发现的物体！
    # 智能体需要通过 COMMUNICATE 告诉对方
    
    return state
```

### 3.5 Evaluate 节点

**职责**：评估任务状态，决定是否继续

```python
def evaluate_node(state: DualAgentState) -> DualAgentState:
    """
    评估情况:
    1. 智能体声称 DONE → 调用 evaluator 验证
       - 成功 → 设置 global_success = True
       - 失败 → 切换到另一个智能体
    2. 智能体声称 FAIL → 尝试切换
    3. 达到步数上限 → 结束
    4. 其他 → 继续
    """
    
    if task_done_by_model:
        success, score = perform_final_evaluation(state)
        if success:
            state["global_success"] = True
        else:
            # DONE 验证失败，切换到另一个智能体
            state["current_agent"] = other_agent_id
    
    return state
```

## 4. 通信机制

### 4.1 通信流程

```
Agent 1                    Message Queue                Agent 2
   │                            │                          │
   │ <COMMUNICATE>              │                          │
   │ "我发现了书..."           │                          │
   │ ─────────────────────────►│                          │
   │                            │                          │
   │                            │    下一轮 Think 节点     │
   │                            │ ─────────────────────────►
   │                            │    收到消息              │
   │                            │                          │
```

### 4.2 数据结构

```python
# 通信消息
{
    "sender": "agent_1",
    "receiver": "agent_2",
    "message": "我在书桌上发现了一本书...",
    "step": 5
}

# 共享记忆（仅存储通信产生的信息）
shared_memory = {
    "key_findings": [
        "[agent_1] 我在书桌上发现了一本书...",
        "[agent_2] 收到，我来打开它"
    ],
    "discovered_objects": {},  # 现在为空，不自动填充
}
```

### 4.3 信息隔离原则

**核心设计**：智能体无法直接访问对方的观察results

```python
# ✅ 正确：通过通信共享
def think_node(state):
    # 只获取通信历史作为共享上下文
    shared_context = build_shared_context(state)  # 仅包含 communication_history
    
    # Prompt 中明确说明
    # "You can ONLY learn about your partner through their MESSAGES"

# ❌ 禁止：自动共享观察
def act_node(state):
    # 以下代码已被移除
    # state["shared_memory"]["discovered_objects"][obj] = position
```

## 5. 协作模式

### 5.1 交替模式 (Alternating)

```
时间轴:
  Step 1   Step 2   Step 3   Step 4
  Agent1   Agent2   Agent1   Agent2
  ├────┤├────┤├────┤├────┤
   T1    T2    T3    T4
       
switch_interval = 1
```

### 5.2 顺序模式 (Sequential)

```
时间轴:
  Step 1 ... Step N   Step N+1 ... Step M
  │    Agent 1     │  │     Agent 2      │
  │   (探索阶段)    │  │    (执行阶段)    │
  └────────────────┘  └─────────────────┘
```

## 6. 与单智能体的兼容性

双智能体系统复用了大量单智能体组件：

| 组件 | 复用方式 |
|------|----------|
| `envs/ai2thor/wrapper.py` | 直接使用（共享环境） |
| `evaluators/base.py` | 直接使用（评估器） |
| `config/load_config.py` | 直接使用（配置加载） |
| `core/llm/schemas.py` | 直接使用（数据结构） |
| `tasks/` | 直接使用（任务定义） |

扩展的组件：
| 组件 | 说明 |
|------|------|
| `dual_agent/core/agent/graph.py` | 新的状态机（4节点） |
| `dual_agent/core/agent/state.py` | 新的状态定义 |
| `dual_agent/core/prompts/dual_agent.py` | 协作 Prompt |
| `dual_agent/config.yaml` | 双智能体配置 |

## 7. 扩展指南

### 7.1 添加新的协作模式

1. 在 `coordinator_node` 中添加新的协作逻辑
2. 在配置文件中添加对应选项
3. 更新文档

### 7.2 修改通信机制

1. 修改 `think_node` 中的通信解析
2. 修改 `build_shared_context` 函数
3. 更新 Prompt 模板
