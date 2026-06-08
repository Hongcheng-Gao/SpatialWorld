# 任务配置示例：将泰迪熊从卧室移到浴室

## 任务描述

将泰迪熊（TeddyBear）从卧室（Bedroom）的床上移动到浴室（Bathroom）。

## 成功条件设置

### 方法 1：只检查目标房间（推荐）

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

**说明**：
- 只要泰迪熊在浴室（Bathroom）中，任务就成功
- 不需要检查是否不在卧室，因为只要在浴室就满足了要求

### 方法 2：组合条件（更严格）

如果需要确保泰迪熊不在卧室，可以添加反向条件：

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

**说明**：
- 第一个条件：泰迪熊必须在浴室
- 第二个条件：泰迪熊必须被放下（不在手中）

## 代码识别能力

✅ **代码已支持 `object_in_room` 条件**

### 实现方式

代码使用 **floorPolygon（精确多边形）** 来判断对象所在的房间：

1. **优先使用 floorPolygon**：
   - 获取对象的 (x, z) 位置
   - 使用 `_point_in_polygon()` 函数检查对象是否在某个房间的多边形内
   - 这是最准确的方法，直接来自 ProcTHOR 的场景数据

2. **回退方案**：
   - 如果 floorPolygon 不可用，会使用对象的 `roomType` 属性（如果存在）

### 代码位置

- **条件检查**：`evaluators/base.py` 第 390-419 行
- **房间判断**：`evaluators/getters.py` 中的 `_build_room_boundaries_from_house_scene()` 和 `_point_in_polygon()`

## 使用示例

### 运行任务评估

```bash
python scripts/evaluate_action_sequence.py \
  --task-config tasks/procthor00003/task.json \
  --action-sequence actions.json
```

### 在代码中使用

```python
from envs.procthor_wrapper import ProcTHOREnvWrapper
import json

# 加载任务配置
with open('tasks/procthor00003/task.json', 'r') as f:
    task_config = json.load(f)

# 创建环境
env = ProcTHOREnvWrapper(
    scene_index=task_config['scene_index'],
    config={'task': task_config}
)

# 重置环境
observation = env.reset(task_config['instruction'])

# 执行动作序列...
# ...

# 评估任务
from evaluators.base import create_evaluator_from_config
evaluator = create_evaluator_from_config(task_config)
score = evaluator.evaluate(env, env.controller.last_event.metadata)
print(f"任务得分: {score}")
```

## 注意事项

1. **房间名称大小写**：代码支持大小写不敏感匹配，`"Bathroom"` 和 `"bathroom"` 都可以

2. **对象类型匹配**：支持语义变体，例如 `"Apple"` 也会匹配 `"AppleSliced"`

3. **多个对象**：如果有多个相同类型的对象，只要有一个满足条件就会返回成功

4. **对象位置**：使用对象的 `position` 字段中的 `x` 和 `z` 坐标来判断房间

## 其他可用的成功条件类型

- `object_state`: 检查对象状态（如 `isOpen`, `isPickedUp` 等）
- `object_in_hand`: 检查对象是否在手中
- `object_in_receptacle`: 检查对象是否在容器中
- `agent_in_room`: 检查智能体是否在指定房间
