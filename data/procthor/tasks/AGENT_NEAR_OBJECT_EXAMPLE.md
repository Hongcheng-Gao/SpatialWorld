# 智能体靠近物体判定配置指南

## 基本格式

在 `task.json` 的 `success_conditions` 中添加 `agent_near_object` 条件：

```json
{
  "type": "agent_near_object",
  "object_type": "Bed",
  "distance": 1.0
}
```

## 参数说明

- **`type`**: 必须为 `"agent_near_object"`
- **`object_type`**: 目标物体类型（如 `"Bed"`, `"Fridge"`, `"Sofa"` 等）
- **`distance`**: 最大距离（米），默认值为 `1.0`（如果未指定）

## 完整示例

### 示例 1：移动到床附近（1米内）

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

### 示例 2：移动到冰箱附近（0.5米内，更严格）

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

### 示例 3：组合条件（移动到床附近 + 在卧室）

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

### 示例 4：移动到沙发附近（2米内，较宽松）

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

## 距离计算方式

- **2D距离**：只计算 X 和 Z 坐标的距离，忽略 Y 轴（高度）
- **公式**：`distance = sqrt((agent_x - object_x)² + (agent_z - object_z)²)`
- **单位**：米（meters）

## 支持的物体类型

支持所有 ProcTHOR 中的物体类型，例如：

- **家具**：`Bed`, `Sofa`, `Chair`, `Table`, `CounterTop`
- **电器**：`Fridge`, `Microwave`, `Stove`, `TV`
- **容器**：`Cabinet`, `Drawer`, `Sink`
- **其他**：任何在场景中存在的物体类型

**注意**：支持语义变体，例如 `"Apple"` 也会匹配 `"AppleSliced"`。

## 实现原理

1. 获取智能体的 (x, z) 位置
2. 在场景中查找所有匹配 `object_type` 的对象（支持语义变体）
3. 计算智能体到每个匹配对象的2D距离
4. 如果最短距离 ≤ `distance`，返回成功（1.0），否则返回失败（0.0）

## 代码位置

- **条件检查**：`evaluators/base.py` 第 445-490 行
- **距离计算**：`evaluators/metrics.py` 中的 `check_agent_near_object()` 函数
- **位置获取**：`evaluators/getters.py` 中的 `get_agent_position()` 函数

## 调试信息

当评估时，会输出详细的调试信息：

```
检查智能体到 Bed 的距离:
智能体位置: (6.75, 5.25)
最近对象: Bed|1|2|3, 距离: 0.85m, 阈值: 1.00m
✅ 智能体在 Bed 附近 (距离: 0.85m <= 1.00m)
```

## 与其他条件的组合

`agent_near_object` 可以与其他条件组合使用：

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
