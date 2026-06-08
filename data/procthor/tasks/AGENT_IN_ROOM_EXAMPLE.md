# 智能体房间位置判定配置指南

## 基本格式

在 `task.json` 的 `success_conditions` 中添加 `agent_in_room` 条件：

```json
{
  "type": "agent_in_room",
  "room_type": "Kitchen"
}
```

## 与 object_in_room 的对比

### 物体在房间（需要指定 object_type）

```json
{
  "type": "object_in_room",
  "object_type": "TeddyBear",  // 必须指定物体类型
  "room_type": "Bedroom"
}
```

### 智能体在房间（不需要 object_type）

```json
{
  "type": "agent_in_room",
  "room_type": "Kitchen"  // 只需要指定房间类型
}
```

## 完整示例

### 示例 1：只检查智能体位置

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

### 示例 2：组合条件（智能体 + 物体）

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

### 示例 3：多个条件组合

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

## 支持的房间类型

- `Kitchen` - 厨房
- `Bathroom` - 浴室
- `Bedroom` - 卧室
- `LivingRoom` - 客厅
- `DiningRoom` - 餐厅
- 其他房间类型（根据场景而定）

**注意**：房间名称大小写不敏感，`"Kitchen"` 和 `"kitchen"` 都可以。

## 实现原理

`agent_in_room` 条件使用 **floorPolygon（精确多边形）** 来判断智能体所在的房间：

1. 获取智能体的 (x, z) 位置
2. 使用 `_point_in_polygon()` 函数检查智能体是否在某个房间的多边形内
3. 这是最准确的方法，直接来自 ProcTHOR 的场景数据

## 代码位置

- **条件检查**：`evaluators/base.py` 第 321-388 行
- **房间判断**：`evaluators/getters.py` 中的 `get_agent_room()` 函数
