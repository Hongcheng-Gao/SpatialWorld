# 3D迷宫游戏MLLM框架接入指南

## 概述

本指南说明如何将现有的3D迷宫游戏（`maze3d_pygame.py`）接入MLLM测试框架，以便使用多模态大语言模型进行游戏测试和评估。

## 文件结构

```
games/maze3d/
├── maze3d_pygame.py          # 原始3D迷宫游戏代码
├── maze3d_adapter.py         # 游戏适配器类
├── maze3d_demo.py            # 完整演示脚本
├── run_maze_evaluation.py    # 运行评测的脚本
├── test_adapter.py           # 适配器测试脚本
└── README.md                 # 本指南
```

## 核心组件

### 1. Maze3DGame 适配器类

`Maze3DGame` 类实现了MLLM框架所需的接口：

- `init()`: 初始化游戏
- `update()`: 更新游戏状态
- `render()`: 渲染游戏画面
- `get_state()`: 获取游戏状态
- `reset()`: 重置游戏

### 2. AI模型函数

提供了多种AI模型用于测试：

- `random_maze_ai_model()`: 随机决策模型
- `heuristic_maze_ai_model()`: 启发式模型
- `intelligent_maze_ai_model()`: 智能模型

### 3. 回调函数

- `frame_callback()`: 帧数据回调
- `state_callback()`: 游戏状态回调
- `action_callback()`: 动作执行回调

## 使用方法

### 基本集成测试

```bash
# 在项目根目录下运行
conda run -n pygame python -c "
from games.maze3d.maze3d_adapter import Maze3DGame
from input_sources.pygame_input_source import PygameInputSource

game = Maze3DGame()
input_source = PygameInputSource()

# 初始化
success = input_source.initialize(game_module=game, headless=True, screen_size=(800, 600))
print(f'Initialization: {success}')

# 获取状态
state = input_source.get_game_state()
print(f'Game state: {state.normalized_state}')

# 清理
input_source.close()
"
```

### 运行完整评测

```bash
# 在项目根目录下运行
conda run -n pygame python games/maze3d/run_maze_evaluation.py
```

### 运行适配器测试

```bash
# 在maze3d目录下运行
conda run -n pygame python test_adapter.py
```

## 游戏状态信息

适配器提供以下游戏状态信息：

- **玩家信息**: 位置、方向
- **出口信息**: 位置
- **游戏状态**: 是否胜利、步数、最大步数
- **距离信息**: 到出口的曼哈顿距离
- **性能指标**: 动画时间、剩余步数

## 支持的动作

适配器支持以下标准动作：

- `key_press "w"`: 向前移动
- `key_press "a"`: 向左转
- `key_press "d"`: 向右转

## 配置选项

### 评测参数

- `max_steps`: 最大步数（默认300）
- `decision_frequency`: 决策频率（默认2.0 Hz）
- `headless`: 无头模式（默认True）

### 游戏参数

- `maze_file`: 迷宫文件路径（可选）
- `max_steps`: 游戏最大步数（默认1000）

## 日志和输出

评测results会保存到 `logs/` 目录，包含：

- 性能统计（FPS、推理时间等）
- 游戏results（是否成功、步数等）
- 详细的状态变化记录

## 扩展指南

### 添加新的AI模型

1. 创建新的AI模型函数
2. 在 `run_maze_evaluation.py` 中注册
3. 调整决策逻辑和推理时间

### 自定义游戏状态

修改 `Maze3DGame.get_state()` 方法以提供更多状态信息。

### 支持新动作

在适配器中添加新的动作处理方法，并在AI模型中调用。

## 故障排除

### 常见问题

1. **导入错误**: 确保路径设置正确
2. **Pygame初始化失败**: 检查无头模式设置
3. **游戏状态获取失败**: 验证适配器接口实现

### 调试建议

1. 使用 `log_level=logging.DEBUG` 获取详细日志
2. 在非无头模式下运行以观察游戏画面
3. 检查回调函数中的状态变化

## 性能优化

- 降低决策频率以减少计算负载
- 使用无头模式提高性能
- 优化AI模型推理时间
- 合理设置最大步数避免无限循环

## 结论

通过本适配器，3D迷宫游戏已成功接入MLLM测试框架，可以用于多模态大语言模型的游戏测试和评估。适配器提供了完整的接口实现、多种AI模型示例和详细的评测功能。