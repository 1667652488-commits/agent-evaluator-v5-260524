# ACEBench 沙箱模拟器

从 ACEBench 官方仓库提取的模拟工具代码，用于在本地执行 Agent 评测。

## 目录结构

```
acebench/
├── multi_step/              # 多步自主执行场景
│   ├── execution_role_step.py   # 模型输出解析 + 工具执行
│   ├── common_agent_step.py     # Agent 基类
│   ├── multi_step_scene.py      # 场景加载
│   ├── multi_step_utils.py      # 工具调用
│   ├── APIModel_agent.py        # API 模型 Agent
│   ├── phone_platform/          # 手机平台模拟类
│   │   ├── base_api.py          # WiFi / 设备登录
│   │   ├── food_services.py     # 外卖平台
│   │   ├── message.py           # 短信系统
│   │   └── reminder.py          # 提醒系统
│   └── scenarioszh/
│       └── travel.py            # 旅行预订模拟
│
└── multi_turn/              # 多轮对话场景
    ├── execution_role.py        # 模型输出解析 + 工具执行
    ├── common_agent.py          # Agent 基类
    ├── multi_turn_scene.py      # 场景加载
    ├── multi_turn_utils.py      # 工具调用
    ├── APIModel_agent.py        # API 模型 Agent
    ├── phone_platform/          # 手机平台模拟类（同 multi_step）
    │   ├── base_api.py
    │   ├── food_services.py
    │   ├── message.py
    │   └── reminder.py
    └── scenarioszh/
        └── travel.py            # 旅行预订模拟
```

## 模拟系统

| 系统 | 类名 | 文件 | 功能 |
|------|------|------|------|
| 基础设备 | BaseApi | base_api.py | WiFi 开关、设备登录 |
| 外卖平台 | FoodPlatform | food_services.py | 用户管理、商家菜单、下单扣款 |
| 短信系统 | MessageApi | message.py | 收发消息、容量管理、按ID删除 |
| 提醒系统 | ReminderApi | reminder.py | 创建/查询提醒 |
| 旅行预订 | Travel | travel.py | 航班查询、酒店预订 |

## 执行流程

1. 加载 `initial_config` 初始化沙箱状态
2. 模型输出 `[ApiName(key='value', ...)]` 格式
3. `execution_role_*.py` 解析调用 → 动态导入对应类
4. `eval()` 执行方法 → 修改沙箱状态
5. 返回 observation 给模型继续下一步

## 来源

- 原仓库：https://github.com/chenchen0103/ACEBench
- 提取路径：`model_inference/multi_step/` 和 `model_inference/multi_turn/`
