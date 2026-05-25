# 本地运行指南

## 快速开始

### 1. 拉取代码

```bash
git clone https://github.com/1667652488-commits/agent-evaluator-v5-260524.git
cd agent-evaluator-v5-260524
```

### 2. 安装依赖

```bash
pip install openai
# 如需要 Web 工具支持
# pip install playwright && playwright install chromium
```

### 3. 配置 API Key

```bash
export SILICONFLOW_API_KEY="sk-nnksashvwdizsenvqlnlcyhevvzpqntwswvutxcqukzfhkyc"
```

### 4. 运行示例

```bash
# 示例 1: ACEBench Agent 单条演示（展示完整流程）
python examples/run_acebench_agent_demo.py

# 示例 2: ACEBench 飞轮（分阶段执行）

```bash
# Phase 1: 冷启动（5% 数据跑 Agent + 评估）
python pipeline/scripts/main.py acebench --phase coldstart --api-key sk-xxx

# Phase 2: 基于冷启动结果生成评估器 + 优化器
python pipeline/scripts/main.py acebench --phase evalopt

# Phase 3: 逐轮迭代优化（10% × N 轮）
python pipeline/scripts/main.py acebench --phase iteration --rounds 10

# Phase 4: 汇总所有轮次结果生成报告
python pipeline/scripts/main.py acebench --phase summary

# 全部一起跑（默认，等价于 --phase all）
python pipeline/scripts/main.py acebench --api-key sk-xxx
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--phase` | 执行阶段: all/coldstart/evalopt/iteration/summary | all |
| `--ratio` | 冷启动抽样比例 | 0.05 |
| `--rounds` | 迭代轮数 | 10 |
| `--iter-ratio` | 每轮迭代抽样比例 | 0.10 |
| `--model` | 模型名称 | Qwen/Qwen2.5-14B-Instruct |
| `--api-key` | SiliconFlow API Key | 从环境变量读取 |
| `--output-dir` | 结果输出目录 | ./results/no_gepa |

**阶段间依赖自动检查：**
- `evalopt` 会自动查找 `results/no_gepa/phase1_coldstart/results.json`
- `iteration` 会自动查找 `results/no_gepa/phase2_optimizer/optimized_agent_prompt.txt`
- `summary` 会自动收集所有 `round_X/summary.json`

### 示例 3: 旧版命令（数据预处理 / AgentBoard）

```bash
# Phase 1 数据预处理
python pipeline/scripts/main.py phase1 --data-dir ./agentboard_data

# 运行 Agent 生成轨迹
python pipeline/scripts/main.py run --input data.jsonl --output traces.jsonl --model siliconflow --api-key sk-xxx

# Rule 评估
python pipeline/scripts/main.py rule --input traces.jsonl --output scored.jsonl

# LLM 评估
python pipeline/scripts/main.py llm --input traces.jsonl --output llm_scores.jsonl --api-key sk-xxx

# Hybrid 融合
python pipeline/scripts/main.py hybrid --rule-input scored.jsonl --llm-input llm_scores.jsonl --output hybrid.jsonl

# GEPA 优化
python pipeline/scripts/main.py optimize --input hybrid.jsonl --output optimized_prompt.json
```

---

## 运行 Agent 示例详解

运行 `python examples/run_acebench_agent_demo.py` 后，你会看到以下完整流程：

### 步骤 1: 加载数据

```
加载任务数: 20
Ground truth 数: 20
示例任务 ID: data_zh_agent_multi_step_1
```

### 步骤 2: 输入数据

**用户请求** (question)：
```
查询附近的商家，为用户推荐一家餐厅并下单
```

**初始状态** (initial_config)：
```json
{
  "BaseApi": {
    "wifi": false,
    "logged_in": false
  },
  "FoodPlatform": {
    "users": {
      "Eve": {
        "user_id": "000001",
        "password": "123456",
        "balance": 200.0
      }
    },
    "logged_in_users": [],
    "orders": []
  }
}
```

**可用工具** (function)：
```
1. turn_on_wifi: 开启WiFi
2. login_device: 登录设备
3. login_food_platform: 登录外卖平台
   - username: string (required)
   - password: string (required)
4. check_balance: 查询余额
5. add_food_delivery_order: 添加外卖订单
6. get_products: 获取商家菜单
```

### 步骤 3: 初始化沙箱

沙箱基于 `initial_config` 创建本地状态副本：
```
沙箱已初始化，当前状态:
{
  "BaseApi": {"wifi": false, "logged_in": false},
  "FoodPlatform": {
    "users": {"Eve": {"balance": 200.0, "password": "123456"}},
    "logged_in_users": [],
    "orders": []
  }
}
```

### 步骤 4: LLM 决策 + 沙箱执行

**执行计划**（共 3 步）：

```
Step 1: 调用 login_food_platform
  参数: {"username": "Eve", "password": "123456"}
  返回: "用户 Eve 登录成功"

Step 2: 调用 add_food_delivery_order
  参数: {"username": "Eve", "merchant_name": "达美乐", "items": [{"product": "超级至尊披萨", "quantity": 1}]}
  返回: "订单已提交: 达美乐，总价 88.0 元"

Step 3: 调用 add_reminder
  参数: {"title": "今日花费", "description": "今日花费88.0元", "time": "2024-07-15 09:30"}
  返回: "提醒已添加: 今日花费"
```

每步执行后沙箱状态实时更新（如余额扣减、订单增加）。

### 步骤 5: 对比 Ground Truth

**Ground Truth 理想最终状态** (possible_answer)：
```json
{
  "FoodPlatform": {
    "logged_in_users": ["Eve"],
    "users": {
      "Eve": {
        "user_id": "000001",
        "balance": 112.0,
        "password": "123456"
      }
    },
    "orders": [{
      "user_name": "Eve",
      "merchant_name": "达美乐",
      "items": [{"product": "超级至尊披萨", "quantity": 1}],
      "total_price": 88.0
    }]
  },
  "ReminderApi": {
    "reminder_list": {
      "1": {
        "reminder_id": 1,
        "title": "今日花费",
        "description": "今日花费88.0元",
        "time": "2024-07-15 09:30",
        "notified": false
      }
    }
  }
}
```

**模型实际达到的最终状态**：
```json
{同上，如果匹配则显示一致的状态}
```

**理想执行路径** (mile_stone)：
```
Step 1: [login_food_platform(username='Eve', password='123456')]
Step 2: [check_balance(user_name='Eve')]
Step 3: [get_products(merchant_name='达美乐')]
Step 4: [add_food_delivery_order(username='Eve', ...)]
Step 5: [add_reminder(title='今日花费', ...)]
```

**模型实际执行路径**：
```
Step 1: login_food_platform({"username": "Eve", "password": "123456"})
Step 2: add_food_delivery_order({"username": "Eve", ...})
Step 3: add_reminder({"title": "今日花费", ...})
```

### 步骤 6: 评估结论

```
评估结果: ✅ 通过
评估说明: 最终状态与 Ground Truth 完全匹配。

路径评估:
  理想步数: 5
  实际步数: 3
  ⚠️ 步数不匹配（可能遗漏或多执行）
  
不匹配详情:
  - .FoodPlatform.orders[0].total_price: 值不匹配 (实际 88.0 vs 期望 88.0)
```

---

## 目录说明

```
agent-evaluator-v5-260524/
├── agent/
│   ├── models/              # Agent 执行器
│   └── sandbox/
│       └── acebench/        # ACEBench 沙箱模拟器（本地运行，无外部依赖）
│           ├── multi_step/  # 多步执行场景
│           └── multi_turn/  # 多轮对话场景
├── evaluator/
│   └── dev/
│       └── scripts/         # 8维评估器 + ACEBench 评估器
├── optimizer/
│   └── dev/
│       └── scripts/         # 优化器（基础版 + GEPA）
├── pipeline/
│   └── scripts/             # 飞轮主控
├── data_preprocessing/
│   └── raw_datasets/
│       ├── ACEBench/        # 全量数据（1017条 + possible_answer）
│       ├── AgentBoard/      # AgentBoard 数据
│       ├── SuperCLUE/       # SuperCLUE 数据
│       └── GeneralFunctionCall-Test/  # 2000条单步数据
├── examples/
│   └── run_acebench_agent_demo.py  # ⭐ 运行示例脚本
├── config.json              # 全局配置（GEPA 默认关闭）
└── README.md                # 本文件
```

---

## 配置说明

编辑 `config.json`：

```json
{
  "siliconflow": {
    "api_key": "sk-xxx",
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "base_url": "https://api.siliconflow.cn/v1"
  },
  "gepa": {
    "enabled": false    // 默认关闭，先跑基础优化飞轮
  },
  "evaluation": {
    "dataset": "ACEBench",
    "cold_start_ratio": 0.05,
    "iteration_ratio": 0.10
  }
}
```

---

## 常见问题

**Q: 需要 Docker 吗？**  
不需要。沙箱是纯 Python 本地代码，无容器、无外网请求。

**Q: 需要真实 API 吗？**  
沙箱工具返回预置值，无需真实外卖/短信/旅行 API。仅 LLM 调用需要 SiliconFlow API Key。

**Q: Agent 子集和 Normal/Special 子集的区别？**  
- Agent: 多步执行（模型自主规划工具调用序列），需要沙箱模拟器
- Normal/Special: 单步判断（模型只输出一次工具调用），直接对比参数即可

**Q: 运行示例时报错 ModuleNotFoundError？**  
确保 `examples/run_acebench_agent_demo.py` 在项目根目录下运行，或添加项目根目录到 `PYTHONPATH`。
