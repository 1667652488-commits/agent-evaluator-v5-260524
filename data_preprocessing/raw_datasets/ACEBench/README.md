# ACEBench 数据集（中文子集）

> 来源：https://github.com/ACEBench/ACEBench  
> 论文：arXiv:2501.12851  
> 发布方：中国科学技术大学（USTC）+ 华为诺亚方舟实验室  
> 许可证：Apache License 2.0

## 简介

ACEBench 是 2025 年 1 月发布的大模型工具调用（Function Calling）评测基准，覆盖单轮、多轮、多步、异常指令等多种场景。

**本仓库仅包含中文子集（data_zh）**，共 **1017 条**评测数据，含完整的 ground truth（possible_answer）。

## 数据分布

| 子集 | 文件 | 数量 | 说明 |
|------|------|------|------|
| **Agent 多步** | `data_agent_multi_step.json` | 20 | 用户发一次任务，模型自主调用工具直到完成 |
| **Agent 多轮** | `data_agent_multi_turn.json` | 30 | 用户多轮参与对话，逐步提供信息或调整需求 |
| Normal-原子-bool | `data_normal_atom_bool.json` | 50 | 原子能力：布尔参数 |
| Normal-原子-enum | `data_normal_atom_enum.json` | 50 | 原子能力：枚举参数 |
| Normal-原子-list | `data_normal_atom_list.json` | 50 | 原子能力：列表参数 |
| Normal-原子-number | `data_normal_atom_number.json` | 50 | 原子能力：数值参数 |
| Normal-原子-object_deep | `data_normal_atom_object_deep.json` | 50 | 原子能力：深层对象参数 |
| Normal-原子-object_short | `data_normal_atom_object_short.json` | 50 | 原子能力：浅层对象参数 |
| Normal-多轮-adjust | `data_normal_multi_turn_user_adjust.json` | 116 | 用户中途调整需求 |
| Normal-多轮-switch | `data_normal_multi_turn_user_switch.json` | 101 | 用户切换意图 |
| Normal-偏好 | `data_normal_preference.json` | 50 | 模型需根据偏好选择 |
| Normal-相似API | `data_normal_similar_api.json` | 50 | 从相似API中选择正确的一个 |
| Normal-单轮-并行 | `data_normal_single_turn_parallel_function.json` | 100 | 单轮调用多个函数 |
| Normal-单轮-单函数 | `data_normal_single_turn_single_function.json` | 100 | 单轮调用单个函数 |
| Special-错误参数 | `data_special_error_param.json` | 50 | 测试参数错误处理能力 |
| Special-参数缺失 | `data_special_incomplete.json` | 50 | 测试必填参数识别 |
| Special-无关请求 | `data_special_irrelevant.json` | 50 | 测试无关请求过滤 |
| **总计** | 17 个文件 | **1017** | — |

## 数据格式

每行一个 JSON 对象，核心字段：

```json
{
  "id": "agent_multi_step_0",
  "question": "我是Eve，我需要在foodplatform中点一个达美乐的超级至尊披萨外卖...",
  "initial_config": {
    "BaseApi": {"wifi": true, "logged_in": true}
  },
  "path": [],
  "function": [
    {
      "name": "login_food_platform",
      "description": "使用用户名和密码登录外卖平台。",
      "parameters": {
        "type": "dict",
        "required": ["username", "password"],
        "properties": {
          "username": {"type": "string"},
          "password": {"type": "string"}
        }
      }
    }
  ],
  "involved_classes": ["BaseApi", "FoodPlatform"]
}
```

## Ground Truth

`possible_answer/` 目录下每个文件对应同名的标准答案，可用于：
- **过程准确度（process accuracy）**：对比执行过程与理想流程
- **端到端准确度（end-to-end accuracy）**：对比最终状态与目标状态

## 与飞轮的适配性

| 评估维度 | 适配度 | 说明 |
|---------|--------|------|
| goal_understanding | ✅ 高 | question 字段即目标描述 |
| planning | ✅ 高 | Agent 子集需要多步规划 |
| tool_selection | ✅ 高 | function 列表提供候选工具 |
| parameter_generation | ✅ 高 | parameters 定义了 JSON Schema |
| execution_accuracy | ✅ 高 | initial_config + possible_answer 可验证 |
| reflection | ⚠️ 中 | 部分场景隐含错误处理（如短信容量满） |
| state_tracking | ✅ 高 | 多步任务需跟踪中间状态 |
| termination | ✅ 高 | possible_answer 定义了终止状态 |

## 快速使用

```python
import json

# 加载 Agent 多步数据
with open('data_agent_multi_step.json') as f:
    data = [json.loads(line) for line in f if line.strip()]

# 加载对应 ground truth
with open('possible_answer/data_agent_multi_step.json') as f:
    answers = [json.loads(line) for line in f if line.strip()]

print(f"样本数: {len(data)}")
print(f"第一条目标: {data[0]['question'][:50]}...")
```

## Citation

```bibtex
@article{chen2025acebench,
  title={ACEBench: Who Wins the Match Point in Tool Learning?},
  author={Chen, Chen and Hao, Xinlong and Liu, Weiwen and Huang, Xu and Zeng, Xingshan and Yu, Shuai and Li, Dexun and Wang, Shuai and Gan, Weinan and Huang, Yuefeng and others},
  journal={arXiv preprint arXiv:2501.12851},
  year={2025}
}
```
