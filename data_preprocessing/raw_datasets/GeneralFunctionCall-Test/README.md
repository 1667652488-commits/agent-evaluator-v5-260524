---
license: Apache License 2.0
---

# 通用函数调用测试数据集（GeneralFunctionCall-Test）

本数据集用于评估模型的 ToolCall 能力，聚焦两项指标：
- ToolCall-Trigger Similarity：是否应触发工具（由官方结果的K2-thinking模型结果对齐得到的 should_call_tool 标注）
- ToolCall-Schema Accuracy：当触发工具时，参数是否满足既定 JSON Schema（由评测工具计算）

数据集以 JSONL 形式提供，每行一个样本，包含对话上下文与工具列表，并给出是否应触发工具的标签，便于复现实验与自动化回归测试。

数据由官方 MoonshotAI [公开样本](https://github.com/MoonshotAI/K2-Vendor-Verifier)合成。

## 数据格式

每行一个 JSON 对象，示例结构：
```json
{
  "messages": [
    // OpenAI 格式消息，含 role 与 content
  ],
  "tools": [
    // OpenAI 工具定义，含 type:function 与 function:{name, parameters}
  ],
  "should_call_tool": true
}
```

字段说明：
- messages：评测输入上下文
- tools：可用函数（用于约束与评测 schema）
- should_call_tool：是否应触发工具的标签（来源于官方 K2-thinking模型结果中 `finish_reason == "tool_calls"`）

## 使用方法

安装 evalscope 工具并运行评测：

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    datasets=['general_fc'],  # 工具调用格式固定为 'general_fc'
    # 如需显式指定，可加入 dataset_args={"general_fc": {"dataset_id": "evalscope/GeneralFunctionCall-Test"}}
)
run_task(task_cfg=task_cfg)
```

具体请参考evalscope工具中的 [自定义模型工具调用评测文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)

输出示例：

```text
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| Model     | Dataset    | Metric                        | Subset   |   Num |   Score | Cat.0   |
+===========+============+===============================+==========+=======+=========+=========+
| qwen-plus | general_fc | count_finish_reason_tool_call | default  |    10 |  3      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | count_successful_tool_call    | default  |    10 |  2      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | schema_accuracy               | default  |    10 |  0.6667 | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | tool_call_f1                  | default  |    10 |  0.5    | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
```

指标说明：
- `count_finish_reason_tool_call`：模型预测调用工具的样本数（`finish_reason == "tool_calls"`）
- `count_successful_tool_call`：在尝试调用的样本中，工具名匹配且参数通过 JSON Schema 校验的样本数
- `schema_accuracy`：在尝试调用的样本中，参数校验通过的比例
- `tool_call_f1`：基于标签 `should_call_tool` 与模型是否调用工具的二分类 F1

## 许可

本数据集遵循 Apache License 2.0。

