# Agent Evaluator — Agent 数据飞轮评估器 (v5.1)

> 完整的 Agent 执行轨迹评估与自动优化系统。支持 **Rule / LLM / Hybrid** 三种评估模式，内置 **GEPA 遗传进化优化器**，可驱动 Agent 能力的持续提升。

---

## 目录结构 (v5.1)

```
agent-evaluator/
├── config.json                          # 全局配置（数据集 / API / 路径 / GEPA 开关）
├── path_utils.py                        # 路径适配工具（自动适配 Windows / Linux）
├── README.md                            # 项目说明
│
├── data_preprocessing/                  # ① 数据前处理
│   ├── raw_datasets/
│   │   ├── agentboard/                  # AgentBoard 原始数据（Tool-Query + Tool-Operation + WebArena）
│   │   ├── superclue/                   # SuperCLUE-Agent 数据集（待接入）
│   │   └── other/                       # 扩展数据集
│   ├── processed_data/
│   │   ├── cold_start/                  # 冷启动数据（20%）
│   │   └── full_datasets/               # 完整数据集（80%验证）
│   └── scripts/
│       └── phase1_preprocess.py         # 数据预处理脚本
│
├── agent/                               # ② 智能体 Agent
│   ├── models/
│   │   └── agent_runner.py              # Agent 执行引擎（Mock / 真实模型）
│   ├── skills/
│   │   └── (预留: 真实工具 Skill)
│   └── configs/
│
├── evaluator/                           # ③ 评估器
│   ├── dev/                             # 开发态：用冷启动数据校准
│   │   ├── skills/
│   │   │   └── gepa/
│   │   │       └── reflection_engine.py # GEPA 反射引擎：分析失败 → 弱点报告
│   │   ├── scripts/
│   │   │   ├── rule_grader.py           # 8维规则评估器
│   │   │   ├── llm_grader.py            # LLM-as-Judge
│   │   │   ├── hybrid_grader.py         # 融合评估器
│   │   │   ├── grader_aligner.py         # 评估器对齐分析
│   │   │   └── quick_rule_grade.py       # 快速评分入口
│   │   └── data/
│   │       ├── hybrid_strategy.json
│   │       └── tree_optimization_report.json
│   └── runtime/                         # 运行态：快速高效评估
│       ├── skills/
│       ├── scripts/
│       ├── reports/
│       └── configs/
│
├── optimizer/                           # ④ 优化器
│   ├── dev/                             # 开发态：规则库 / LLM prompt 调试
│   │   ├── skills/
│   │   │   └── gepa/
│   │   │       ├── prompt_candidate.py   # PromptCandidate 数据结构
│   │   │       ├── fitness_evaluator.py  # 适应度计算（8维评分）
│   │   │       ├── mutation_engine.py    # GEPA 变异引擎：生成 prompt 补丁
│   │   │       └── pareto_selector.py    # GEPA Pareto 选择器：双目标选优
│   │   ├── scripts/
│   │   │   └── gepa_optimizer.py        # GEPA 主循环（协调 Reflection → Mutation → Pareto）
│   │   └── data/
│   └── runtime/                         # 运行态：快速生成优化配置
│       ├── skills/
│       ├── scripts/
│       └── configs/
│
└── pipeline/                            # ⑤ 飞轮运行 Pipeline
    ├── scripts/
    │   └── main.py                        # 统一 CLI 入口
    ├── configs/
    ├── logs/
    └── reports/                           # 对比报告输出
```

---

## 架构

```
Phase 1: 数据预处理  →  Phase 2: Agent 执行  →  Phase 3: 评估打分
        data_preprocessing/        agent/            evaluator/runtime/
              ↓                        ↓                      ↓
        Phase 4: 对齐分析  →  Phase 5: GEPA 优化  →  重新注入 Agent (闭环)
       evaluator/dev/         optimizer/dev/
              ↓                        ↓
       冷启动数据校准            运行态部署
```

---

## 8 维评估体系

| 维度 | 英文 | 权重 | 评估重点 |
|------|------|------|---------|
| 目标理解 | goal_understanding | 1.0 | Agent 是否正确解析了用户意图 |
| 规划能力 | planning | 1.0 | 是否有合理的 todo list / sub-goals |
| 工具选择 | tool_selection | 1.0 | 每步选的 tool 是否合适 |
| 参数生成 | parameter_generation | 1.0 | tool 的参数是否完整、正确 |
| 执行准确度 | execution_accuracy | 1.0 | 工具执行结果是否符合预期 |
| 反思自纠错 | reflection_correction | 1.0 | 出错后能否识别并修正 |
| 状态跟踪 | state_tracking | 1.0 | 是否记住中间结果并正确传递 |
| 终止控制 | termination_control | 1.0 | 是否在完成时正确停止 |

---

## GEPA 开关 (config.json)

```json
{
  "evaluator": { "gepa": { "enabled": false } },
  "optimizer": {
    "gepa": {
      "enabled": false,
      "generations": 3,
      "population_size": 5,
      "elite_size": 2,
      "max_population": 6,
      "complexity_penalty": 0.01
    }
  }
}
```

**开启 GEPA**：将 `enabled` 设为 `true`，运行 `python pipeline/scripts/main.py optimize ...`。

---

## 快速开始

```bash
# 1) 预处理数据
python pipeline/scripts/main.py phase1 --data-dir ./data_preprocessing/raw_datasets/agentboard

# 2) 运行 Agent
python pipeline/scripts/main.py run --input ./data/cold_start.jsonl --output ./output/traces.jsonl

# 3) Rule 评估
python pipeline/scripts/main.py rule --input ./output/traces.jsonl --output ./output/rule_scores.jsonl

# 4) LLM 评估
python pipeline/scripts/main.py llm --input ./output/traces.jsonl --output ./output/llm_scores.jsonl

# 5) 融合评估
python pipeline/scripts/main.py hybrid \
  --rule-input ./output/rule_scores.jsonl \
  --llm-input ./output/llm_scores.jsonl \
  --output ./output/hybrid_scores.jsonl

# 6) GEPA 优化
python pipeline/scripts/main.py optimize \
  --input ./output/hybrid_scores.jsonl \
  --output ./output/optimized_prompt.json
```

---

## License

MIT
