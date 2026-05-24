# Agent Evaluator — Agent 数据飞轮评估器

> 一个完整的 Agent 执行轨迹评估与自动优化系统。支持 **Rule / LLM / Hybrid** 三种评估模式，内置 **GEPA 遗传进化优化器**，可驱动 Agent 能力的持续提升。

---

## 目录

- [项目概述](#项目概述)
- [架构](#架构)
- [8 维评估体系](#8-维评估体系)
- [快速开始](#快速开始)
- [完整操作流程](#完整操作流程)
  - [Phase 1: 数据预处理](#phase-1-数据预处理)
  - [Phase 2: Agent 执行](#phase-2-agent-执行)
  - [Phase 3: 评估打分](#phase-3-评估打分)
  - [Phase 4: 对齐分析](#phase-4-对齐分析)
  - [Phase 5: 优化迭代](#phase-5-优化迭代)
  - [飞轮主控](#飞轮主控)
- [模块详解](#模块详解)
- [数据格式](#数据格式)
- [环境依赖](#环境依赖)
- [常见问题](#常见问题)
- [下一步](#下一步)

---

## 项目概述

Agent Evaluator 解决的核心问题：**如何量化评估一个 Agent 的执行质量，并自动优化它。**

它不是一个黑箱打分工具，而是一个可解释、可对比、可迭代的评估引擎：

| 能力 | 说明 |
|------|------|
| **8 维评估** | 从目标理解到终止控制，覆盖 Agent 全生命周期 |
| **多评估器融合** | Rule（零成本、快）+ LLM（语义细）+ Hybrid（取长补短） |
| **评估器对齐分析** | 自动定位 Rule 与 LLM 评分差异最大的维度，输出调参建议 |
| **GEPA 遗传优化** | 用进化算法自动搜索最优 Prompt，带 Pareto 多目标约束 |
| **回归约束** | 优化时不允许任何维度退化超过阈值 |

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Evaluator                       │
├─────────────────────────────────────────────────────────────┤
│  Phase 1          Phase 2           Phase 3                  │
│  数据预处理   →   Agent 执行    →   评估打分                  │
│                                                              │
│  ┌─────────┐    ┌──────────┐     ┌─────────────────────┐     │
│  │AgentBoard│ → │  Agent   │  →  │  RuleGrader (规则)   │     │
│  │  数据   │    │  Runner  │     │  LLMGrader (语义)    │     │
│  └─────────┘    └──────────┘     │  HybridGrader (融合) │     │
│                                    └─────────────────────┘     │
│                                           ↓                  │
│  Phase 4                           Phase 5                   │
│  对齐分析                      →   GEPA 优化器                 │
│                                                              │
│  ┌─────────────┐              ┌─────────────────────┐       │
│  │GraderAligner│              │  遗传进化 + Pareto   │       │
│  │(差异定位)   │              │  多目标 + 回归约束   │       │
│  └─────────────┘              └─────────────────────┘       │
│                                           ↓                  │
│                                    优化后的 Prompt            │
│                                           ↓                  │
│                                    重新注入 Agent (闭环)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 8 维评估体系

| 维度 | 英文 | 权重 | 评估重点 |
|------|------|------|---------|
| 目标理解 | goal_understanding | 1.0 | Agent 是否正确解析了用户意图 |
| 规划能力 | planning | 1.0 | 是否有合理的 todo list / sub-goals |
| 工具选择 | tool_selection | 1.0 | 每步选的 tool 是否合适 |
| 参数生成 | argument_generation | 1.0 | tool 的参数是否完整、正确 |
| 执行准确度 | execution_accuracy | 1.0 | 工具执行结果是否符合预期 |
| 反思自纠错 | reflection_correction | 1.0 | 出错后能否识别并修正 |
| 状态跟踪 | state_tracking | 1.0 | 是否记住中间结果并正确传递 |
| 终止控制 | termination_control | 1.0 | 是否在完成时正确停止 |

**评分范围**: 每维度 0–10 分，总分 0–80 分。

---

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/YOUR_NAME/agent-evaluator.git
cd agent-evaluator
pip install -r requirements.txt
```

### 2. 配置 API Key（可选，仅 LLMGrader / GEPA 需要）

```bash
export SILICONFLOW_API_KEY="sk-xxxxxxxx"
# 或修改 src/llm_grader.py 和 src/gepa_optimizer.py 中的默认 key
```

### 3. 一条命令跑完评估流程

```bash
# 1) 预处理数据（需先有 AgentBoard 数据集）
python main.py phase1 --data-dir ./agentboard_data --output-dir ./data

# 2) 运行 Agent 生成轨迹
python main.py run --input ./data/cold_start.jsonl --output ./output/agent_traces.jsonl

# 3) Rule 评估
python main.py rule --input ./output/agent_traces.jsonl --output ./output/rule_scores.jsonl

# 4) LLM 评估
python main.py llm --input ./output/agent_traces.jsonl --output ./output/llm_scores.jsonl

# 5) 融合评估
python main.py hybrid \
  --rule-input ./output/rule_scores.jsonl \
  --llm-input ./output/llm_scores.jsonl \
  --output ./output/hybrid_scores.jsonl

# 6) 对齐分析
python main.py align \
  --rule ./output/rule_scores.jsonl \
  --llm ./output/llm_scores.jsonl \
  --output-dir ./output/alignment
```

---

## 完整操作流程

### Phase 1: 数据预处理

**目标**: 从 AgentBoard 原始数据中提取 badcase，规整为 7 字段标准格式。

**输入**: AgentBoard 数据集 (`tool-query`, `tool-operation`, `webarena` 三个子集)

**输出**:
- `data/cold_start.jsonl` — 20% 冷启动数据
- `data/validation.jsonl` — 80% 验证数据
- `data/all_badcases.jsonl` — 全部 badcase
- Excel 备份 (`.xlsx`)

**执行**:
```bash
python main.py phase1 --data-dir /path/to/agentboard_data --output-dir ./data
```

**数据字段**:
```json
{
  "id": "task_001",
  "question": "原始任务描述",
  "steps": [
    {
      "step": 1,
      "thought": "Agent 思考",
      "tool_call": "search",
      "observation": "搜索结果",
      "latency_ms": 1200,
      "tokens": 256
    }
  ],
  "output": "最终答案",
  "ground_truth": "标准答案",
  "outcome": "success / failure"
}
```

---

### Phase 2: Agent 执行

**目标**: 让 Agent 在任务上执行，生成可评估的轨迹。

**两种模式**:

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| Mock | 内置启发式 Agent，不调用真实 LLM | 快速验证流程 |
| SiliconFlow | 调用 Qwen2.5-14B / DeepSeek-V3 等真实模型 | 真实能力评估 |

**Mock 模式快速测试**:
```bash
python main.py run \
  --input ./data/cold_start.jsonl \
  --output ./output/mock_traces.jsonl \
  --model mock
```

**真实模型模式**:
```bash
python main.py run \
  --input ./data/cold_start.jsonl \
  --output ./output/real_traces.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct \
  --api-key $SILICONFLOW_API_KEY \
  --max-steps 15
```

---

### Phase 3: 评估打分

#### 3a) RuleGrader — 规则评估（零成本、速度快）

基于正则、关键词、结构分析的 8 维评分。适合大批量初筛。

```bash
python main.py rule --input ./output/traces.jsonl --output ./output/rule_scores.jsonl
```

**特点**:
- 零 API 成本
- 毫秒级评分
- 对结构化特征敏感（如是否有 todo list、参数是否缺失）
- 对语义理解较弱

#### 3b) LLMGrader — LLM-as-Judge（语义细、成本高）

用 Qwen2.5-14B 对每条轨迹按 8 维度逐条评分，带详细理由。

```bash
python main.py llm \
  --input ./output/traces.jsonl \
  --output ./output/llm_scores.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct
```

**特点**:
- 能捕捉语义退化（如参数质量下降）
- 方差大：同一条轨迹两次评分可能差 1–2 分
- 建议同一轨迹评 3 次取中位数
- 每条耗时 5–15 秒

#### 3c) HybridGrader — 融合评估（推荐）

按维度动态选择 Rule 或 LLM，取长补短。

```bash
python main.py hybrid \
  --rule-input ./output/rule_scores.jsonl \
  --llm-input ./output/llm_scores.jsonl \
  --output ./output/hybrid_scores.jsonl
```

**默认策略**:
| 维度 | 策略 | 原因 |
|------|------|------|
| goal_understanding | LLM | 语义理解 |
| planning | Hybrid | 结构 + 语义 |
| tool_selection | Rule | 关键词匹配 |
| argument_generation | Rule | 参数格式检查 |
| execution_accuracy | Hybrid | 结果匹配 |
| reflection_correction | LLM | 语义判断 |
| state_tracking | Hybrid | 上下文跟踪 |
| termination_control | Hybrid | 终止时机 |

---

### Phase 4: 对齐分析

对比 RuleGrader 与 LLMGrader 的输出，回答三个问题：
1. 哪些维度两者一致性高？（Rule 可以独立承担）
2. 哪些维度差异大？（需要 LLM 介入或调参）
3. 整体相关性如何？

```bash
python main.py align \
  --rule ./output/rule_scores.jsonl \
  --llm ./output/llm_scores.jsonl \
  --output-dir ./output/alignment
```

**输出**:
- `alignment_report.json` — 结构化报告
- `alignment_report.txt` — 可读版本

**报告内容**:
- 维度级 Pearson 相关系数
- 差异样本列表
- 调参建议（如 "planning 维度 Rule 偏严，建议放宽 todo list 检测"）

---

### Phase 5: 优化迭代（GEPA）

GEPA = **G**enetic **E**volution with **P**areto and **A**daptive prompt.

**核心思想**: 用遗传算法自动进化 Agent 的 system prompt，每一代用评估器打分作为 fitness，多目标 Pareto 选择保留最优解。

**关键约束 — 回归保护**:
- 任何维度不得退化超过 `REGRESSION_THRESHOLD = 1.0` 分
- 优化目标从"最大化目标维度"改为"最大化目标维度 + 惩罚退化"

**执行**:
```bash
python main.py optimize \
  --input ./output/hybrid_scores.jsonl \
  --output ./output/optimized_prompt.json \
  --generations 5 \
  --population 6
```

**输出**:
- `optimized_prompt.json` — 最优 prompt + 各代 fitness 曲线
- `gepa_output/` — 每代候选 prompt 存档

---

### 飞轮主控

上述 5 个 Phase 可以串成一个自动循环：

```
Round 1: Baseline Agent → 评估 → 记录分数
      ↓
   GEPA 优化 → 生成 Prompt Patch
      ↓
Round 2: 注入 Patch 的 Agent → 评估 → 对比 Round 1
      ↓
  回归检测: 任何维度退化 > 1.0？
    ├─ YES → 缩小 patch 范围 / 拒绝本次优化
    └─ NO  → 接受优化，进入 Round 3
```

飞轮脚本示例（需自行编写主控脚本调用 main.py 各子命令）:
```bash
# 伪代码
for round in 1 2 3; do
  python main.py run --input tasks.jsonl --output traces_r${round}.jsonl --prompt prompt_r${round}.txt
  python main.py hybrid --rule-input rule_r${round}.jsonl --llm-input llm_r${round}.jsonl --output hybrid_r${round}.jsonl
  python main.py optimize --input hybrid_r${round}.jsonl --output prompt_r$((round+1)).txt
done
```

---

## 模块详解

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `src/phase1_preprocess.py` | AgentBoard 数据清洗、字段规整、20/80 拆分 | 原始数据集 | `cold_start.jsonl`, `validation.jsonl` |
| `src/agent_runner.py` | Agent 执行引擎（Mock / 真实模型） | 任务 JSONL | 轨迹 JSONL |
| `src/rule_grader.py` | 8 维规则评估器 | 轨迹 JSONL | 评分 JSONL |
| `src/llm_grader.py` | LLM-as-Judge 评估器 | 轨迹 JSONL | 评分 JSONL + 理由 |
| `src/hybrid_grader.py` | Rule + LLM 融合评估器 | Rule + LLM 评分 JSONL | 融合评分 JSONL |
| `src/grader_aligner.py` | 评估器对齐分析器 | Rule + LLM 评分 JSONL | 对齐报告 |
| `src/gepa_optimizer.py` | GEPA 遗传进化优化器 | 评分 JSONL (fitness) | 优化后 prompt |
| `src/quick_rule_grade.py` | 快速命令行评分入口 | 轨迹 JSONL | 评分 JSONL |
| `main.py` | 统一 CLI 入口 | 子命令参数 | 各阶段输出 |

---

## 数据格式

### 轨迹文件 (Agent 执行后产出)

```jsonl
{"id": "task_001", "question": "...", "steps": [...], "output": "...", "ground_truth": "...", "outcome": "failure"}
```

### 评分文件 (Grader 产出)

```jsonl
{
  "id": "task_001",
  "scores": {
    "goal_understanding": 5.9,
    "planning": 4.8,
    "tool_selection": 7.3,
    "argument_generation": 8.5,
    "execution_accuracy": 5.6,
    "reflection_correction": 3.6,
    "state_tracking": 5.0,
    "termination_control": 3.6
  },
  "total_score": 44.3,
  "comments": { "goal_understanding": "...", ... },
  "evidence": { "goal_understanding": ["step 1 thought: ..."], ... }
}
```

---

## 环境依赖

- Python 3.10+
- 核心依赖: `requests`, `openai`, `pandas`, `numpy`, `scikit-learn`
- 可选: `beautifulsoup4`（真实搜索工具）, `playwright`（真实浏览器工具）

```bash
pip install -r requirements.txt
```

**API 配置**:
- 硅基流动: `https://api.siliconflow.cn/v1`
- 支持模型: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `deepseek-ai/DeepSeek-V3`

---

## 常见问题

**Q: LLMGrader 评分方差大怎么办？**
> 同一轨迹评 3 次取中位数。HybridGrader 已内置多次评分接口。

**Q: RuleGrader 和 LLMGrader 结论不一致怎么办？**
> 用 `grader_aligner.py` 分析差异维度，定位后调参或改用 Hybrid。

**Q: Ground Truth 匹配率 0%？**
> WebArena 的 GT 是内部测试网站路径，公网无法访问。如需真实验证，筛选公网可验证任务（如电影查询、论文比较），或接入真实浏览器 + 搜索引擎。

**Q: 网络环境限制搜索引擎访问？**
> 当前运行环境可能限制外网访问。建议在本地（有完整外网）运行真实搜索测试，或使用 Mock 模式验证流程。

**Q: GEPA 优化效果不明显？**
> 14B 模型对 prompt 补丁响应有限。建议：
> 1. 换 32B+ 模型做优化器
> 2. 优化器输出 concrete few-shot 示例而非抽象规则
> 3. 换 DeepSeek-R1 等推理模型生成 patch

---

## 下一步

- [ ] 接入真实浏览器（Playwright）执行 WebArena 任务
- [ ] 扩展 Tree 评估器覆盖更多维度（零成本、无方差）
- [ ] 评估器多次评分取平均，消除 LLM 方差
- [ ] 优化器输出 concrete few-shot 示例
- [ ] 支持更多数据源（GAIA、SWE-bench）
- [ ] Web UI 可视化评估结果与飞轮进度

---

## License

MIT
