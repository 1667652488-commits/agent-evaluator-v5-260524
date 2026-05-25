#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_grader.py — LLM-as-Judge 8维评估器
======================================
用 Qwen2.5-14B-Instruct 对 Agent 执行轨迹按 8 个维度评分。
输入: JSONL 轨迹文件 (goal, thoughts, tool_calls, observations, output, ...)
输出: JSONL 评分结果 (llm_scores + 理由)

设计要点:
- 零样本 + 详细评分标准 (rubric) + 思维链 (CoT)
- 记录每条耗时与 token 消耗
- 支持断点续跑 (已处理记录跳过)
- 输出格式与 agent_grader_v2 兼容，方便后续对齐

2025-05-23 | 冷灯
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
API_BASE = "https://api.siliconflow.cn/v1"
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-nnksashvwdizsenvqlnlcyhevvzpqntwswvutxcqukzfhkyc")

DIMENSIONS = [
    "goal_understanding",
    "planning",
    "tool_selection",
    "argument_generation",
    "execution_accuracy",
    "reflection_correction",
    "state_tracking",
    "termination_control",
]

DIMENSION_NAMES_CN = {
    "goal_understanding": "目标理解",
    "planning": "规划能力",
    "tool_selection": "工具选择",
    "argument_generation": "参数生成",
    "execution_accuracy": "执行准确度",
    "reflection_correction": "反思自纠错",
    "state_tracking": "状态跟踪",
    "termination_control": "终止控制",
}

# 评分标准 (rubric) —— 详细到每个分数档，减少 LLM 方差
RUBRIC = """
【评分标准】每个维度 1-10 分，必须严格按以下档位打分：

1. goal_understanding (目标理解)
   - 10分: Agent 在 Thought 中明确复述了目标的关键实体和约束条件，理解完全正确。
   - 7-9分: 目标理解基本正确，但遗漏了部分约束或实体。
   - 4-6分: 目标理解有偏差，执行方向与目标部分不符。
   - 1-3分: 目标理解严重错误，执行方向完全偏离。
   - 0分:  未体现任何目标理解。

2. planning (规划能力)
   - 10分: 有明确的 todo list 或步骤拆解，步骤间逻辑顺序合理，无冗余。
   - 7-9分: 有步骤规划，但步骤数偏少(1-2步)或逻辑顺序有轻微问题。
   - 4-6分: 规划模糊，仅提到要做事但未拆解，或步骤有冗余/顺序错乱。
   - 1-3分: 无规划，直接执行单一动作。
   - 0分:  无任何执行动作。

3. tool_selection (工具选择)
   - 10分: 每个工具调用都与目标精确匹配，无多余或错误工具。
   - 7-9分: 工具选择基本正确，有1次轻微不匹配。
   - 4-6分: 工具选择有明显错误(如查询任务调用代码执行工具)。
   - 1-3分: 严重错误工具选择，或完全未调用必要工具。
   - 0分:  未调用任何工具。

4. argument_generation (参数生成)
   - 10分: 所有参数格式正确、完整，JSON 语法无误，必填项无缺失。
   - 7-9分: 参数基本正确，有1处轻微格式问题或可选字段缺失。
   - 4-6分: 参数有明显错误(类型错误、必填项缺失、格式错乱)。
   - 1-3分: 参数严重错误导致工具无法执行。
   - 0分:  未生成参数或参数完全为空。

5. execution_accuracy (执行准确度)
   - 10分: 执行结果与 ground_truth 完全一致，无偏差。
   - 7-9分: 执行结果基本正确，有轻微遗漏或格式差异。
   - 4-6分: 执行结果部分正确，有遗漏或错误但部分达成目标。
   - 1-3分: 执行结果严重偏离目标，几乎未达成。
   - 0分:  执行完全失败，无任何有效输出。

6. reflection_correction (反思自纠错)
   - 10分: 当出现错误或偏差时，Agent 主动识别、分析原因并修正，且修正有效。
   - 7-9分: 有反思迹象(提到"检查""错误")，但未明确修正或修正无效。
   - 4-6分: 仅在 Thought 中提到问题，未采取行动修正。
   - 1-3分: 执行失败但无任何反思或问题识别。
   - 0分:  无错误场景(无法评估反思能力)。
   注意: 若执行成功且无错误，此维度默认给 8 分(无需反思)。

7. state_tracking (状态跟踪)
   - 10分: 多步骤任务中，Agent 持续关注已完成/待完成事项，Thought 中有状态总结。
   - 7-9分: 隐含状态跟踪(步骤推进合理)，但未明确提及历史状态。
   - 4-6分: 有轻微遗忘迹象(重复查询已获取的信息)。
   - 1-3分: 严重状态遗忘，重复执行同一动作或遗漏关键步骤。
   - 0分:  单步骤任务(无状态跟踪需求)。

8. termination_control (终止控制)
   - 10分: 任务完成后及时终止，输出完整且正确，无多余步骤。
   - 7-9分: 终止时机基本合理，输出基本完整。
   - 4-6分: 终止过早(输出不完整)或过晚(有冗余步骤)。
   - 1-3分: 严重过早终止(几乎无输出)或进入无限循环。
   - 0分:  未进行任何操作。
"""

# ------------------------------------------------------------------
# OpenAI 客户端初始化
# ------------------------------------------------------------------

def get_client():
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE, api_key=API_KEY)
    except Exception as e:
        print(f"[ERR] OpenAI client 初始化失败: {e}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Prompt 构造
# ------------------------------------------------------------------

def build_grading_prompt(record: Dict[str, Any]) -> str:
    """构造 LLM 评分 prompt，包含轨迹详情和评分标准。"""
    goal = record.get("goal", record.get("question", "N/A"))
    thoughts = record.get("thoughts", "")
    if isinstance(thoughts, list):
        thoughts = "\n".join(str(t) for t in thoughts)
    tool_calls = record.get("tool_calls", "")
    if isinstance(tool_calls, list):
        tool_calls = json.dumps(tool_calls, ensure_ascii=False, indent=2)
    observations = record.get("observations", "")
    if isinstance(observations, list):
        observations = "\n".join(str(o) for o in observations)
    output = record.get("output", "")
    ground_truth = record.get("ground_truth", "")
    outcome = record.get("outcome", "N/A")
    num_steps = record.get("num_steps", 0)

    # 截断过长的字段，防止超出上下文
    MAX_LEN = 800
    thoughts = (thoughts[:MAX_LEN] + "...[截断]") if len(str(thoughts)) > MAX_LEN else thoughts
    observations = (observations[:MAX_LEN] + "...[截断]") if len(str(observations)) > MAX_LEN else observations
    tool_calls = (tool_calls[:MAX_LEN] + "...[截断]") if len(str(tool_calls)) > MAX_LEN else tool_calls
    ground_truth = (ground_truth[:MAX_LEN] + "...[截断]") if len(str(ground_truth)) > MAX_LEN else ground_truth

    prompt = f"""你是一位严格的 Agent 执行评估专家。请根据以下 Agent 执行轨迹，按 8 个维度进行评分。

【Agent 执行轨迹】
目标: {goal}
结果状态: {outcome}
执行步数: {num_steps}

思考过程 (Thoughts):
{thoughts}

工具调用 (Tool Calls):
{tool_calls}

环境反馈 (Observations):
{observations}

最终输出 (Output):
{output}

标准答案 (Ground Truth):
{ground_truth}

{RUBRIC}

【输出要求】
请对以上轨迹进行思考链分析，然后输出 STRICT JSON，格式如下:
{{
  "analysis": "你的分析过程，说明每个维度的判断依据",
  "scores": {{
    "goal_understanding": 0,
    "planning": 0,
    "tool_selection": 0,
    "argument_generation": 0,
    "execution_accuracy": 0,
    "reflection_correction": 0,
    "state_tracking": 0,
    "termination_control": 0
  }},
  "reasons": {{
    "goal_understanding": "理由",
    "planning": "理由",
    "tool_selection": "理由",
    "argument_generation": "理由",
    "execution_accuracy": "理由",
    "reflection_correction": "理由",
    "state_tracking": "理由",
    "termination_control": "理由"
  }}
}}

注意:
1. 分数必须是 0-10 的整数。
2. analysis 字段必须有实质性内容，不能空泛。
3. 只输出 JSON，不要 Markdown 代码块标记。
"""
    return prompt


# ------------------------------------------------------------------
# LLM 调用
# ------------------------------------------------------------------

def call_llm_grade(client, prompt: str, max_retries: int = 2) -> Tuple[Dict[str, Any], float, int, int]:
    """
    调用 LLM 评分，返回 (result_dict, elapsed_sec, prompt_tokens, completion_tokens)。
    失败时重试，最终失败返回空 dict。
    """
    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # 低温度减少方差
                max_tokens=1500,
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content.strip()
            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

            # 提取 JSON
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            result = json.loads(content)

            # 校验分数
            scores = result.get("scores", {})
            for dim in DIMENSIONS:
                if dim not in scores or not isinstance(scores[dim], (int, float)):
                    scores[dim] = 0
                else:
                    scores[dim] = int(max(0, min(10, scores[dim])))

            return result, elapsed, prompt_tokens, completion_tokens

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON 解析失败 (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt == max_retries:
                return {}, elapsed, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"  [WARN] LLM 调用失败 (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt == max_retries:
                return {}, time.time() - start, 0, 0
            time.sleep(1)
    return {}, 0.0, 0, 0


# ------------------------------------------------------------------
# 数据 I/O
# ------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] 文件不存在: {path}", file=sys.stderr)
        return records
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def grade_file(input_path: str, output_path: str, limit: int = 0, resume: bool = True) -> Dict[str, Any]:
    client = get_client()
    records = load_jsonl(input_path)
    print(f"[INFO] 加载 {len(records)} 条记录 from {input_path}")

    if limit > 0:
        records = records[:limit]
        print(f"[INFO] 限制处理前 {limit} 条")

    # 断点续跑: 已存在的输出文件中读取已处理 ID
    processed_ids = set()
    if resume and Path(output_path).exists():
        existing = load_jsonl(output_path)
        processed_ids = {r.get("id", i) for i, r in enumerate(existing) if "llm_scores" in r}
        print(f"[INFO] 断点续跑: 已处理 {len(processed_ids)} 条，跳过")

    results = []
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "total_elapsed_sec": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }

    for i, rec in enumerate(records):
        rec_id = rec.get("id", i)
        if rec_id in processed_ids:
            continue

        # 字段适配: all_traces.jsonl 没有 goal，用 category+id 拼接作为 goal
        if not rec.get("goal") and not rec.get("question"):
            rec["goal"] = f"[{rec.get('category','unknown')}] task {rec.get('id','?')}"

        stats["total"] += 1
        print(f"\n[{i+1}/{len(records)}] ID={rec_id} 评分中...")

        prompt = build_grading_prompt(rec)
        result, elapsed, pt, ct = call_llm_grade(client, prompt)

        stats["total_elapsed_sec"] += elapsed
        stats["total_prompt_tokens"] += pt
        stats["total_completion_tokens"] += ct

        if result and "scores" in result:
            rec["llm_scores"] = result["scores"]
            rec["llm_reasons"] = result.get("reasons", {})
            rec["llm_analysis"] = result.get("analysis", "")
            rec["llm_meta"] = {
                "elapsed_sec": round(elapsed, 2),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "model": MODEL_NAME,
            }
            results.append(rec)
            stats["success"] += 1
            print(f"  ✓ 完成 | 耗时 {elapsed:.1f}s | tokens {pt}+{ct}")
            # 逐条保存，防崩溃丢失
            save_jsonl(results, output_path)
        else:
            stats["failed"] += 1
            print(f"  ✗ 失败 | 耗时 {elapsed:.1f}s")

    # 最终保存
    save_jsonl(results, output_path)

    print(f"\n{'='*60}")
    print("LLM Grader 完成")
    print(f"  总计: {stats['total']} | 成功: {stats['success']} | 失败: {stats['failed']}")
    print(f"  总耗时: {stats['total_elapsed_sec']:.1f}s")
    print(f"  总 prompt tokens: {stats['total_prompt_tokens']}")
    print(f"  总 completion tokens: {stats['total_completion_tokens']}")
    print(f"  输出: {output_path}")
    print(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge 8维评估器")
    parser.add_argument("--input", required=True, help="输入 JSONL 轨迹文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 评分文件")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数 (0=全部)")
    parser.add_argument("--no-resume", action="store_true", help="不启用断点续跑")
    args = parser.parse_args()

    grade_file(args.input, args.output, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
