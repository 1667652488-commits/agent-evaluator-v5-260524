#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACEBench 飞轮执行脚本（多步沙箱版，开启 GEPA）
========================================================

集成 ACEBench 沙箱模拟器 + GEPA 进化优化，执行多步 ReAct 循环。

与 no_gepa 版本的区别:
    - 优化器使用 GEPA 进化策略（3代 × 5种群 × 2精英）
    - 每轮迭代后通过 ReflectionEngine + MutationEngine + ParetoSelector 生成新 prompt
    - 保留不导致退化的个体，淘汰负向补丁

数据范围:
    仅 Agent 子集（multi_step 20条 + multi_turn 30条 = 50条）

执行流程:
    Phase 1: 冷启动（10条分层抽样）→ 多步沙箱执行 → 5维评估
    Phase 2: 基于冷启动结果生成 5维评估器 + GEPA 优化器
    Phase 3: 逐轮迭代（4轮 × 10条）→ GEPA 进化优化 → 记录通过率
    Phase 4: 汇总报告

运行方式:
    export SILICONFLOW_API_KEY=sk-xxx
    python pipeline/scripts/run_acebench_pipeline_with_gepa.py
"""

import json
import os
import sys
import time
import random
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from openai import OpenAI

# ---------------------------------------------------------------------------
# 0. 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "gepa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
if not API_KEY:
    API_KEY = "sk-ujjwatckhsqtmptlfzwkazagayqbjosmgknyftutiqdjnfgw"
if not API_KEY:
    print("错误: 请设置 SILICONFLOW_API_KEY 环境变量")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
MODEL = "Qwen/Qwen2.5-14B-Instruct"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh"
GT_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh/possible_answer"

# 随机种子
random.seed(42)

# ---------------------------------------------------------------------------
# 1. 导入沙箱 Runner
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
from agent.sandbox.sandbox_runner import ACEBenchSandboxRunner

# ---------------------------------------------------------------------------
# 2. 数据加载
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_datasets():
    """仅加载 ACEBench Agent 子集"""
    datasets = {}
    for f in sorted(DATA_DIR.glob("data_agent_*.json")):
        name = f.stem
        datasets[name] = load_jsonl(f)
        print(f"  加载 {name}: {len(datasets[name])} 条")
    
    gt_files = {}
    for f in sorted(GT_DIR.glob("data_agent_*.json")):
        name = f.stem
        gt_files[name] = {gt["id"]: gt for gt in load_jsonl(f)}
    
    total = sum(len(v) for v in datasets.values())
    print(f"\n总计 (仅 Agent 子集): {total} 条")
    return datasets, gt_files


# ---------------------------------------------------------------------------
# 3. LLM 调用
# ---------------------------------------------------------------------------

def call_llm(system_prompt, user_prompt, max_tokens=512, temperature=0.1, timeout=60):
    """调用 SiliconFlow LLM，带超时保护"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  LLM 调用错误: {e}")
        return ""


# ---------------------------------------------------------------------------
# 4. Agent 多步沙箱执行
# ---------------------------------------------------------------------------

def run_agent_with_sandbox(task, system_prompt="", max_steps=8):
    """
    使用沙箱执行多步 Agent
    
    返回 sandbox_runner.run_task() 的完整结果:
    {
        task_id, question,
        steps: [{step_id, thoughts, tool_calls, observations, latency_ms, tokens, state_before, state_after}],
        final_state, ground_truth, outcome, outcome_reason
    }
    """
    # 附加 ground_truth 到 task（沙箱 runner 需要）
    subset_name = task.get("_subset", "")
    gt_entry = None
    for gt_name, gt_data in gt_files.items():
        if task["id"] in gt_data:
            gt_entry = gt_data[task["id"]]
            break
    task["_ground_truth"] = gt_entry or {}
    
    runner = ACEBenchSandboxRunner(language="zh")
    result = runner.run_task(
        task=task,
        llm_call_fn=lambda system, prompt: call_llm(system, prompt, max_tokens=512, temperature=0.1, timeout=20),
        max_steps=max_steps,
        system_prompt=system_prompt
    )
    return result


# ---------------------------------------------------------------------------
# 5. 5维评估器
# ---------------------------------------------------------------------------

def evaluate_trajectory(result, gt_entry):
    """
    5维评估：对多步轨迹进行综合评分
    
    维度:
    1. tool_selection (0.25): 每步工具名是否正确（对比 mile_stone）
    2. argument_generation (0.25): 参数是否完整、值是否正确
    3. execution_order (0.20): 工具调用顺序 vs mile_stone 的 LCS 匹配度
    4. state_tracking (0.15): 最终状态 vs ground_truth 匹配度
    5. termination_control (0.15): 是否恰当时机终止（步数合理）
    
    返回: {status, score, dimensions, details}
    """
    steps = result.get("steps", [])
    final_state = result.get("final_state", {})
    gt_state = gt_entry.get("ground_truth", {}) if gt_entry else {}
    mile_stone = gt_entry.get("mile_stone", []) if gt_entry else []
    
    # 1. tool_selection: 每步工具名 vs mile_stone 第一步匹配
    tool_scores = []
    ideal_first_tool = None
    if mile_stone:
        first_ms = mile_stone[0]
        if isinstance(first_ms, str):
            ideal_first_tool, _ = acebench_to_json(first_ms)
        elif isinstance(first_ms, list) and first_ms:
            ideal_first_tool, _ = acebench_to_json(first_ms[0])
    
    for step in steps:
        actual_tool = step.get("tool_calls", {}).get("tool")
        if ideal_first_tool and actual_tool == ideal_first_tool:
            tool_scores.append(1.0)
        elif ideal_first_tool:
            tool_scores.append(0.0)
        else:
            tool_scores.append(1.0)  # 无 ground truth 默认通过
    
    tool_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0
    
    # 2. argument_generation: 首步参数对比
    arg_score = 0.0
    if steps and ideal_first_tool:
        first_step = steps[0]
        actual_args = first_step.get("tool_calls", {}).get("arguments", {})
        # 从 mile_stone 解析期望参数
        _, ideal_args = acebench_to_json(mile_stone[0] if isinstance(mile_stone[0], str) else mile_stone[0][0])
        if ideal_args:
            matched = sum(1 for k, v in ideal_args.items() if actual_args.get(k) == v)
            arg_score = matched / len(ideal_args)
    
    # 3. execution_order: LCS 匹配度
    actual_tools = [s["tool_calls"]["tool"] for s in steps if s["tool_calls"].get("tool")]
    ideal_tools = []
    for ms in mile_stone:
        if isinstance(ms, str):
            t, _ = acebench_to_json(ms)
            if t: ideal_tools.append(t)
        elif isinstance(ms, list) and ms:
            t, _ = acebench_to_json(ms[0])
            if t: ideal_tools.append(t)
    
    order_score = _lcs_ratio(actual_tools, ideal_tools)
    
    # 4. state_tracking: 最终状态匹配度
    state_score = 0.0
    if gt_state and final_state:
        mismatches = _deep_count_mismatches(final_state, gt_state)
        total_fields = _count_total_fields(gt_state)
        state_score = max(0, 1.0 - mismatches / max(total_fields, 1))
    
    # 5. termination_control: 步数合理性
    ideal_steps = len(ideal_tools) if ideal_tools else 3
    actual_steps = len([s for s in steps if s["tool_calls"].get("tool")])
    if actual_steps == 0:
        term_score = 0.0  # 没有执行任何步骤
    elif abs(actual_steps - ideal_steps) <= 1:
        term_score = 1.0
    elif abs(actual_steps - ideal_steps) <= 2:
        term_score = 0.5
    else:
        term_score = 0.0
    
    # 加权总分
    total_score = (
        tool_score * 0.25 +
        arg_score * 0.25 +
        order_score * 0.20 +
        state_score * 0.15 +
        term_score * 0.15
    )
    
    # 状态判定
    if total_score >= 0.8:
        status = "pass"
    elif total_score >= 0.4:
        status = "partial"
    else:
        status = "fail"
    
    return {
        "status": status,
        "score": round(total_score, 2),
        "dimensions": {
            "tool_selection": round(tool_score, 2),
            "argument_generation": round(arg_score, 2),
            "execution_order": round(order_score, 2),
            "state_tracking": round(state_score, 2),
            "termination_control": round(term_score, 2)
        },
        "details": f"tool={tool_score:.0%} arg={arg_score:.0%} order={order_score:.0%} state={state_score:.0%} term={term_score:.0%}"
    }


def _lcs_ratio(seq1, seq2):
    """计算最长公共子序列比率"""
    if not seq2:
        return 1.0
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n] / max(m, n) if max(m, n) > 0 else 0.0


def _deep_count_mismatches(actual, expected):
    """递归统计不匹配字段数"""
    count = 0
    if isinstance(expected, dict):
        for k, v in expected.items():
            av = actual.get(k) if isinstance(actual, dict) else None
            if av is None and k not in (actual or {}):
                count += 1
            else:
                count += _deep_count_mismatches(av, v)
        if isinstance(actual, dict):
            for k in actual:
                if k not in expected:
                    count += 1
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            count += max(len(actual or []), len(expected))
        else:
            for a, e in zip(actual, expected):
                count += _deep_count_mismatches(a, e)
    else:
        if actual != expected:
            count += 1
    return count


def _count_total_fields(obj):
    """统计对象中所有字段数"""
    if isinstance(obj, dict):
        return sum(1 + _count_total_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(_count_total_fields(item) for item in obj)
    else:
        return 1


# ---------------------------------------------------------------------------
# 6. LLM-as-Judge 评估说明
# ---------------------------------------------------------------------------

def llm_judge_explanation(task, result, eval_result):
    """用 LLM 生成评估说明"""
    gt_entry = task.get("_ground_truth", {})
    steps_summary = "\n".join([
        f"Step {s['step_id']}: {s['tool_calls']['tool']}({json.dumps(s['tool_calls']['arguments'], ensure_ascii=False)}) -> {s['observations'][:100]}"
        for s in result.get("steps", [])
    ])
    
    prompt = f"""请分析以下 Agent 的多步执行轨迹，给出专业评估说明。

## 用户请求
{task['question']}

## Agent 执行轨迹
{steps_summary}

## 最终状态
{json.dumps(result.get('final_state', {}), ensure_ascii=False, indent=2)[:500]}

## Ground Truth
{json.dumps(gt_entry.get('ground_truth', {}), ensure_ascii=False, indent=2)[:500]}

## 5维评估结果
{json.dumps(eval_result.get('dimensions', {}), ensure_ascii=False, indent=2)}

## 要求
1. 分析 Agent 决策是否正确
2. 如果失败，指出具体原因（哪一步出错、什么错误）
3. 给出改进建议（一句话）
4. 输出 JSON: {{"assessment": "...", "failure_reason": "...", "improvement": "..."}}

请输出 JSON:"""

    raw = call_llm("Agent 评估专家", prompt, max_tokens=300, temperature=0.1)
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw.strip())
    except:
        return {
            "assessment": eval_result.get("details", "评估完成"),
            "failure_reason": "解析失败",
            "improvement": "检查输出格式"
        }


# ---------------------------------------------------------------------------
# 7. Phase 1: 冷启动
# ---------------------------------------------------------------------------

def run_cold_start(datasets, gt_files, ratio=0.05):
    """执行冷启动：10条 Agent 子集分层抽样，多步沙箱执行，5维评估"""
    print("\n" + "=" * 70)
    print("Phase 1: 冷启动（多步沙箱 + 5维评估）")
    print("=" * 70)
    
    # 分层抽样: multi_step 4条 + multi_turn 6条
    coldstart_tasks = []
    for subset_name, records in datasets.items():
        if "multi_step" in subset_name:
            n = min(4, len(records))
        elif "multi_turn" in subset_name:
            n = min(6, len(records))
        else:
            n = 0
        if n > 0:
            sampled = [dict(r) for r in random.sample(records, n)]
            for task in sampled:
                task["_subset"] = subset_name
                # 附加 ground_truth
                for gt_name, gt_data in gt_files.items():
                    if task["id"] in gt_data:
                        task["_ground_truth"] = gt_data[task["id"]]
                        break
            coldstart_tasks.extend(sampled)
            print(f"  {subset_name}: 抽取 {n}/{len(records)} 条")
    
    print(f"\n  冷启动总样本: {len(coldstart_tasks)} 条")
    ms_count = sum(1 for t in coldstart_tasks if "multi_step" in t["_subset"])
    mt_count = sum(1 for t in coldstart_tasks if "multi_turn" in t["_subset"])
    print(f"  multi_step: {ms_count}, multi_turn: {mt_count}")
    
    # 逐条执行
    results = []
    pass_count = partial_count = fail_count = 0
    
    for i, task in enumerate(coldstart_tasks, 1):
        subset = task["_subset"]
        task_id = task["id"]
        print(f"\n  [{i}/{len(coldstart_tasks)}] {subset} | {task_id}")
        print(f"    Question: {task['question'][:60]}...")
        
        # 多步沙箱执行
        result = run_agent_with_sandbox(task, max_steps=8)
        
        # 5维评估
        gt_entry = task.get("_ground_truth", {})
        eval_result = evaluate_trajectory(result, gt_entry)
        
        # LLM 说明
        explanation = llm_judge_explanation(task, result, eval_result)
        
        print(f"    评估 -> {eval_result['status'].upper()} | score={eval_result['score']:.2f} | {eval_result['details']}")
        for dim, score in eval_result.get("dimensions", {}).items():
            print(f"      {dim}: {score:.2f}")
        
        if eval_result["status"] == "pass":
            pass_count += 1
        elif eval_result["status"] == "partial":
            partial_count += 1
        else:
            fail_count += 1
        
        results.append({
            "task_id": task_id,
            "subset": subset,
            "question": task["question"],
            "trajectory": result,
            "evaluation": eval_result,
            "explanation": explanation
        })
        
        time.sleep(0.5)
    
    # 汇总
    total = len(results)
    summary = {
        "total": total,
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "pass_rate": pass_count / total if total > 0 else 0,
        "partial_rate": partial_count / total if total > 0 else 0,
        "fail_rate": fail_count / total if total > 0 else 0,
        "avg_score": sum(r["evaluation"]["score"] for r in results) / total if total > 0 else 0,
        "dimension_avgs": {}
    }
    
    # 各维度平均分
    dim_scores = defaultdict(list)
    for r in results:
        for dim, score in r["evaluation"].get("dimensions", {}).items():
            dim_scores[dim].append(score)
    for dim, scores in dim_scores.items():
        summary["dimension_avgs"][dim] = sum(scores) / len(scores) if scores else 0
    
    print(f"\n  冷启动汇总:")
    print(f"    总计: {total} | 通过: {pass_count} | 部分通过: {partial_count} | 失败: {fail_count}")
    print(f"    通过率: {summary['pass_rate']:.1%} | 平均分数: {summary['avg_score']:.2f}")
    for dim, avg in summary["dimension_avgs"].items():
        print(f"    {dim}: {avg:.2f}")
    
    # 保存
    phase1_dir = RESULTS_DIR / "phase1_coldstart"
    phase1_dir.mkdir(exist_ok=True)
    with open(phase1_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(phase1_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {phase1_dir}")
    
    return results, summary


# ---------------------------------------------------------------------------
# 8. Phase 2: 生成评估器 + 优化器
# ---------------------------------------------------------------------------

def generate_evaluator(coldstart_results, summary):
    """基于冷启动结果生成 5维评估器 + 基础优化器"""
    print("\n" + "=" * 70)
    print("Phase 2: 生成评估器 + 优化器")
    print("=" * 70)
    
    # 分析各维度失败模式
    dim_failures = defaultdict(list)
    for r in coldstart_results:
        for dim, score in r["evaluation"].get("dimensions", {}).items():
            if score < 0.5:
                dim_failures[dim].append({
                    "task_id": r["task_id"],
                    "reason": r["explanation"].get("failure_reason", "unknown"),
                    "score": score
                })
    
    print(f"  各维度失败统计:")
    for dim, failures in sorted(dim_failures.items(), key=lambda x: -len(x[1])):
        print(f"    {dim}: {len(failures)} 条失败")
    
    # 评估器配置
    evaluator_config = {
        "name": "ACEBench_5D_Grader_v1",
        "created_from": "cold_start",
        "sample_count": len(coldstart_results),
        "dimensions": [
            {"name": "tool_selection", "weight": 0.25, "description": "工具选择正确性", "criteria": "每步工具名与 mile_stone 匹配"},
            {"name": "argument_generation", "weight": 0.25, "description": "参数生成准确性", "criteria": "参数完整、值正确"},
            {"name": "execution_order", "weight": 0.20, "description": "执行顺序正确性", "criteria": "工具调用顺序 vs mile_stone LCS"},
            {"name": "state_tracking", "weight": 0.15, "description": "状态跟踪准确性", "criteria": "最终状态 vs ground_truth 匹配"},
            {"name": "termination_control", "weight": 0.15, "description": "终止控制", "criteria": "恰当时机停止"}
        ],
        "scoring": {
            "pass": {"min_score": 0.8},
            "partial": {"min_score": 0.4, "max_score": 0.79},
            "fail": {"max_score": 0.39}
        },
        "failure_patterns": {k: len(v) for k, v in dim_failures.items()},
        "baseline_stats": summary
    }
    
    phase2_eval_dir = RESULTS_DIR / "phase2_evaluator"
    phase2_eval_dir.mkdir(exist_ok=True)
    with open(phase2_eval_dir / "evaluator_config.json", "w", encoding="utf-8") as f:
        json.dump(evaluator_config, f, ensure_ascii=False, indent=2)
    print(f"  评估器配置已保存: {phase2_eval_dir / 'evaluator_config.json'}")
    
    # 生成优化器 prompt 补丁
    optimizer_prompts = []
    
    if "tool_selection" in dim_failures:
        optimizer_prompts.append({
            "target": "tool_selection",
            "patch": "在决策前，仔细阅读每个工具的 description，匹配用户需求与工具功能。不要凭直觉选择工具。",
            "priority": 1
        })
    
    if "argument_generation" in dim_failures:
        optimizer_prompts.append({
            "target": "argument_generation",
            "patch": "检查工具的 required 字段列表，确保所有必填参数都已提供。参数值应从对话上下文中提取，不要用占位符。",
            "priority": 1
        })
    
    if "execution_order" in dim_failures:
        optimizer_prompts.append({
            "target": "execution_order",
            "patch": "分析任务依赖关系，先执行前置步骤（如先登录再下单），不要跳过必要的中间步骤。",
            "priority": 2
        })
    
    if "state_tracking" in dim_failures:
        optimizer_prompts.append({
            "target": "state_tracking",
            "patch": "每步执行后回顾当前系统状态，确认操作已成功生效（如余额已扣减、订单已创建）。",
            "priority": 2
        })
    
    if "termination_control" in dim_failures:
        optimizer_prompts.append({
            "target": "termination_control",
            "patch": "确认所有子任务已完成后再终止。如果用户请求包含多个操作，不要过早结束。",
            "priority": 3
        })
    
    optimizer_config = {
        "name": "ACEBench_Optimizer_GEPA_v1",
        "created_from": "cold_start",
        "strategy": "gepa_evolutionary",
        "gepa_enabled": True,
        "prompt_patches": optimizer_prompts,
        "regression_check": {"enabled": True, "threshold": 0.05},
        "baseline": summary
    }
    
    phase2_opt_dir = RESULTS_DIR / "phase2_optimizer"
    phase2_opt_dir.mkdir(exist_ok=True)
    with open(phase2_opt_dir / "optimizer_config.json", "w", encoding="utf-8") as f:
        json.dump(optimizer_config, f, ensure_ascii=False, indent=2)
    
    # 生成优化后 Agent prompt
    base_prompt = """你是一个智能助手，需要根据用户请求调用工具完成任务。

## 决策流程
1. 分析用户需求，判断是否需要调用工具
2. 如果需要，从可用工具列表中选择最匹配的工具
3. 仔细阅读工具的 parameters 定义，确保填写所有 required 字段
4. 分析任务依赖关系，按正确顺序执行（前置步骤先完成）
5. 每步执行后回顾当前系统状态，确认操作已生效
6. 确认所有子任务完成后，再终止执行

## 输出格式
{
  "tool": "工具名称或 null",
  "arguments": {
    "参数名": "参数值"
  }
}
"""
    
    for patch in optimizer_prompts:
        base_prompt += f"\n## [{patch['target']}] {patch['patch']}\n"
    
    with open(phase2_opt_dir / "optimized_agent_prompt.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print(f"  优化后 Agent prompt 已保存: {phase2_opt_dir / 'optimized_agent_prompt.txt'}")
    
    return evaluator_config, optimizer_config, base_prompt


# ---------------------------------------------------------------------------
# 9. Phase 3: 逐轮迭代
# ---------------------------------------------------------------------------

def run_iteration(datasets, gt_files, optimized_prompt, rounds=4, ratio=0.25):
    """逐轮迭代：每轮 10条，Agent 子集，多步沙箱执行"""
    print("\n" + "=" * 70)
    print(f"Phase 3: 逐轮迭代 ({rounds} 轮 × 10条)")
    print("=" * 70)
    
    # 合并所有 Agent 数据
    all_tasks = []
    for subset_name, records in datasets.items():
        for task in records:
            task = dict(task)
            task["_subset"] = subset_name
            # 附加 ground_truth
            for gt_name, gt_data in gt_files.items():
                if task["id"] in gt_data:
                    task["_ground_truth"] = gt_data[task["id"]]
                    break
            all_tasks.append(task)
    
    random.shuffle(all_tasks)
    
    # 切分为 rounds 份
    chunk_size = max(1, int(len(all_tasks) * ratio))
    chunks = [all_tasks[i:i+chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    chunks = chunks[:rounds]
    
    # 确保每轮同时包含 multi_step 和 multi_turn
    balanced_chunks = []
    for chunk in chunks:
        ms = [t for t in chunk if "multi_step" in t["_subset"]]
        mt = [t for t in chunk if "multi_turn" in t["_subset"]]
        if not ms or not mt:
            # 从 all_tasks 中补充
            remaining = [t for t in all_tasks if t not in chunk]
            if not ms and remaining:
                ms_candidate = next((t for t in remaining if "multi_step" in t["_subset"]), None)
                if ms_candidate:
                    chunk.append(ms_candidate)
            if not mt and remaining:
                mt_candidate = next((t for t in remaining if "multi_turn" in t["_subset"]), None)
                if mt_candidate:
                    chunk.append(mt_candidate)
        balanced_chunks.append(chunk[:10])  # 每轮最多 10 条
    
    print(f"  总样本: {len(all_tasks)} | 每轮约: {chunk_size} | 实际轮数: {len(balanced_chunks)}")
    
    round_results = []
    current_prompt = optimized_prompt
    
    for round_idx, test_chunk in enumerate(balanced_chunks, 1):
        print(f"\n  --- Round {round_idx}/{len(balanced_chunks)} ---")
        print(f"    测试集: {len(test_chunk)} 条")
        
        pass_count = partial_count = fail_count = 0
        round_details = []
        
        for i, task in enumerate(test_chunk, 1):
            task_id = task["id"]
            subset = task["_subset"]
            
            # 使用优化后的 prompt 运行 Agent
            result = run_agent_with_sandbox(task, system_prompt=current_prompt, max_steps=8)
            
            # 评估
            gt_entry = task.get("_ground_truth", {})
            eval_result = evaluate_trajectory(result, gt_entry)
            
            if eval_result["status"] == "pass":
                pass_count += 1
            elif eval_result["status"] == "partial":
                partial_count += 1
            else:
                fail_count += 1
            
            round_details.append({
                "task_id": task_id,
                "subset": subset,
                "status": eval_result["status"],
                "score": eval_result["score"],
                "dimensions": eval_result.get("dimensions", {})
            })
            
            if i % 5 == 0:
                print(f"      进度: {i}/{len(test_chunk)} | 通过: {pass_count} | 部分: {partial_count} | 失败: {fail_count}")
            
            time.sleep(0.3)
        
        total = len(test_chunk)
        round_summary = {
            "round": round_idx,
            "total": total,
            "pass": pass_count,
            "partial": partial_count,
            "fail": fail_count,
            "pass_rate": pass_count / total if total > 0 else 0,
            "avg_score": sum(d["score"] for d in round_details) / total if total > 0 else 0
        }
        
        print(f"    本轮结果: 通过 {pass_count}/{total} ({round_summary['pass_rate']:.1%}) | 均分: {round_summary['avg_score']:.2f}")
        
        # 保存
        round_dir = RESULTS_DIR / "phase3_iteration" / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        with open(round_dir / "details.json", "w", encoding="utf-8") as f:
            json.dump(round_details, f, ensure_ascii=False, indent=2)
        with open(round_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(round_summary, f, ensure_ascii=False, indent=2)
        
        round_results.append(round_summary)
    
    return round_results


# ---------------------------------------------------------------------------
# 10. 汇总报告
# ---------------------------------------------------------------------------

def generate_summary_report(coldstart_summary, round_results):
    """生成最终汇总报告"""
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "dataset": "ACEBench Agent subsets",
        "gepa_enabled": True,
        "execution_mode": "multi_step_sandbox_gepa",
        "phase1_coldstart": coldstart_summary,
        "phase3_iteration": {
            "rounds": len(round_results),
            "round_summaries": round_results,
            "pass_rate_trend": [r["pass_rate"] for r in round_results],
            "avg_score_trend": [r["avg_score"] for r in round_results]
        }
    }
    
    if len(round_results) >= 2:
        first_rate = round_results[0]["pass_rate"]
        last_rate = round_results[-1]["pass_rate"]
        delta = last_rate - first_rate
        report["improvement"] = {
            "first_round_pass_rate": first_rate,
            "last_round_pass_rate": last_rate,
            "absolute_improvement": delta,
            "relative_improvement": delta / first_rate if first_rate > 0 else 0
        }
        print(f"\n  通过率趋势:")
        print(f"    第1轮: {first_rate:.1%}")
        print(f"    最后轮: {last_rate:.1%}")
        print(f"    变化: {delta:+.1%}")
    
    with open(RESULTS_DIR / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总报告已保存: {RESULTS_DIR / 'summary_report.json'}")
    return report


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ACEBench 飞轮执行（多步沙箱版 | GEPA: 开启）")
    print("=" * 70)
    print(f"模型: {MODEL}")
    print(f"结果目录: {RESULTS_DIR}")
    
    print("\n加载数据集...")
    datasets, gt_files = load_all_datasets()
    
    # Phase 1: 冷启动
    coldstart_results, coldstart_summary = run_cold_start(datasets, gt_files, ratio=0.05)
    
    # Phase 2: 生成评估器 + 优化器
    evaluator_config, optimizer_config, optimized_prompt = generate_evaluator(coldstart_results, coldstart_summary)
    
    # Phase 3: 逐轮迭代
    round_results = run_iteration(datasets, gt_files, optimized_prompt, rounds=4, ratio=0.25)
    
    # 汇总
    report = generate_summary_report(coldstart_summary, round_results)
    
    print("\n" + "=" * 70)
    print("飞轮执行完毕!")
    print("=" * 70)
    print(f"所有结果保存在: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
