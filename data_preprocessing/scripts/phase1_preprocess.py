#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 v5: AgentBoard 数据预处理
核心变更：
  1. 新增 question 字段（原始问题）
  2. thoughts/tool_calls/observations 改为 step-based 结构
  3. 6 个字段 → 7 个字段（goal + steps[] + output + ground_truth + outcome）

输出目录: agent_board_selected_data/ (workspace 路径下)
字段: question, steps[{step, thought, tool_call, observation, latency_ms, tokens}], output, ground_truth, outcome
"""

import json
import os
import random
import sys
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ===================== 配置区 =====================
POSSIBLE_DATA_DIRS = [
    "/mnt/d/bank3.0/agent_board_data/agentboard_data",
    "/mnt/d/agentboard_data",
    "./agentboard_data",
    "../agentboard_data",
]

DIR_SELECTED = Path("./agent_board_selected_data")
DIR_COLD     = Path("./cold_data2grader")
DIR_VALID    = Path("./validation_data")
DIR_GRADER   = Path("./generated_grader")
DIR_ADVISOR  = Path("./advisor")

DATASETS = {
    "tool-query":      {"subdir": "tool-query",      "category": "tool-query"},
    "tool-operation":  {"subdir": "tool-operation",  "category": "tool-operation"},
    "webarena":        {"subdir": "webarena",        "category": "webarena"},
}

SEED = 42
SPLIT_RATIO = 0.2
# =================================================

def find_data_dir():
    for p in POSSIBLE_DATA_DIRS:
        p = Path(p)
        if p.exists() and any((p / ds["subdir"] / "test.jsonl").exists() for ds in DATASETS.values()):
            print(f"[OK] 发现数据目录: {p}")
            return p
    print("[ERR] 未找到 AgentBoard 数据。请修改 POSSIBLE_DATA_DIRS 为你的实际路径。")
    sys.exit(1)

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def extract_ground_truth(sample):
    for key in ("answer", "subgoals", "target", "goal", "evaluator_answer", "intent", "gt"):
        if key in sample and sample[key]:
            return sample[key]
    return ""

def is_badcase(sample, ground_truth):
    success = sample.get("success")
    result = sample.get("result")
    reward = sample.get("reward")
    if success is False:
        return True, "success=False"
    if isinstance(reward, (int, float)) and reward < 0:
        return True, f"reward={reward}<0"
    if result and ground_truth and result != ground_truth:
        return True, "result != ground_truth"
    if success is True or (isinstance(reward, (int, float)) and reward > 0):
        return False, "success=True"
    return True, "unclear outcome"

def determine_outcome(sample, ground_truth, is_bad, reason):
    if is_bad:
        if sample.get("success") is False:
            return "失败"
        if sample.get("result") and ground_truth and sample.get("result") != ground_truth:
            return "失败"
        return "部分通过"
    return "通过"

def extract_steps(sample):
    """
    从轨迹中提取 step-based 结构
    返回: [{"step": n, "thought": str, "tool_call": str, "observation": str, "latency_ms": int, "tokens": int}]
    """
    trajectory = sample.get("trajectory") or sample.get("conversations") or sample.get("messages") or sample.get("steps") or []
    steps = []

    if isinstance(trajectory, list):
        for i, step_data in enumerate(trajectory):
            if isinstance(step_data, dict):
                thought = step_data.get("thought") or step_data.get("thinking") or step_data.get("reasoning") or ""
                action = step_data.get("action") or step_data.get("tool") or step_data.get("tool_call") or step_data.get("name") or ""
                params = step_data.get("params") or step_data.get("parameters") or step_data.get("args") or {}

                # 构造 tool_call 字符串
                if action:
                    if isinstance(params, dict) and params:
                        tool_call = action + '(' + json.dumps(params, ensure_ascii=False) + ')'
                    else:
                        tool_call = action
                else:
                    tool_call = ""

                obs = step_data.get("observation") or step_data.get("result") or step_data.get("output") or step_data.get("response") or ""
                latency = step_data.get("latency_ms") or step_data.get("latency") or step_data.get("elapsed") or 0
                tokens = step_data.get("tokens") or step_data.get("token_count") or step_data.get("usage", {}).get("total_tokens", 0)

                steps.append({
                    "step": i + 1,
                    "thought": str(thought)[:2000],
                    "tool_call": str(tool_call)[:500],
                    "observation": str(obs)[:1000],
                    "latency_ms": int(latency) if isinstance(latency, (int, float)) else 0,
                    "tokens": int(tokens) if isinstance(tokens, (int, float)) else 0,
                })
            else:
                # 非 dict 的 step，作为 observation 处理
                steps.append({
                    "step": i + 1,
                    "thought": "",
                    "tool_call": "",
                    "observation": str(step_data)[:1000],
                    "latency_ms": 0,
                    "tokens": 0,
                })

    return steps

def normalize_record(sample, category):
    gt = extract_ground_truth(sample)
    is_bad, reason = is_badcase(sample, gt)
    outcome = determine_outcome(sample, gt, is_bad, reason)

    steps = extract_steps(sample)

    # 从最后一步或独立字段提取 output
    output = sample.get("output") or sample.get("result") or sample.get("answer") or ""
    if not output and steps:
        last_obs = steps[-1].get("observation", "")
        if "任务结束" in last_obs or "finish" in last_obs.lower():
            # 尝试从最后一步的 tool_call 提取 finish 参数
            last_tool = steps[-1].get("tool_call", "")
            if "finish" in last_tool.lower():
                import re as re_local
                m = re_local.search(r'"answer"\s*:\s*"([^"]*)"', last_tool)
                if m:
                    output = m.group(1)

    return {
        "id": sample.get("id") or sample.get("task_id") or sample.get("index", ""),
        "category": category,
        "goal": sample.get("goal") or sample.get("query") or sample.get("instruction") or sample.get("task") or sample.get("goal") or "",
        "steps": steps,
        "output": str(output)[:2000],
        "ground_truth": gt,
        "outcome": outcome,
        "badcase_reason": reason,
        "num_steps": len(steps),
        "raw_success": sample.get("success"),
        "raw_reward": sample.get("reward"),
    }

def split_records(records, ratio=SPLIT_RATIO, seed=SEED):
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]

def save_json(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] JSON: {path} ({len(records)} 条)")

def save_excel(records, path):
    if not HAS_PANDAS or not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 把 steps 数组展开为字符串以便 Excel 展示
    df_records = []
    for r in records:
        r_copy = r.copy()
        r_copy["steps_json"] = json.dumps(r_copy.get("steps", []), ensure_ascii=False)
        del r_copy["steps"]
        df_records.append(r_copy)
    df = pd.DataFrame(df_records)
    df.to_excel(path, index=False)
    print(f"[OK] Excel: {path} ({len(records)} 条)")

def main():
    data_dir = find_data_dir()

    for d in [DIR_SELECTED, DIR_COLD, DIR_VALID, DIR_GRADER, DIR_ADVISOR]:
        d.mkdir(parents=True, exist_ok=True)

    all_badcases = []
    stats = {}

    for name, info in DATASETS.items():
        test_path = data_dir / info["subdir"] / "test.jsonl"
        if not test_path.exists():
            print(f"[WARN] 跳过不存在: {test_path}")
            continue

        raw = load_jsonl(test_path)
        print(f"[INFO] {name}: 加载 {len(raw)} 条原始记录")

        badcases = []
        for r in raw:
            gt = extract_ground_truth(r)
            is_bad, reason = is_badcase(r, gt)
            if is_bad:
                badcases.append(normalize_record(r, info["category"]))

        stats[name] = {"total_raw": len(raw), "badcases": len(badcases)}
        all_badcases.extend(badcases)

        save_json(badcases, DIR_SELECTED / f"{name}_badcases.jsonl")
        save_excel(badcases, DIR_SELECTED / f"{name}_badcases.xlsx")

    save_json(all_badcases, DIR_SELECTED / "all_badcases.jsonl")
    save_excel(all_badcases, DIR_SELECTED / "all_badcases.xlsx")

    cold_data, val_data = split_records(all_badcases)

    save_json(cold_data, DIR_COLD / "cold_start.jsonl")
    save_excel(cold_data, DIR_COLD / "cold_start.xlsx")

    save_json(val_data, DIR_VALID / "validation.jsonl")
    save_excel(val_data, DIR_VALID / "validation.xlsx")

    print("\n" + "=" * 60)
    print("Phase 1 v5 预处理完成 — step-based + question 字段")
    print("=" * 60)
    for name, s in stats.items():
        print(f"  {name:20s} 原始={s['total_raw']:4d}  badcases={s['badcases']:4d}")
    print(f"\n  总 badcases: {len(all_badcases)}")
    print(f"  冷启动 20%: {len(cold_data)} 条 -> {DIR_COLD}/")
    print(f"  验证   80%: {len(val_data)} 条 -> {DIR_VALID}/")
    print(f"\n  输出目录:")
    print(f"    {DIR_SELECTED}/  — 全部 badcase 数据（goal + steps[]）")
    print(f"    {DIR_COLD}/      — 冷启动数据")
    print(f"    {DIR_VALID}/     — 验证数据")
    print(f"    {DIR_GRADER}/    — 评估器输出（Phase 2 使用）")
    print(f"    {DIR_ADVISOR}/   — 建议输出（Phase 4 使用）")
    print("=" * 60)
    print("\n字段说明:")
    print("  goal: 原始用户问题")
    print("  steps[n]: {step, thought, tool_call, observation, latency_ms, tokens}")
    print("  output:   agent 最终输出")
    print("  ground_truth: 标准答案")
    print("  outcome:  通过/失败/部分通过")

if __name__ == "__main__":
    main()
