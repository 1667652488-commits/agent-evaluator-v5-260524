#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_grader_v5.py — 8 维评估器 (step-based 结构)
适配字段: goal + steps[{step, thought, tool_call, observation, latency_ms, tokens}]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

DIMENSIONS = {
    "goal_understanding": {"weight": 1.0, "desc": "目标理解"},
    "planning": {"weight": 1.0, "desc": "规划能力"},
    "tool_selection": {"weight": 1.0, "desc": "工具选择"},
    "argument_generation": {"weight": 1.0, "desc": "参数生成"},
    "execution_accuracy": {"weight": 1.0, "desc": "执行准确度"},
    "reflection_correction": {"weight": 1.0, "desc": "反思自纠错"},
    "state_tracking": {"weight": 1.0, "desc": "状态跟踪"},
    "termination_control": {"weight": 1.0, "desc": "终止控制"},
}

class BaseGrader:
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class CompositeGrader:
    def __init__(self, graders: List[BaseGrader]):
        self.graders = {g.dimension: g for g in graders}
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        result = {"id": sample.get("id", ""), "scores": {}, "comments": {}, "evidence": {}}
        total = 0.0
        weight_sum = 0.0
        for dim_name, meta in DIMENSIONS.items():
            grader = self.graders.get(dim_name)
            if grader is None:
                result["scores"][dim_name] = 0.0
                result["comments"][dim_name] = "未配置"
                continue
            r = grader.grade(sample)
            score = float(r.get("score", 0))
            result["scores"][dim_name] = score
            result["comments"][dim_name] = r.get("comment", "")
            result["evidence"][dim_name] = r.get("evidence", None)
            total += score * meta["weight"]
            weight_sum += meta["weight"]
        result["total_score"] = round(total / weight_sum, 2) if weight_sum > 0 else 0
        return result
    def batch_grade(self, samples: List[Dict[str, Any]], progress_every: int = 50) -> List[Dict[str, Any]]:
        results = []
        for i, s in enumerate(samples):
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"[Progress] 已评估 {i+1}/{len(samples)}")
            results.append(self.grade(s))
        return results

# ===================== 8 维规则评估器 =====================

class GoalUnderstandingGrader(BaseGrader):
    dimension = "goal_understanding"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("goal", "")
        steps = sample.get("steps", [])
        outcome = sample.get("outcome", "")

        # 从所有 steps 的 thought 中提取关键词
        all_thoughts = " ".join([s.get("thought", "") for s in steps]).lower()
        keywords = set(re.findall(r"[a-zA-Z_\u4e00-\u9fa5]+", goal.lower()))
        keywords = {k for k in keywords if len(k) > 1}
        matched = sum(1 for k in keywords if k in all_thoughts)
        coverage = matched / len(keywords) if keywords else 1.0

        # 检查约束词
        constraint_violation = False
        negations = ["不", "禁止", "无需", "不要", "避免"]
        for neg in negations:
            if neg in goal.lower():
                # 简单规则：如果 question 有"不"但 steps 里有相反动作
                pass

        if outcome == "失败":
            score = max(0, int(coverage * 10) - 3)
            comment = f"目标理解偏差，覆盖率 {coverage:.0%}，执行失败"
        elif coverage < 0.5:
            score = 4
            comment = f"部分理解偏差，覆盖率 {coverage:.0%}"
        elif coverage < 0.8:
            score = 7
            comment = f"基本理解目标，覆盖率 {coverage:.0%}"
        else:
            score = min(10, 8 + int(coverage * 2))
            comment = f"目标理解良好，覆盖率 {coverage:.0%}"

        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"coverage": coverage, "keywords": list(keywords)[:10]}}

class PlanningGrader(BaseGrader):
    dimension = "planning"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])
        num_steps = len(steps)

        # 检测规划信号
        planning_signals = ["步骤", "计划", "先", "然后", "再", "最后", "第一步", "分解", "拆解", "stage", "step", "plan"]
        has_planning = any(s in " ".join([step.get("thought", "") for step in steps]).lower() for s in planning_signals)

        # 检测冗余（重复 tool_call）
        tools = [s.get("tool_call", "").split("(")[0].strip() for s in steps if s.get("tool_call")]
        redundant = len(tools) - len(set(tools))

        if not has_planning and num_steps < 3:
            score = 2
            comment = "无规划迹象，步骤极少"
        elif redundant > 2:
            score = 5
            comment = f"有规划但冗余({redundant}次重复工具)"
        elif has_planning and num_steps >= 3:
            score = min(10, 7 + num_steps)
            comment = f"规划良好，{num_steps} 步，有明确拆解"
        else:
            score = 6
            comment = f"规划一般，{num_steps} 步"

        return {"dimension": self.dimension, "score": min(10, score), "comment": comment,
                "evidence": {"steps": num_steps, "redundant": redundant, "has_planning": has_planning}}

class ToolSelectionGrader(BaseGrader):
    dimension = "tool_selection"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])
        question = sample.get("goal", "").lower()

        tools = [s.get("tool_call", "").split("(")[0].strip() for s in steps if s.get("tool_call")]
        total = len(tools)
        if total == 0:
            return {"dimension": self.dimension, "score": 0, "comment": "无工具调用", "evidence": None}

        # 简单关键词匹配检测错误工具
        wrong = 0
        if "天气" in question and any("code" in t.lower() for t in tools):
            wrong += 1
        if "翻译" in question and not any("translat" in t.lower() for t in tools):
            wrong += 1

        error_rate = wrong / total if total > 0 else 0
        score = max(0, int(10 * (1 - error_rate)))

        if score >= 9: comment = f"工具选择精准 ({total} 次)"
        elif score >= 5: comment = f"部分有误 ({wrong} 次错误)"
        else: comment = f"大量错误 ({wrong} 次)"

        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"total": total, "wrong": wrong}}

class ArgumentGenerationGrader(BaseGrader):
    dimension = "argument_generation"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])

        param_errors = 0
        error_signals = ["参数错误", "参数缺失", "格式错误", "invalid argument", "missing required"]
        for s in steps:
            obs = s.get("observation", "").lower()
            for sig in error_signals:
                if sig in obs:
                    param_errors += 1

        has_json = any("{" in s.get("tool_call", "") and "}" in s.get("tool_call", "") for s in steps)

        if param_errors > 2: score = 2
        elif param_errors > 0: score = 5
        elif not has_json and steps: score = 6
        else: score = 8

        comment = f"参数问题: {param_errors} 处" if param_errors > 0 else "参数格式正确"
        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"param_errors": param_errors, "has_json": has_json}}

class ExecutionAccuracyGrader(BaseGrader):
    dimension = "execution_accuracy"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])
        output = sample.get("output", "")
        ground_truth = sample.get("ground_truth", "")
        outcome = sample.get("outcome", "")

        deviation_signals = ["偏差", "错误", "失败", "未成功", "error", "failed"]
        deviations = sum(1 for s in steps if any(sig in s.get("observation", "").lower() for sig in deviation_signals))

        if outcome == "失败":
            score = max(0, 3 - deviations)
            comment = f"执行失败，{deviations} 处偏差"
        elif deviations > 3: score = 4
        elif deviations > 0: score = 7
        else: score = 9

        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"deviations": deviations, "outcome": outcome}}

class ReflectionCorrectionGrader(BaseGrader):
    dimension = "reflection_correction"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])
        outcome = sample.get("outcome", "")

        signals = ["反思", "修正", "重新", "调整", "改", "纠正", "retry", "correct", "alternative", "检查", "排查"]
        found = []
        for s in steps:
            thought_obs = (s.get("thought", "") + " " + s.get("observation", "")).lower()
            found.extend([sig for sig in signals if sig in thought_obs])
        found = list(set(found))  # 去重

        if outcome == "失败":
            score = 5 if found else 1
            comment = f"失败但尝试修正: {', '.join(found[:3])}" if found else "失败且无修正"
        elif outcome == "部分通过":
            score = 7 if found else 3
            comment = f"部分成功，有修正" if found else "部分成功，未修正"
        else:
            score = 10 if found else 8
            comment = "成功且有反思" if found else "成功完成"

        return {"dimension": self.dimension, "score": score, "comment": comment, "evidence": found}

class StateTrackingGrader(BaseGrader):
    dimension = "state_tracking"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])

        forget_signals = ["忘记", "遗忘", "重复", "再次", "又", "already", "again"]
        forget_count = 0
        for s in steps:
            txt = (s.get("thought", "") + " " + s.get("observation", "")).lower()
            forget_count += sum(1 for sig in forget_signals if sig in txt)

        tools = [s.get("tool_call", "").split("(")[0].strip() for s in steps if s.get("tool_call")]
        seen = set()
        repeats = 0
        for t in tools:
            if t in seen:
                repeats += 1
            seen.add(t)

        if forget_count > 2 or repeats > 2: score = 3
        elif forget_count > 0 or repeats > 0: score = 6
        else: score = 9

        comment = f"跟踪问题: {forget_count} 处遗忘，{repeats} 次重复" if (forget_count or repeats) else "状态跟踪良好"
        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"forget_signals": forget_count, "repeats": repeats}}

class TerminationControlGrader(BaseGrader):
    dimension = "termination_control"
    def grade(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        steps = sample.get("steps", [])
        output = sample.get("output", "")
        outcome = sample.get("outcome", "")

        num_steps = len(steps)
        loop_signals = ["循环", "无限", "重复输出", "same result"]
        loops = sum(1 for s in steps if any(sig in s.get("observation", "").lower() for sig in loop_signals))

        premature = (not output or len(output) < 10) and outcome != "通过"
        excessive = num_steps > 50 and loops > 0

        if excessive: score = 2
        elif premature: score = 3
        elif loops > 0: score = 5
        else: score = 9

        comment = "无限循环" if excessive else ("提前终止" if premature else ("有循环迹象" if loops else "终止控制良好"))
        return {"dimension": self.dimension, "score": score, "comment": comment,
                "evidence": {"steps": num_steps, "loops": loops, "premature": premature}}

# ===================== 建议生成器 =====================
ADVICE_TEMPLATES = {
    "goal_understanding": {
        "high": ["优化指令解析 prompt，增加『目标拆解+约束核对』步骤", "对模糊指令增加追问机制"],
        "medium": ["建立常见指令语料库，强化相似指令识别"],
    },
    "planning": {
        "high": ["引入『任务拆解 prompt』，强制输出子步骤及逻辑顺序", "建立常见任务标准流程模板"],
        "medium": ["增加规划校验环节，检查步骤合理性"],
    },
    "tool_selection": {
        "high": ["建立『工具-任务』映射表，强化工具功能认知", "优化工具调用触发条件"],
        "medium": ["增加工具匹配提醒 prompt"],
    },
    "argument_generation": {
        "high": ["提供工具参数模板，强制按模板生成参数", "增加参数校验环节"],
        "medium": ["对常见参数错误做针对性 prompt 优化"],
    },
    "execution_accuracy": {
        "high": ["优化执行步骤 prompt，明确执行标准", "增加执行结果校验环节"],
        "medium": ["对常见执行偏差增加针对性约束"],
    },
    "reflection_correction": {
        "high": ["引入『错误反思 prompt』，强制排查错误原因", "建立常见错误库及修正方案"],
        "medium": ["增加多轮纠错机制"],
    },
    "state_tracking": {
        "high": ["引入『状态跟踪 prompt』，每步更新已完成/未完成子任务", "采用上下文总结机制"],
        "medium": ["对长时序任务拆分上下文，分阶段跟踪"],
    },
    "termination_control": {
        "high": ["优化终止判断 prompt，明确任务完成标准", "增加终止校验环节"],
        "medium": ["限制最大执行步数，避免无限循环"],
    },
}

def generate_advice(scores: Dict[str, float], top_k: int = 3) -> List[Dict[str, Any]]:
    sorted_dims = sorted(scores.items(), key=lambda x: x[1])
    advice_list = []
    for dim_name, score in sorted_dims[:top_k]:
        templates = ADVICE_TEMPLATES.get(dim_name, {"high": ["请检查此维度"], "medium": []})
        if score <= 3:
            priority, suggestions = "high", templates.get("high", [])
        elif score <= 6:
            priority, suggestions = "medium", templates.get("medium", templates.get("high", []))
        else:
            priority, suggestions = "low", templates.get("medium", [])
        if not suggestions:
            suggestions = [f"{dim_name} 得分 {score}，建议关注"]
        advice_list.append({"target_dimension": dim_name, "current_score": score,
                           "priority": priority, "suggestions": suggestions})
    return advice_list

def format_advice_text(advice_list: List[Dict[str, Any]]) -> str:
    lines = ["=" * 50, "Agent 评估优化建议", "=" * 50]
    for i, adv in enumerate(advice_list, 1):
        lines.append("\n【优先级 " + str(i) + "】" + adv["target_dimension"] + " (得分: " + str(adv["current_score"]) + ")")
        lines.append("优先级: " + adv["priority"].upper())
        for s in adv["suggestions"]:
            lines.append("  - " + s)
    lines.append("\n" + "=" * 50)
    return "\n".join(lines)

# ===================== 工具函数 =====================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
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

def save_jsonl(records: List[Dict[str, Any]], path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    dim_scores = {d: [] for d in DIMENSIONS.keys()}
    totals = []
    for r in results:
        totals.append(r.get("total_score", 0))
        for d in DIMENSIONS.keys():
            dim_scores[d].append(r["scores"].get(d, 0))
    avg_by_dim = {d: round(sum(scores) / len(scores), 2) if scores else 0 for d, scores in dim_scores.items()}
    weakest = []
    for dim, score in sorted(avg_by_dim.items(), key=lambda x: x[1])[:3]:
        high_count = sum(1 for r in results if any(a["target_dimension"] == dim and a["priority"] == "high" for a in r.get("advice", [])))
        weakest.append({"dimension": dim, "avg_score": score, "priority": "high" if high_count > n * 0.3 else "medium"})
    return {
        "total_samples": n,
        "avg_total_score": round(sum(totals) / n, 2),
        "avg_by_dimension": avg_by_dim,
        "weakest_dimensions": weakest,
    }

# ===================== 主流水线 =====================
def run_pipeline(input_path: str, output_dir: str, sample_limit: int = 0):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] 输入: {input_path}")
    print(f"[Pipeline] 输出: {output_dir}")

    samples = load_jsonl(str(input_path))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    print(f"[Pipeline] 加载 {len(samples)} 条样本")
    if not samples:
        print("[ERR] 无数据，退出")
        return

    graders = [
        GoalUnderstandingGrader(), PlanningGrader(), ToolSelectionGrader(),
        ArgumentGenerationGrader(), ExecutionAccuracyGrader(),
        ReflectionCorrectionGrader(), StateTrackingGrader(), TerminationControlGrader(),
    ]
    grader = CompositeGrader(graders)
    results = grader.batch_grade(samples, progress_every=50)

    print("[Pipeline] 生成优化建议...")
    for r in results:
        advice = generate_advice(r["scores"], top_k=3)
        r["advice"] = advice
        r["advice_text"] = format_advice_text(advice)

    save_jsonl(results, str(output_dir / "grader_results.jsonl"))
    print(f"[Pipeline] 逐条结果已保存")

    summary = generate_summary(results)
    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] 汇总报告已保存")

    print("\n" + "=" * 60)
    print("评估完成摘要")
    print("=" * 60)
    print(f"  样本总数:      {summary['total_samples']}")
    print(f"  平均分:        {summary['avg_total_score']}")
    print("\n  各维度平均分:")
    for dim, score in summary["avg_by_dimension"].items():
        print(f"    {dim:25s}: {score:.2f}")
    print("\n  最弱维度（Top 3）:")
    for item in summary["weakest_dimensions"]:
        print(f"    {item['dimension']:25s}: {item['avg_score']:.2f}  [{item['priority']}]")
    print("=" * 60)
    return results, summary

def main():
    parser = argparse.ArgumentParser(description="Agent 8维评估器 v5 (step-based)")
    parser.add_argument("--input", required=True, help="输入 jsonl 文件路径")
    parser.add_argument("--output", default="./generated_grader", help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="限制处理样本数（0=全部）")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.limit)

if __name__ == "__main__":
    main()
