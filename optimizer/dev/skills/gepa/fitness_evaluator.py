#!/usr/bin/env python3
"""
Fitness Evaluator — GEPA 种群适应度计算
基于 8 维规则评分计算 prompt candidate 的 fitness
"""
import re
from typing import Dict, Any, List, Tuple
from collections import Counter

from .prompt_candidate import DIMENSIONS, PromptCandidate


def extract_thoughts(record: Dict[str, Any]) -> str:
    thoughts = record.get("thoughts", "")
    if isinstance(thoughts, list):
        return "\n".join(str(t) for t in thoughts)
    return str(thoughts) if thoughts else ""


def extract_tool_calls(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = record.get("tool_calls", [])
    if isinstance(tool_calls, list):
        return tool_calls
    if isinstance(tool_calls, str):
        calls = []
        for part in tool_calls.split("|"):
            part = part.strip()
            if not part:
                continue
            m = re.search(r'Action:\s*(\w+)\((.*)\)', part)
            if m:
                calls.append({"tool": m.group(1), "params": m.group(2)})
            else:
                calls.append({"tool": part, "params": {}})
        return calls
    return []


def grade_goal_understanding(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = extract_thoughts(record)
    goal = str(record.get("goal", record.get("question", "")))
    signals = []
    if goal and thoughts:
        goal_keywords = set(re.findall(r"\w{3,}", goal.lower()))
        thought_keywords = set(re.findall(r"\w{3,}", thoughts.lower()))
        if goal_keywords:
            overlap = len(goal_keywords & thought_keywords) / len(goal_keywords)
            if overlap < 0.2:
                signals.append("low_goal_keyword_overlap")
                return 4, f"thoughts 中未充分体现目标关键词 (coverage {overlap:.0%})", signals
    return 8, "目标理解基本正确", signals


def grade_planning(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = extract_thoughts(record).lower()
    tool_calls = extract_tool_calls(record)
    signals = []
    planning_markers = re.findall(
        r"(?i)(step \d+|first|second|third|then|next|plan|planning|拆解|步骤|先|然后|再|子任务|subtask)",
        thoughts
    )
    num_steps = len(tool_calls) if tool_calls else record.get("num_steps", 0)
    has_markers = len(planning_markers) > 0
    if num_steps >= 3 and has_markers:
        return 9, "多步骤执行，有规划表述", signals
    elif num_steps >= 2:
        return 7, "有步骤执行，规划表述一般", signals
    elif num_steps == 1:
        return 5, "仅1个步骤，规划过于简单", signals
    return 3, "无任何执行步骤或规划", signals


def grade_tool_selection(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    tool_calls = extract_tool_calls(record)
    if not tool_calls:
        return 2, "未选择任何工具", []
    return 8, "有工具调用", []


def grade_parameter_generation(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    tool_calls = extract_tool_calls(record)
    observations = str(record.get("observations", ""))
    param_error_keywords = [
        "missing required parameter", "invalid parameter", "参数错误",
        "format error", "missing argument", "缺少参数", "syntax error"
    ]
    for kw in param_error_keywords:
        if kw.lower() in observations.lower():
            return 4, f"参数格式错误: {kw}", ["param_error"]
    if tool_calls:
        for tc in tool_calls:
            params = tc.get("params", {})
            if not params or params == {}:
                return 6, "部分参数为空", ["empty_params"]
    return 9, "参数格式正确", []


def grade_execution_accuracy(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    outcome = str(record.get("outcome", ""))
    if "成功" in outcome or outcome == "success" or outcome == "通过":
        return 10, "执行成功", ["execution_success"]
    elif "部分" in outcome or outcome == "partial":
        return 6, "部分执行成功", ["partial_execution"]
    return 3, "执行失败", ["execution_failed"]


def grade_reflection(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = extract_thoughts(record).lower()
    reflection_keywords = [
        "error", "mistake", "wrong", "fix", "correct", "检查", "错误", "修正",
        "纠正", "反思", "retry", "re-try", "let me check", "adjust", "modify"
    ]
    has_reflection = any(kw in thoughts for kw in reflection_keywords)
    outcome = str(record.get("outcome", ""))
    if "成功" in outcome or outcome == "success":
        return 10 if has_reflection else 9, "成功完成" + ("且有反思" if has_reflection else ""), []
    elif "部分" in outcome or outcome == "partial":
        return 7 if has_reflection else 5, "部分成功" + ("且有反思" if has_reflection else "，未见反思"), []
    return 5 if has_reflection else 2, "失败" + ("但有反思" if has_reflection else "且无反思"), []


def grade_state_tracking(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = extract_thoughts(record).lower()
    tracking_keywords = [
        "previous", "before", "already", "done", "completed", "上文", "之前",
        "已经", "完成", "history", "recall", "remember", "track", "state"
    ]
    has_tracking = any(kw in thoughts for kw in tracking_keywords)
    num_steps = len(extract_tool_calls(record)) if extract_tool_calls(record) else record.get("num_steps", 0)
    if num_steps >= 3 and has_tracking:
        return 10, "多步骤且有状态跟踪", ["tracks_state_explicitly"]
    elif num_steps >= 2:
        return 7, "有步骤执行，状态跟踪一般", ["tracks_state_implicitly"]
    return 5, "单步骤或无状态跟踪", ["single_step_no_tracking"]


def grade_termination(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    num_steps = record.get("num_steps", 0)
    outcome = str(record.get("outcome", ""))
    if ("失败" in outcome or outcome == "failure") and num_steps <= 1:
        return 4, "复杂任务仅少量步骤，可能提前终止", ["premature_termination"]
    if "成功" in outcome or outcome == "success":
        return 10, "及时终止，任务完成", ["normal_termination"]
    return 6, "终止控制情况一般", ["termination_unclear"]


GRADERS = {
    "goal_understanding": grade_goal_understanding,
    "planning": grade_planning,
    "tool_selection": grade_tool_selection,
    "parameter_generation": grade_parameter_generation,
    "execution_accuracy": grade_execution_accuracy,
    "reflection": grade_reflection,
    "state_tracking": grade_state_tracking,
    "termination": grade_termination,
}


def grade_record(record: Dict[str, Any]) -> Dict[str, Any]:
    scores = {}
    reasons = {}
    signals = []
    for dim, fn in GRADERS.items():
        score, reason, sigs = fn(record)
        scores[dim] = score
        reasons[dim] = reason
        signals.extend([f"{dim}:{s}" for s in sigs])
    return {
        "scores": scores,
        "reasons": reasons,
        "signals": signals,
        "overall": sum(scores.values()),
    }


def evaluate_batch(trajectories: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """Score a batch of trajectories. Returns overall_avg, dim_avg, scored_records."""
    scored = []
    dim_sums = {d: 0.0 for d in DIMENSIONS}
    for traj in trajectories:
        graded = grade_record(traj)
        scored.append({**traj, "graded": graded})
        for d in DIMENSIONS:
            dim_sums[d] += graded["scores"][d]
    n = len(trajectories)
    overall_avg = sum(r["graded"]["overall"] for r in scored) / n if n else 0
    dim_avg = {d: dim_sums[d] / n if n else 0 for d in DIMENSIONS}
    return overall_avg, dim_avg, scored


def compute_fitness(population: List[PromptCandidate], batch: List[Dict[str, Any]]) -> List[PromptCandidate]:
    """计算 prompt candidate 种群的 fitness"""
    overall_avg, dim_avg, scored = evaluate_batch(batch)
    all_signals = []
    for r in scored:
        all_signals.extend(r["graded"]["signals"])
    signal_freq = Counter(all_signals)
    top_signals = [s for s, _ in signal_freq.most_common(6)]
    weak_dims = [d for d, v in sorted(dim_avg.items(), key=lambda x: x[1])[:3]]

    synonyms = {
        "goal_understanding": ["goal", "objective", "target", "understand"],
        "planning": ["plan", "step", "sub-task", "decompose", "拆解"],
        "reflection": ["reflect", "correct", "error", "check", "fix"],
        "state_tracking": ["track", "state", "history", "summary", "context"],
        "termination": ["terminate", "stop", "finish", "verify", "complete"],
        "tool_selection": ["tool", "select", "choose", "match"],
        "parameter_generation": ["parameter", "argument", "json", "format"],
        "execution_accuracy": ["execute", "accuracy", "verify", "result"],
    }

    for candidate in population:
        base = overall_avg
        prompt_lower = candidate.prompt.lower()
        bonus = 0.0

        for sig in top_signals:
            parts = sig.split(":")
            dim_part = parts[0] if parts else sig
            if dim_part in prompt_lower:
                bonus += 1.5
            for syn in synonyms.get(dim_part, []):
                if syn in prompt_lower:
                    bonus += 0.8
                    break

        rule_count = len(re.findall(r"(?i)^\s*\d+\.|^\s*-\s+|^\s*\*\s+", candidate.prompt, re.M))
        bonus += min(rule_count * 0.5, 3.0)
        penalty = max(0, (candidate.prompt_length - 400) * 0.01)

        candidate.fitness = base + bonus - penalty
        candidate.dim_scores = dim_avg
        candidate.reflective_feedback = f"weak_dims={weak_dims}; top_signals={top_signals}; bonus={bonus:.1f}"

    return population
