#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_grader.py — 融合评估器 (Rule + LLM Hybrid)
=================================================
结合 RuleGrader（零成本、速度快）和 LLMGrader（语义判断、柔性），
按维度分配策略动态选择评估方式。

支持两种模式:
- 固定策略: 事前指定每个维度用 Rule 或 LLM
- 动态降级: 若某维度 Rule/LLM 历史一致性低，自动切到 LLM

输入: JSONL 轨迹文件
输出: JSONL 统一 8维评分 (scores + sources 标注每条分数来自 rule/llm/hybrid)

2025-05-23 | 冷灯
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------

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

# 默认固定策略: reflection + goal_understanding 走 LLM，tool + arg 走 Rule，其余动态
DEFAULT_FIXED_STRATEGY = {
    "goal_understanding": "llm",
    "planning": "hybrid",
    "tool_selection": "rule",
    "argument_generation": "rule",
    "execution_accuracy": "hybrid",
    "reflection_correction": "llm",
    "state_tracking": "hybrid",
    "termination_control": "hybrid",
}

# LLM 配置 (与 llm_grader.py 一致)
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
API_BASE = "https://api.siliconflow.cn/v1"
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-nnksashvwdizsenvqlnlcyhevvzpqntwswvutxcqukzfhkyc")


# ------------------------------------------------------------------
# 导入 RuleGrader (从 agent_grader_v2.py 复用)
# ------------------------------------------------------------------

# 为避免文件路径依赖，把 agent_grader_v2 的核心 grading 函数内联进来
# 这些函数与 agent_grader_v2.py 中的实现一致

import re
from collections import Counter


def _extract_thoughts(record: Dict[str, Any]) -> str:
    thoughts = record.get("thoughts", "")
    if isinstance(thoughts, list):
        return "\n".join(str(t) for t in thoughts)
    return str(thoughts) if thoughts else ""


def _extract_tool_calls(record: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _grade_goal_understanding(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = _extract_thoughts(record)
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


def _grade_planning(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = _extract_thoughts(record).lower()
    tool_calls = _extract_tool_calls(record)
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


def _grade_tool_selection(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    tool_calls = _extract_tool_calls(record)
    if not tool_calls:
        return 2, "未选择任何工具", []
    return 8, "有工具调用", []


def _grade_argument_generation(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    tool_calls = _extract_tool_calls(record)
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


def _grade_execution_accuracy(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    outcome = str(record.get("outcome", ""))
    if "成功" in outcome or outcome == "success" or outcome == "通过":
        return 10, "执行成功", ["execution_success"]
    elif "部分" in outcome or outcome == "partial":
        return 6, "部分执行成功", ["partial_execution"]
    return 3, "执行失败", ["execution_failed"]


def _grade_reflection(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = _extract_thoughts(record).lower()
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


def _grade_state_tracking(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    thoughts = _extract_thoughts(record).lower()
    tracking_keywords = [
        "previous", "before", "already", "done", "completed", "上文", "之前",
        "已经", "完成", "history", "recall", "remember", "track", "state"
    ]
    has_tracking = any(kw in thoughts for kw in tracking_keywords)
    num_steps = len(_extract_tool_calls(record)) if _extract_tool_calls(record) else record.get("num_steps", 0)
    if num_steps >= 3 and has_tracking:
        return 10, "多步骤且有状态跟踪", ["tracks_state_explicitly"]
    elif num_steps >= 2:
        return 7, "有步骤执行，状态跟踪一般", ["tracks_state_implicitly"]
    return 5, "单步骤或无状态跟踪", ["single_step_no_tracking"]


def _grade_termination(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    num_steps = record.get("num_steps", 0)
    outcome = str(record.get("outcome", ""))
    if ("失败" in outcome or outcome == "failure") and num_steps <= 1:
        return 4, "复杂任务仅少量步骤，可能提前终止", ["premature_termination"]
    if "成功" in outcome or outcome == "success":
        return 10, "及时终止，任务完成", ["normal_termination"]
    return 6, "终止控制情况一般", ["termination_unclear"]


_RULE_GRADERS = {
    "goal_understanding": _grade_goal_understanding,
    "planning": _grade_planning,
    "tool_selection": _grade_tool_selection,
    "argument_generation": _grade_argument_generation,
    "execution_accuracy": _grade_execution_accuracy,
    "reflection_correction": _grade_reflection,
    "state_tracking": _grade_state_tracking,
    "termination_control": _grade_termination,
}


def rule_grade(record: Dict[str, Any]) -> Dict[str, Any]:
    """RuleGrader 内联实现，返回统一格式。"""
    scores = {}
    reasons = {}
    signals = []
    for dim, fn in _RULE_GRADERS.items():
        score, reason, sigs = fn(record)
        scores[dim] = score
        reasons[dim] = reason
        signals.extend([f"{dim}:{s}" for s in sigs])
    return {
        "scores": scores,
        "reasons": reasons,
        "signals": signals,
        "overall": sum(scores.values()),
        "source": "rule",
    }


# ------------------------------------------------------------------
# LLM Grader 轻量调用 (仅评估指定维度)
# ------------------------------------------------------------------

def _build_dim_prompt(record: Dict[str, Any], dims: List[str]) -> str:
    """构造仅评估指定维度的轻量 prompt，减少 token 消耗。"""
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
    outcome = record.get("outcome", "N/A")
    num_steps = record.get("num_steps", 0)

    # 截断
    MAX = 600
    thoughts = (thoughts[:MAX] + "...[截断]") if len(str(thoughts)) > MAX else thoughts
    observations = (observations[:MAX] + "...[截断]") if len(str(observations)) > MAX else observations

    dim_rubrics = {
        "goal_understanding": "10分: 明确复述目标关键实体和约束。1-3分: 严重偏离。",
        "planning": "10分: 明确 todo list，步骤逻辑合理。1-3分: 无规划直接执行。",
        "tool_selection": "10分: 工具与目标精确匹配。1-3分: 严重错误或未调用。",
        "argument_generation": "10分: 参数格式正确完整。1-3分: 严重错误导致无法执行。",
        "execution_accuracy": "10分: 结果与标准答案一致。1-3分: 严重偏离。",
        "reflection_correction": "10分: 错误时主动识别并修正且有效。8分: 成功且无错误场景。1-3分: 失败且无反思。",
        "state_tracking": "10分: 持续关注已完成/待完成事项。1-3分: 严重遗忘或重复。",
        "termination_control": "10分: 及时终止输出完整。1-3分: 严重过早或过晚。",
    }

    rubric_text = "\n".join(f"{DIMENSION_NAMES_CN[d]} ({d}): {dim_rubrics[d]}" for d in dims)

    prompt = f"""评估以下 Agent 轨迹的指定维度。

目标: {goal}
结果: {outcome}
步数: {num_steps}

思考: {thoughts}
工具: {tool_calls}
反馈: {observations}
输出: {output}

【仅评估以下维度，每项 0-10 分整数】
{rubric_text}

输出 STRICT JSON:
{{
  "scores": {{{", ".join(f'"{d}": 0' for d in dims)}}},
  "reasons": {{{", ".join(f'"{d}": "理由"' for d in dims)}}}
}}
只输出 JSON，无 Markdown。
"""
    return prompt


def _get_client():
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE, api_key=API_KEY)
    except Exception as e:
        print(f"[ERR] OpenAI client 失败: {e}", file=sys.stderr)
        raise


def llm_grade_dims(record: Dict[str, Any], dims: List[str], client=None) -> Tuple[Dict[str, Any], float, int, int]:
    """对指定维度调用 LLM 评分，返回 (result, elapsed, prompt_tokens, completion_tokens)。"""
    if client is None:
        client = _get_client()

    prompt = _build_dim_prompt(record, dims)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        elapsed = time.time() - start
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        result = json.loads(content)

        usage = resp.usage
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0

        scores = result.get("scores", {})
        reasons = result.get("reasons", {})
        # 校验
        for d in dims:
            if d not in scores or not isinstance(scores[d], (int, float)):
                scores[d] = 0
            else:
                scores[d] = int(max(0, min(10, scores[d])))
            if d not in reasons:
                reasons[d] = ""

        return {"scores": scores, "reasons": reasons}, elapsed, pt, ct

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [WARN] LLM 调用失败: {e}", file=sys.stderr)
        # 失败时返回空，调用方应 fallback
        return {}, elapsed, 0, 0


# ------------------------------------------------------------------
# Hybrid Grader
# ------------------------------------------------------------------

class HybridGrader:
    """
    融合评估器。
    - fixed_strategy: Dict[dim, "rule"|"llm"|"hybrid"]
    - dynamic_fallback: 若 hybrid 维度 Rule/LLM 差异 > threshold，以 LLM 为准
    """

    def __init__(
        self,
        fixed_strategy: Optional[Dict[str, str]] = None,
        dynamic_fallback: bool = True,
        fallback_threshold: float = 3.0,  # 分数差异超过此值触发 fallback
        llm_dims_per_call: int = 4,       # 每次 LLM 调用评估几个维度，减少请求数
    ):
        self.strategy = fixed_strategy or DEFAULT_FIXED_STRATEGY.copy()
        self.dynamic_fallback = dynamic_fallback
        self.fallback_threshold = fallback_threshold
        self.llm_dims_per_call = llm_dims_per_call
        self.client = None
        self._init_llm()
        self.stats = {
            "rule_calls": 0,
            "llm_calls": 0,
            "llm_tokens_prompt": 0,
            "llm_tokens_completion": 0,
            "llm_elapsed": 0.0,
            "fallbacks": 0,
        }

    def _init_llm(self):
        try:
            self.client = _get_client()
            print("[HybridGrader] LLM client 初始化成功")
        except Exception:
            self.client = None
            print("[HybridGrader] LLM client 初始化失败，LLM 维度将返回 0")

    def grade(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """对单条记录进行 Hybrid 评分。"""
        scores = {}
        reasons = {}
        sources = {}  # 每个维度的来源: rule / llm / hybrid

        # 先计算 Rule 全维度 (零成本，作为 baseline)
        rule_result = rule_grade(record)
        self.stats["rule_calls"] += 1

        # 收集需要 LLM 评估的维度
        llm_dims = []
        for dim in DIMENSIONS:
            strategy = self.strategy.get(dim, "hybrid")
            if strategy == "rule":
                scores[dim] = rule_result["scores"][dim]
                reasons[dim] = rule_result["reasons"][dim]
                sources[dim] = "rule"
            elif strategy == "llm":
                llm_dims.append(dim)
            elif strategy == "hybrid":
                # hybrid: 先取 rule，后续可能与 LLM 对比
                scores[dim] = rule_result["scores"][dim]
                reasons[dim] = rule_result["reasons"][dim]
                sources[dim] = "rule"
                # 如果启用动态降级，也加入 LLM 待评估列表
                if self.dynamic_fallback:
                    llm_dims.append(dim)

        # 调用 LLM 评估需要 LLM 的维度 (分批，减少 API 调用)
        if llm_dims and self.client:
            # 分批
            for i in range(0, len(llm_dims), self.llm_dims_per_call):
                batch_dims = llm_dims[i:i + self.llm_dims_per_call]
                llm_result, elapsed, pt, ct = llm_grade_dims(record, batch_dims, self.client)
                self.stats["llm_calls"] += 1
                self.stats["llm_elapsed"] += elapsed
                self.stats["llm_tokens_prompt"] += pt
                self.stats["llm_tokens_completion"] += ct

                if not llm_result:
                    # LLM 失败，保持 rule 分数 (已设置)
                    continue

                for dim in batch_dims:
                    if dim not in llm_result.get("scores", {}):
                        continue
                    llm_score = llm_result["scores"][dim]
                    rule_score = rule_result["scores"].get(dim, 0)

                    # 决策逻辑
                    strategy = self.strategy.get(dim, "hybrid")
                    if strategy == "llm":
                        scores[dim] = llm_score
                        reasons[dim] = llm_result["reasons"].get(dim, "LLM评估")
                        sources[dim] = "llm"
                    elif strategy == "hybrid" and self.dynamic_fallback:
                        diff = abs(rule_score - llm_score)
                        if diff >= self.fallback_threshold:
                            # 差异大，以 LLM 为准 (假设 LLM 语义判断更准)
                            scores[dim] = llm_score
                            reasons[dim] = f"[LLM override] Rule={rule_score}, LLM={llm_score}. " + llm_result["reasons"].get(dim, "")
                            sources[dim] = "hybrid-llm"
                            self.stats["fallbacks"] += 1
                        else:
                            # 差异小，信任 Rule (零成本)
                            scores[dim] = rule_score
                            reasons[dim] = f"[Rule] 与LLM一致(diff={diff}). " + rule_result["reasons"][dim]
                            sources[dim] = "hybrid-rule"

        # 对未获得 LLM 分数的 llm 维度 (client 失败时)，保留 rule 并标记
        for dim in llm_dims:
            if sources.get(dim) in ("rule", None):
                scores[dim] = rule_result["scores"][dim]
                reasons[dim] = "[Rule fallback] LLM unavailable. " + rule_result["reasons"][dim]
                sources[dim] = "rule-fallback"

        overall = sum(scores.values())
        return {
            "scores": scores,
            "reasons": reasons,
            "sources": sources,
            "overall": overall,
            "rule_baseline": rule_result["scores"],
        }

    def batch_grade(self, records: List[Dict[str, Any]], progress_every: int = 10) -> List[Dict[str, Any]]:
        """批量评分。"""
        results = []
        for i, rec in enumerate(records):
            result = self.grade(rec)
            result["id"] = rec.get("id", i)
            result["goal"] = rec.get("goal", rec.get("question", ""))
            results.append(result)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"[Progress] 已评估 {i+1}/{len(records)} | LLM calls={self.stats['llm_calls']} | fallbacks={self.stats['fallbacks']}")
        return results


# ------------------------------------------------------------------
# 数据 I/O
# ------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    p = Path(path)
    if not p.exists():
        print(f"[ERR] 文件不存在: {path}", file=sys.stderr)
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


def save_json(data: Any, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hybrid Grader (Rule + LLM)")
    parser.add_argument("--input", required=True, help="输入 JSONL 轨迹文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 评分文件")
    parser.add_argument("--strategy", default="", help="维度策略 JSON 文件 (可选，默认内置策略)")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数")
    parser.add_argument("--threshold", type=float, default=3.0, help="动态降级阈值 (默认3.0)")
    parser.add_argument("--llm-batch", type=int, default=4, help="每次 LLM 调用评估维度数 (默认4)")
    args = parser.parse_args()

    # 加载策略
    strategy = None
    if args.strategy:
        with open(args.strategy, "r", encoding="utf-8") as f:
            strategy = json.load(f)
        print(f"[INFO] 加载外部策略: {args.strategy}")
    else:
        print(f"[INFO] 使用默认策略: reflection+goal→LLM, tool+arg→Rule, 其余→Hybrid")

    # 加载数据
    records = load_jsonl(args.input)
    print(f"[INFO] 加载 {len(records)} 条记录")
    if args.limit > 0:
        records = records[:args.limit]
        print(f"[INFO] 限制处理前 {args.limit} 条")

    # 初始化 Hybrid Grader
    grader = HybridGrader(
        fixed_strategy=strategy,
        dynamic_fallback=True,
        fallback_threshold=args.threshold,
        llm_dims_per_call=args.llm_batch,
    )

    # 批量评分
    results = grader.batch_grade(records, progress_every=10)

    # 保存结果
    save_jsonl(results, args.output)

    # 保存统计
    stats = grader.stats
    stats["avg_llm_time_per_call"] = round(stats["llm_elapsed"] / stats["llm_calls"], 2) if stats["llm_calls"] else 0
    stats["total_records"] = len(records)
    stats["strategy"] = strategy or DEFAULT_FIXED_STRATEGY
    stats["threshold"] = args.threshold
    stats_path = str(Path(args.output).with_suffix("")) + "_stats.json"
    save_json(stats, stats_path)

    # 打印摘要
    print(f"\n{'='*60}")
    print("Hybrid Grader 完成")
    print(f"  总记录: {stats['total_records']}")
    print(f"  Rule 调用: {stats['rule_calls']} (零成本)")
    print(f"  LLM 调用: {stats['llm_calls']} 次")
    print(f"  LLM 总耗时: {stats['llm_elapsed']:.1f}s | 平均每次: {stats['avg_llm_time_per_call']}s")
    print(f"  LLM prompt tokens: {stats['llm_tokens_prompt']}")
    print(f"  LLM completion tokens: {stats['llm_tokens_completion']}")
    print(f"  动态降级次数: {stats['fallbacks']}")
    print(f"  输出: {args.output}")
    print(f"  统计: {stats_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
