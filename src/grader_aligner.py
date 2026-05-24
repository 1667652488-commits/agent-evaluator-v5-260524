#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grader_aligner.py — Rule vs LLM 评估器对齐器
=============================================
对比 RuleGrader 与 LLMGrader 的输出，计算维度级一致性，
定位差异最大维度，输出调参建议。

输入:
  - rule_scores.jsonl (RuleGrader 输出，含 scores 字段)
  - llm_scores.jsonl   (LLMGrader 输出，含 llm_scores 字段)

输出:
  - alignment_report.json  (对齐报告: 相关性 + 差异分析 + 调参建议)
  - alignment_report.txt   (可读版本)

2025-05-23 | 冷灯
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# ------------------------------------------------------------------
# 数据加载
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


def merge_scores(rule_records: List[Dict], llm_records: List[Dict]) -> List[Dict]:
    """按 id 合并 Rule 和 LLM 评分，只保留两边都存在的记录。"""
    rule_by_id = {str(r.get("id", i)): r for i, r in enumerate(rule_records)}
    llm_by_id = {str(r.get("id", i)): r for i, r in enumerate(llm_records)}
    common_ids = sorted(set(rule_by_id.keys()) & set(llm_by_id.keys()))

    merged = []
    for rid in common_ids:
        rule_rec = rule_by_id[rid]
        llm_rec = llm_by_id[rid]
        merged.append({
            "id": rid,
            "goal": rule_rec.get("goal", rule_rec.get("question", "")),
            "rule_scores": rule_rec.get("scores", {}),
            "llm_scores": llm_rec.get("llm_scores", {}),
            "rule_total": rule_rec.get("overall", 0),
            "llm_total": llm_rec.get("llm_total", sum(llm_rec.get("llm_scores", {}).values())),
        })
    return merged


# ------------------------------------------------------------------
# 统计计算
# ------------------------------------------------------------------

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std_dev(vals: List[float]) -> float:
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if vals else 0.0


def pearson_r(x: List[float], y: List[float]) -> float:
    """计算 Pearson 相关系数。"""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = std_dev(x), std_dev(y)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    return cov / (sx * sy)


def spearman_r(x: List[float], y: List[float]) -> float:
    """计算 Spearman 秩相关系数 (近似版)。"""
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0
    # 计算秩
    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0] * len(vals)
        for i, (idx, _) in enumerate(sorted_vals):
            ranks[idx] = i + 1
        return ranks
    rx, ry = rank(x), rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def compute_dim_alignment(merged: List[Dict], dim: str) -> Dict[str, Any]:
    """计算单个维度的一致性指标。"""
    rule_vals = [m["rule_scores"].get(dim, 0) for m in merged]
    llm_vals = [m["llm_scores"].get(dim, 0) for m in merged]

    # 只保留两边都有有效分数的记录
    valid_pairs = [(r, l) for r, l in zip(rule_vals, llm_vals) if r is not None and l is not None]
    if not valid_pairs:
        return {"error": "无有效数据"}

    rv, lv = zip(*valid_pairs)
    rv, lv = list(rv), list(lv)
    n = len(rv)

    diffs = [r - l for r, l in zip(rv, lv)]
    abs_diffs = [abs(d) for d in diffs]

    return {
        "n": n,
        "rule_mean": round(mean(rv), 2),
        "llm_mean": round(mean(lv), 2),
        "mean_diff": round(mean(diffs), 2),
        "mae": round(mean(abs_diffs), 2),          # Mean Absolute Error
        "rmse": round(math.sqrt(mean([d ** 2 for d in diffs])), 2),
        "pearson_r": round(pearson_r(rv, lv), 3),
        "spearman_r": round(spearman_r(rv, lv), 3),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0,
        "diff_distribution": {
            "diff_0": sum(1 for d in abs_diffs if d == 0),
            "diff_1": sum(1 for d in abs_diffs if d == 1),
            "diff_2": sum(1 for d in abs_diffs if d == 2),
            "diff_3+": sum(1 for d in abs_diffs if d >= 3),
        },
    }


# ------------------------------------------------------------------
# 调参建议生成
# ------------------------------------------------------------------

def generate_tuning_advice(dim: str, stats: Dict[str, Any]) -> List[str]:
    """基于对齐统计生成 RuleGrader 的调参建议。"""
    advice = []
    cn = DIMENSION_NAMES_CN.get(dim, dim)

    if "error" in stats:
        return [f"{cn}: 数据不足，无法生成建议"]

    mae = stats.get("mae", 0)
    mean_diff = stats.get("mean_diff", 0)
    pearson = stats.get("pearson_r", 0)

    # 一致性分级
    if pearson >= 0.7 and mae <= 1.5:
        advice.append(f"{cn}: 一致性高 (r={pearson}, MAE={mae})，RuleGrader 可独立承担此维度评估。")
    elif pearson >= 0.5 and mae <= 2.5:
        advice.append(f"{cn}: 一致性中等 (r={pearson}, MAE={mae})，RuleGrader 可用但建议抽样 LLM 复核。")
    else:
        advice.append(f"{cn}: 一致性低 (r={pearson}, MAE={mae})，此维度建议走 LLMGrader 或 Hybrid 模式。")

    # 系统性偏差方向
    if mean_diff > 1.5:
        advice.append(f"  → RuleGrader 系统性偏高 (+{mean_diff:.1f}分)，建议下调该维度阈值/权重。")
    elif mean_diff < -1.5:
        advice.append(f"  → RuleGrader 系统性偏低 ({mean_diff:.1f}分)，建议上调该维度 bonus 或降低 penalty。")

    # 具体维度的针对性建议
    if dim == "reflection_correction" and (pearson < 0.5 or mae > 2.0):
        advice.append(f"  → reflection_correction 差异大: Rule 只看关键词出现，LLM 判断'反思是否导致行为改变'。")
        advice.append(f"    建议: 增加行为改变检测(如'retry后结果变化')，而非仅统计关键词。")

    if dim == "goal_understanding" and (pearson < 0.5 or mae > 2.0):
        advice.append(f"  → goal_understanding 差异大: Rule 用关键词覆盖率，LLM 理解语义匹配。")
        advice.append(f"    建议: 引入语义相似度(embedding)替代简单关键词匹配。")

    if dim == "planning" and (pearson < 0.5 or mae > 2.0):
        advice.append(f"  → planning 差异大: Rule 数步骤和关键词，LLM 判断逻辑合理性。")
        advice.append(f"    建议: 增加步骤间依赖关系检测(如'搜索后才能点击')。")

    if dim == "state_tracking" and (pearson < 0.5 or mae > 2.0):
        advice.append(f"  → state_tracking 差异大: Rule 检测重复调用，LLM 判断上下文利用。")
        advice.append(f"    建议: 增加先前步骤引用检测(如'上一步已获取XX')。")

    if dim in ("tool_selection", "argument_generation") and pearson >= 0.6:
        advice.append(f"  → {cn} 一致性良好: 格式/规则类维度适合 RuleGrader，可保持当前策略。")

    return advice


# ------------------------------------------------------------------
# 报告生成
# ------------------------------------------------------------------

def build_report(merged: List[Dict]) -> Dict[str, Any]:
    report = {
        "sample_count": len(merged),
        "dimensions": {},
        "overall": {},
        "tuning_advice": [],
        "hybrid_strategy": {},
    }

    # 总分对齐
    rule_totals = [m["rule_total"] for m in merged]
    llm_totals = [m["llm_total"] for m in merged]
    report["overall"] = {
        "rule_mean": round(mean(rule_totals), 2),
        "llm_mean": round(mean(llm_totals), 2),
        "mean_diff": round(mean([r - l for r, l in zip(rule_totals, llm_totals)]), 2),
        "mae": round(mean([abs(r - l) for r, l in zip(rule_totals, llm_totals)]), 2),
        "pearson_r": round(pearson_r(rule_totals, llm_totals), 3),
        "spearman_r": round(spearman_r(rule_totals, llm_totals), 3),
    }

    # 逐维度对齐
    for dim in DIMENSIONS:
        stats = compute_dim_alignment(merged, dim)
        report["dimensions"][dim] = stats
        advice = generate_tuning_advice(dim, stats)
        report["tuning_advice"].extend(advice)

    # Hybrid 维度分配策略
    for dim in DIMENSIONS:
        stats = report["dimensions"][dim]
        pearson = stats.get("pearson_r", 0)
        mae = stats.get("mae", 10)
        if pearson >= 0.7 and mae <= 1.5:
            strategy = "rule"           # Rule 足够可靠
        elif pearson >= 0.4 and mae <= 2.5:
            strategy = "hybrid"         # 两者并行，差异大时以 LLM 为准
        else:
            strategy = "llm"            # Rule 不可靠，走 LLM
        report["hybrid_strategy"][dim] = {
            "strategy": strategy,
            "pearson_r": pearson,
            "mae": mae,
            "reason": f"r={pearson}, MAE={mae}",
        }

    return report


def format_report_text(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("RuleGrader vs LLMGrader 对齐报告")
    lines.append("=" * 70)
    lines.append(f"样本数: {report['sample_count']}")
    lines.append("")

    ov = report["overall"]
    lines.append("【总分对齐】")
    lines.append(f"  Rule 平均分: {ov['rule_mean']}  |  LLM 平均分: {ov['llm_mean']}")
    lines.append(f"  平均差异: {ov['mean_diff']}  |  MAE: {ov['mae']}")
    lines.append(f"  Pearson r: {ov['pearson_r']}  |  Spearman r: {ov['spearman_r']}")
    lines.append("")

    lines.append("【逐维度对齐】")
    for dim in DIMENSIONS:
        cn = DIMENSION_NAMES_CN.get(dim, dim)
        st = report["dimensions"][dim]
        if "error" in st:
            lines.append(f"  {cn}: {st['error']}")
            continue
        lines.append(f"  {cn}:")
        lines.append(f"    Rule均值={st['rule_mean']} LLM均值={st['llm_mean']} "
                     f"差异={st['mean_diff']:+.1f} MAE={st['mae']} "
                     f"r={st['pearson_r']}")
        dd = st["diff_distribution"]
        lines.append(f"    差异分布: 0分={dd['diff_0']} 1分={dd['diff_1']} "
                     f"2分={dd['diff_2']} 3+分={dd['diff_3+']}")
    lines.append("")

    lines.append("【调参建议】")
    for adv in report["tuning_advice"]:
        lines.append(f"  {adv}")
    lines.append("")

    lines.append("【Hybrid 维度分配策略】")
    for dim in DIMENSIONS:
        cn = DIMENSION_NAMES_CN.get(dim, dim)
        st = report["hybrid_strategy"][dim]
        flag = {"rule": "[R]", "hybrid": "[H]", "llm": "[L]"}[st["strategy"]]
        lines.append(f"  {flag} {cn}: {st['strategy'].upper()}  ({st['reason']})")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule vs LLM 评估器对齐器")
    parser.add_argument("--rule", required=True, help="RuleGrader 输出 JSONL")
    parser.add_argument("--llm", required=True, help="LLMGrader 输出 JSONL")
    parser.add_argument("--output", default="./0523task/alignment_output", help="输出目录")
    args = parser.parse_args()

    print(f"[INFO] 加载 RuleGrader: {args.rule}")
    rule_records = load_jsonl(args.rule)
    print(f"[INFO] 加载 LLMGrader: {args.llm}")
    llm_records = load_jsonl(args.llm)

    print(f"[INFO] Rule 记录数: {len(rule_records)} | LLM 记录数: {len(llm_records)}")

    merged = merge_scores(rule_records, llm_records)
    print(f"[INFO] 可对齐记录数: {len(merged)}")

    if not merged:
        print("[ERR] 无共同记录，退出")
        return

    report = build_report(merged)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "alignment_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON 报告 → {json_path}")

    txt_path = out_dir / "alignment_report.txt"
    txt_path.write_text(format_report_text(report), encoding="utf-8")
    print(f"[INFO] 文本报告 → {txt_path}")

    # 打印摘要
    print("\n" + format_report_text(report))


if __name__ == "__main__":
    main()
