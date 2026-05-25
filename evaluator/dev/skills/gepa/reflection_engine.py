#!/usr/bin/env python3
"""
GEPA ReflectionEngine — 基于失败轨迹生成弱点报告
位置: evaluator/dev/skills/gepa/
输入: Agent 失败轨迹 + 8维评估结果
输出: JSON 格式的 reflection_report
"""
import json
import os
import textwrap
from collections import Counter
from typing import Dict, Any, List

from optimizer.dev.skills.gepa.fitness_evaluator import extract_thoughts


class ReflectionEngine:
    """
    从评分后的轨迹中抽取失败模式，生成反射报告。
    Mode = 'rule' 使用启发式规则; mode = 'llm' 调用反射模型（预留接口）。
    """

    def __init__(self, mode: str = "rule"):
        self.mode = mode
        self._client = None
        if mode == "llm":
            self._init_llm()

    def _init_llm(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")
            if api_key:
                self._client = OpenAI(base_url=base_url, api_key=api_key)
                print("[ReflectionEngine] LLM client initialized.")
            else:
                print("[ReflectionEngine] No API key found; falling back to rule mode.")
                self.mode = "rule"
        except Exception as e:
            print(f"[ReflectionEngine] LLM init failed ({e}); using rule mode.")
            self.mode = "rule"

    def reflect(self, scored_records: List[Dict[str, Any]], dim_avg: Dict[str, float]) -> str:
        """返回描述失败和模式的反射反馈文本。"""
        if self.mode == "llm" and self._client:
            return self._reflect_llm(scored_records, dim_avg)
        return self._reflect_rule(scored_records, dim_avg)

    def _reflect_rule(self, scored_records: List[Dict[str, Any]], dim_avg: Dict[str, float]) -> str:
        # 识别最弱的维度
        weakest_dims = sorted(dim_avg.items(), key=lambda x: x[1])[:3]
        lines = ["=== Reflective Dataset ===", ""]
        lines.append(f"Weakest dimensions: {', '.join(f'{d}={s:.1f}' for d,s in weakest_dims)}")
        lines.append("")

        # 按最弱维度收集失败样例
        for dim, avg in weakest_dims:
            lines.append(f"--- {dim} failures (avg={avg:.1f}) ---")
            failures = [r for r in scored_records if r["graded"]["scores"][dim] <= 5]
            for i, rec in enumerate(failures[:3], 1):
                lines.append(f"  [{i}] Task: {rec.get('goal', rec.get('question', 'N/A'))[:80]}...")
                lines.append(f"      Score: {rec['graded']['scores'][dim]}, Reason: {rec['graded']['reasons'][dim]}")
                if rec["graded"]["signals"]:
                    rel = [s for s in rec["graded"]["signals"] if s.startswith(dim)]
                    if rel:
                        lines.append(f"      Signals: {', '.join(rel[:3])}")
            lines.append("")

        # 抽象失败模式统计
        lines.append("--- Failure Patterns ---")
        all_signals = []
        for r in scored_records:
            all_signals.extend(r["graded"]["signals"])
        freq = Counter(all_signals)
        for sig, cnt in freq.most_common(8):
            lines.append(f"  {sig}: {cnt} occurrences")
        lines.append("")

        # 根因假设（规则生成）
        lines.append("--- Root-Cause Hypotheses ---")
        if dim_avg.get("goal_understanding", 10) < 6:
            lines.append("  - Agent does not explicitly parse goal keywords before acting.")
        if dim_avg.get("planning", 10) < 6:
            lines.append("  - No explicit sub-task decomposition in thoughts.")
        if dim_avg.get("reflection", 10) < 6:
            lines.append("  - No self-correction signals when errors occur in observations.")
        if dim_avg.get("state_tracking", 10) < 6:
            lines.append("  - Agent does not reference prior steps or context.")
        if dim_avg.get("termination", 10) < 6:
            lines.append("  - Premature termination on complex tasks.")
        lines.append("")

        return "\n".join(lines)

    def _reflect_llm(self, scored_records: List[Dict[str, Any]], dim_avg: Dict[str, float]) -> str:
        weakest = sorted(dim_avg.items(), key=lambda x: x[1])[:3]
        failures = []
        for r in scored_records:
            low_dims = [d for d, s in r["graded"]["scores"].items() if s <= 5]
            if low_dims:
                failures.append({
                    "task": r.get("goal", r.get("question", "N/A"))[:120],
                    "low_dims": low_dims,
                    "scores": r["graded"]["scores"],
                    "thoughts": extract_thoughts(r)[:200],
                    "observations": str(r.get("observations", ""))[:200],
                })

        prompt = textwrap.dedent(f"""\
            You are an expert prompt engineer. Analyze the following agent execution failures and identify root causes.

            Average dimension scores: {json.dumps(dim_avg, ensure_ascii=False)}
            Weakest dimensions: {json.dumps([d for d,_ in weakest], ensure_ascii=False)}

            Failure cases (up to 5):
            {json.dumps(failures[:5], ensure_ascii=False, indent=2)}

            Instructions:
            1. Identify 2-3 specific failure patterns.
            2. For each pattern, explain WHY the current system prompt is insufficient.
            3. Propose concrete prompt-level fixes (e.g., add a rule, change wording, add a step).
            4. Output ONLY a structured reflective report.
        """)
        try:
            r = self._client.chat.completions.create(
                model="kimi-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500,
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"[ReflectionEngine] LLM reflection failed ({e}), falling back to rule.")
            return self._reflect_rule(scored_records, dim_avg)

    def generate_report(self, scored_records: List[Dict[str, Any]], dim_avg: Dict[str, float]) -> Dict[str, Any]:
        """生成结构化的 reflection_report JSON"""
        weakest = sorted(dim_avg.items(), key=lambda x: x[1])[:3]
        all_signals = []
        for r in scored_records:
            all_signals.extend(r["graded"]["signals"])
        freq = Counter(all_signals)

        dim_failures = {}
        for dim, avg in dim_avg.items():
            failures = [r for r in scored_records if r["graded"]["scores"][dim] <= 5]
            dim_failures[dim] = {
                "avg_score": round(avg, 2),
                "failure_count": len(failures),
                "top_reasons": list(set(r["graded"]["reasons"][dim] for r in failures))[:3]
            }

        return {
            "weakest_dimensions": [{"dim": d, "avg_score": round(s, 2)} for d, s in weakest],
            "failure_signals": dict(freq.most_common(10)),
            "dimension_analysis": dim_failures,
            "reflection_text": self.reflect(scored_records, dim_avg),
        }
