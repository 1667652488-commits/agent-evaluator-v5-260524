#!/usr/bin/env python3
"""
GEPA (Genetic Pareto) Prompt Optimizer for Agent System
=========================================================
Implements the GEPA evolutionary loop:
  1. EVALUATE  — score prompt candidates on mini-batch
  2. REFLECT   — build reflective dataset from failures
  3. PROPOSE   — LLM-guided mutation based on feedback
  4. SELECT    — Pareto-aware candidate selection
  5. REPEAT    — until budget exhaustion

Designed to replace the simple rule-based optimizer in the agent flywheel.
Uses 8-dimension grader (from agent_grader_v2) as fitness function.

2025-05-23  |  cold-lantern
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# Config & Constants
# ------------------------------------------------------------------

DIMENSIONS = [
    "goal_understanding",
    "planning",
    "tool_selection",
    "parameter_generation",
    "execution_accuracy",
    "reflection",
    "state_tracking",
    "termination",
]

SEED_PROMPT = textwrap.dedent("""\
    You are an autonomous agent. Complete the given task step by step.
    Rules:
    1. Analyze the goal before acting.
    2. Break complex tasks into sub-tasks.
    3. Choose the right tool for each step.
    4. Track your progress and reflect on errors.
    5. Stop when the task is complete.
""")

# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] File not found: {path}", file=sys.stderr)
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
# Grader (inline copy from agent_evaluator.py for self-contained use)
# ------------------------------------------------------------------

def extract_thoughts(record: Dict[str, Any]) -> str:
    thoughts = record.get("thoughts", "")
    if isinstance(thoughts, list):
        return "\n".join(str(t) for t in thoughts)
    return str(thoughts) if thoughts else ""


def extract_tool_calls(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = record.get("tool_calls", [])
    if isinstance(tool_calls, list):
        return tool_calls
    # Fallback: parse pipe-separated string
    if isinstance(tool_calls, str):
        calls = []
        for part in tool_calls.split("|"):
            part = part.strip()
            if not part:
                continue
            # Try parse "Action: tool(params)"
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


# ------------------------------------------------------------------
# Prompt Candidate
# ------------------------------------------------------------------

@dataclass
class PromptCandidate:
    prompt: str
    fitness: float = 0.0              # overall score (0-80 scale, 8 dims * 10)
    dim_scores: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_type: str = "seed"
    reflective_feedback: str = ""      # what failure analysis led to this candidate

    @property
    def prompt_length(self) -> int:
        return len(self.prompt)

    @property
    def id(self) -> str:
        return f"g{self.generation}_{hash(self.prompt) & 0xFFFF:04x}"


# ------------------------------------------------------------------
# Reflection Engine (rule-based fallback + LLM-ready interface)
# ------------------------------------------------------------------

class ReflectionEngine:
    """
    Build a reflective dataset from scored trajectories.
    Mode = 'rule' uses heuristics; mode = 'llm' would call a reflection model.
    """

    def __init__(self, mode: str = "rule"):
        self.mode = mode
        # LLM client placeholder (activated when API key available)
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
        """Return reflective feedback text describing failures and patterns."""
        if self.mode == "llm" and self._client:
            return self._reflect_llm(scored_records, dim_avg)
        return self._reflect_rule(scored_records, dim_avg)

    def _reflect_rule(self, scored_records: List[Dict[str, Any]], dim_avg: Dict[str, float]) -> str:
        # Identify weakest dimensions
        weakest_dims = sorted(dim_avg.items(), key=lambda x: x[1])[:3]
        lines = ["=== Reflective Dataset ===", ""]
        lines.append(f"Weakest dimensions: {', '.join(f'{d}={s:.1f}' for d,s in weakest_dims)}")
        lines.append("")

        # Collect failure examples per weakest dim
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

        # Abstract failure patterns
        lines.append("--- Failure Patterns ---")
        all_signals = []
        for r in scored_records:
            all_signals.extend(r["graded"]["signals"])
        from collections import Counter
        freq = Counter(all_signals)
        for sig, cnt in freq.most_common(8):
            lines.append(f"  {sig}: {cnt} occurrences")
        lines.append("")

        # Root-cause hypotheses (rule-based)
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
        # Build a compact prompt for the reflection model
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


# ------------------------------------------------------------------
# Mutation Engine
# ------------------------------------------------------------------

class MutationEngine:
    """
    Generate prompt mutations based on reflective feedback.
    Mode = 'rule' uses template mutations; mode = 'llm' would call a generator model.
    """

    # Template patches keyed by dimension weakness
    PATCH_LIBRARY = {
        "goal_understanding": [
            "Before any action, explicitly restate the goal in your own words to confirm understanding.",
            "List the key entities and constraints from the goal before planning.",
        ],
        "planning": [
            "Break every task into at least 3 explicit sub-tasks. Number them.",
            "Write a brief plan first. Only then execute step 1.",
        ],
        "reflection": [
            "After every tool call, evaluate whether the output matches expectations. If not, correct before continuing.",
            "When you see 'Error' or 'fail' in observations, stop and diagnose before the next action.",
        ],
        "state_tracking": [
            "Maintain a running summary of completed and pending sub-tasks after each step.",
            "Before each new action, briefly recall what has already been done.",
        ],
        "termination": [
            "Before finishing, verify that all sub-tasks are complete and the original goal is satisfied.",
            "Do not terminate after a single step unless the task is trivial.",
        ],
        "tool_selection": [
            "Match the tool name to the verb in the goal (search→find, click→navigate, etc.).",
        ],
        "parameter_generation": [
            "Always provide complete JSON parameters. Never leave required fields empty.",
        ],
        "execution_accuracy": [
            "Compare your final output against the goal constraints before submitting.",
        ],
    }

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
        except Exception as e:
            print(f"[MutationEngine] LLM init failed ({e}); using rule mode.")
            self.mode = "rule"

    def mutate(self, parent: PromptCandidate, reflective_feedback: str, population_size: int = 3) -> List[PromptCandidate]:
        """Generate children from parent using reflective feedback."""
        if self.mode == "llm" and self._client:
            return self._mutate_llm(parent, reflective_feedback, population_size)
        return self._mutate_rule(parent, reflective_feedback, population_size)

    def _mutate_rule(self, parent: PromptCandidate, reflective_feedback: str, population_size: int) -> List[PromptCandidate]:
        children = []
        # Parse weakest dimensions from feedback
        weak_dims = []
        for dim in DIMENSIONS:
            if dim in reflective_feedback and f"avg=" in reflective_feedback:
                # crude heuristic: dimension mentioned near low score
                pass
            # simpler: look for dimension names in the failure patterns section
            if dim.replace("_", " ") in reflective_feedback.lower() or dim in reflective_feedback:
                weak_dims.append(dim)
        # Deduplicate and limit
        weak_dims = list(dict.fromkeys(weak_dims))[:3]
        if not weak_dims:
            weak_dims = ["planning", "reflection", "goal_understanding"]

        base_prompt = parent.prompt
        for i in range(population_size):
            # Pick a weak dimension to patch (with some randomness)
            target_dim = random.choice(weak_dims)
            patches = self.PATCH_LIBRARY.get(target_dim, ["Improve clarity and specificity."])
            patch = random.choice(patches)

            # Mutation strategy:
            # 1. Append patch as a new rule
            # 2. Insert patch near relevant section
            strategy = random.choice(["append", "insert", "rewrite"])
            new_prompt = base_prompt
            if strategy == "append":
                new_prompt = base_prompt.rstrip() + f"\n\nAdditional rule ({target_dim}):\n{patch}\n"
            elif strategy == "insert":
                lines = base_prompt.splitlines()
                insert_idx = max(1, len(lines) // 2)
                lines.insert(insert_idx, f"  - {patch}")
                new_prompt = "\n".join(lines) + "\n"
            else:
                # Rewrite: keep structure, strengthen language
                new_prompt = base_prompt.replace(
                    "Complete the given task step by step.",
                    f"Complete the given task step by step with strong attention to {target_dim.replace('_', ' ')}."
                )
                new_prompt = new_prompt.rstrip() + f"\n\n{patch}\n"

            children.append(PromptCandidate(
                prompt=new_prompt,
                generation=parent.generation + 1,
                parent_id=parent.id,
                mutation_type=f"rule_{strategy}_{target_dim}",
                reflective_feedback=reflective_feedback[:200],
            ))
        return children

    def _mutate_llm(self, parent: PromptCandidate, reflective_feedback: str, population_size: int) -> List[PromptCandidate]:
        prompt = textwrap.dedent(f"""\
            You are a prompt optimization expert.
            Current system prompt:
            ```
            {parent.prompt}
            ```

            Reflective feedback from evaluation:
            {reflective_feedback[:800]}

            Generate {population_size} improved versions of the system prompt.
            Each version must:
            1. Keep the original structure.
            2. Add or modify 1-2 rules to address the failures.
            3. Be concise (under 300 words).

            Output ONLY a JSON array of strings, one per candidate.
        """)
        try:
            r = self._client.chat.completions.create(
                model="kimi-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            content = r.choices[0].message.content.strip()
            # Extract JSON array
            m = re.search(r'\[(.*?)\]', content, re.DOTALL)
            if m:
                candidates = json.loads("[" + m.group(1) + "]")
            else:
                candidates = [content]
            return [
                PromptCandidate(
                    prompt=c,
                    generation=parent.generation + 1,
                    parent_id=parent.id,
                    mutation_type="llm_mutation",
                    reflective_feedback=reflective_feedback[:200],
                )
                for c in candidates[:population_size]
            ]
        except Exception as e:
            print(f"[MutationEngine] LLM mutation failed ({e}), falling back to rule.")
            return self._mutate_rule(parent, reflective_feedback, population_size)


# ------------------------------------------------------------------
# Pareto Selector
# ------------------------------------------------------------------

class ParetoSelector:
    """
    Pareto-aware selection:
    - Maximize fitness (overall score)
    - Minimize prompt complexity (length / number of rules)
    A candidate dominates another if it is better or equal on both objectives
    and strictly better on at least one.
    """

    def __init__(self, complexity_penalty: float = 0.01):
        self.complexity_penalty = complexity_penalty  # per-char penalty on fitness

    def effective_fitness(self, c: PromptCandidate) -> float:
        return c.fitness - self.complexity_penalty * c.prompt_length

    def dominates(self, a: PromptCandidate, b: PromptCandidate) -> bool:
        fa, fb = self.effective_fitness(a), self.effective_fitness(b)
        ca, cb = a.prompt_length, b.prompt_length
        better_fitness = fa >= fb
        better_complexity = ca <= cb
        strictly_better = fa > fb or ca < cb
        return better_fitness and better_complexity and strictly_better

    def select(self, candidates: List[PromptCandidate], elite_size: int = 2, max_pop: int = 6) -> List[PromptCandidate]:
        if not candidates:
            return []
        # Compute Pareto frontier
        frontier = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is not c and self.dominates(other, c):
                    dominated = True
                    break
            if not dominated:
                frontier.append(c)
        # Sort by effective fitness desc
        frontier.sort(key=lambda c: self.effective_fitness(c), reverse=True)
        # Elite = top by raw fitness (exploit)
        elite = sorted(candidates, key=lambda c: c.fitness, reverse=True)[:elite_size]
        # Combine frontier + elite, deduplicate by prompt hash, trim to max_pop
        combined = frontier + [e for e in elite if e not in frontier]
        seen = set()
        result = []
        for c in combined:
            h = hash(c.prompt)
            if h not in seen:
                seen.add(h)
                result.append(c)
            if len(result) >= max_pop:
                break
        return result


# ------------------------------------------------------------------
# GEPA Optimizer (main loop)
# ------------------------------------------------------------------

class GEPAPromptOptimizer:
    def __init__(
        self,
        seed_prompt: str = SEED_PROMPT,
        reflection_mode: str = "rule",
        mutation_mode: str = "rule",
        max_generations: int = 5,
        population_size: int = 4,
        elite_size: int = 2,
        max_population: int = 6,
        complexity_penalty: float = 0.01,
    ):
        self.seed_prompt = seed_prompt
        self.reflection = ReflectionEngine(mode=reflection_mode)
        self.mutation = MutationEngine(mode=mutation_mode)
        self.selector = ParetoSelector(complexity_penalty=complexity_penalty)
        self.max_generations = max_generations
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_population = max_population
        self.history: List[Dict[str, Any]] = []

    def evaluate_population(self, population: List[PromptCandidate], batch: List[Dict[str, Any]]) -> List[PromptCandidate]:
        """
        Evaluate each candidate by checking how well its prompt addresses
        the failure patterns in the current batch.
        Base score = batch average; bonus = keyword match against failure signals.
        """
        # Compute batch baseline
        overall_avg, dim_avg, scored = evaluate_batch(batch)
        # Extract failure signals (flatten)
        all_signals = []
        for r in scored:
            all_signals.extend(r["graded"]["signals"])
        from collections import Counter
        signal_freq = Counter(all_signals)
        top_signals = [s for s, _ in signal_freq.most_common(6)]

        # Weak dimensions (low avg)
        weak_dims = [d for d, v in sorted(dim_avg.items(), key=lambda x: x[1])[:3]]

        for candidate in population:
            base = overall_avg
            prompt_lower = candidate.prompt.lower()
            bonus = 0.0

            # Bonus 1: match failure signals in prompt text
            for sig in top_signals:
                # e.g., "planning:no_steps" → look for "plan", "step"
                parts = sig.split(":")
                dim_part = parts[0] if parts else sig
                if dim_part in prompt_lower:
                    bonus += 1.5
                # Also check common synonyms
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
                for syn in synonyms.get(dim_part, []):
                    if syn in prompt_lower:
                        bonus += 0.8
                        break

            # Bonus 2: prompt contains explicit rules / numbered lists → more structured
            rule_count = len(re.findall(r"(?i)^\s*\d+\.|^\s*-\s+|^\s*\*\s+", candidate.prompt, re.M))
            bonus += min(rule_count * 0.5, 3.0)

            # Penalty: excessive length
            penalty = max(0, (candidate.prompt_length - 400) * 0.01)

            candidate.fitness = base + bonus - penalty
            candidate.dim_scores = dim_avg
            # Store diagnostic info
            candidate.reflective_feedback = f"weak_dims={weak_dims}; top_signals={top_signals}; bonus={bonus:.1f}"
        return population

    def run(self, dataset: List[Dict[str, Any]], batch_size: int = 5, output_dir: str = "./gepa_output") -> Dict[str, Any]:
        """
        Run GEPA optimization loop.
        dataset: list of trajectory records (with goal, thoughts, observations, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initial population: seed + 2 random perturbations
        population = [PromptCandidate(prompt=self.seed_prompt, generation=0, mutation_type="seed")]
        for i in range(2):
            p = deepcopy(population[0])
            p.generation = 0
            p.mutation_type = "seed_perturb"
            p.prompt = p.prompt.replace("step by step", f"step by step (variant {i+1})")
            population.append(p)

        best_overall = 0.0
        best_candidate = population[0]

        for gen in range(1, self.max_generations + 1):
            print(f"\n{'='*60}")
            print(f"GEPA Generation {gen}/{self.max_generations}")
            print(f"{'='*60}")

            # 1. Sample mini-batch
            batch = random.sample(dataset, min(batch_size, len(dataset)))
            print(f"[GEPA] Mini-batch size: {len(batch)}")

            # 2. Evaluate population (fitness assignment)
            population = self.evaluate_population(population, batch)
            for c in population:
                print(f"  {c.id} | fitness={c.fitness:.1f} | len={c.prompt_length} | type={c.mutation_type}")

            # Track best
            gen_best = max(population, key=lambda c: c.fitness)
            if gen_best.fitness > best_overall:
                best_overall = gen_best.fitness
                best_candidate = gen_best
                print(f"[GEPA] New best: {gen_best.id} fitness={gen_best.fitness:.1f}")

            # 3. Reflect on failures
            _, dim_avg, scored = evaluate_batch(batch)
            reflective_feedback = self.reflection.reflect(scored, dim_avg)
            reflect_path = output_dir / f"gen{gen}_reflection.txt"
            reflect_path.write_text(reflective_feedback, encoding="utf-8")
            print(f"[GEPA] Reflection saved → {reflect_path}")

            # 4. Mutate parents into children
            children = []
            for parent in population:
                kids = self.mutation.mutate(parent, reflective_feedback, population_size=self.population_size)
                children.extend(kids)
            print(f"[GEPA] Generated {len(children)} children")

            # 5. Evaluate children
            children = self.evaluate_population(children, batch)

            # 6. Pareto selection across parents + children
            combined = population + children
            population = self.selector.select(combined, elite_size=self.elite_size, max_pop=self.max_population)
            print(f"[GEPA] Selected {len(population)} for next generation")
            for c in population:
                print(f"  → {c.id} fitness={c.fitness:.1f} len={c.prompt_length} type={c.mutation_type}")

            # Record history
            self.history.append({
                "generation": gen,
                "batch_ids": [r.get("id", i) for i, r in enumerate(batch)],
                "population": [
                    {
                        "id": c.id,
                        "fitness": c.fitness,
                        "length": c.prompt_length,
                        "mutation_type": c.mutation_type,
                        "parent_id": c.parent_id,
                    }
                    for c in population
                ],
                "best_fitness": gen_best.fitness,
                "dim_avg": dim_avg,
            })

        # Final summary
        summary = {
            "best_fitness": best_overall,
            "best_prompt": best_candidate.prompt,
            "best_generation": best_candidate.generation,
            "best_id": best_candidate.id,
            "history": self.history,
        }
        save_json(summary, str(output_dir / "gepa_summary.json"))
        (output_dir / "best_prompt.txt").write_text(best_candidate.prompt, encoding="utf-8")
        print(f"\n[GEPA] Optimization complete. Best fitness={best_overall:.1f}")
        print(f"[GEPA] Best prompt saved → {output_dir / 'best_prompt.txt'}")
        return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GEPA Prompt Optimizer (pilot)")
    parser.add_argument("--input", required=True, help="Input JSONL trajectory file")
    parser.add_argument("--output", default="./0523task/gepa_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Mini-batch size")
    parser.add_argument("--generations", type=int, default=3, help="Max generations")
    parser.add_argument("--pop-size", type=int, default=3, help="Children per parent")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule", help="Reflection/mutation mode")
    parser.add_argument("--limit", type=int, default=20, help="Limit input records for pilot")
    args = parser.parse_args()

    dataset = load_jsonl(args.input)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    print(f"[Main] Loaded {len(dataset)} records from {args.input}")
    if not dataset:
        print("[ERR] Empty dataset. Exit.")
        return

    optimizer = GEPAPromptOptimizer(
        max_generations=args.generations,
        population_size=args.pop_size,
        reflection_mode=args.mode,
        mutation_mode=args.mode,
    )
    summary = optimizer.run(dataset, batch_size=args.batch_size, output_dir=args.output)
    print(json.dumps({
        "best_fitness": summary["best_fitness"],
        "best_id": summary["best_id"],
        "generations_run": len(summary["history"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
