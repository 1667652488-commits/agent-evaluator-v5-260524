#!/usr/bin/env python3
"""
GEPA MutationEngine — 基于弱点报告生成 prompt 变异
位置: optimizer/dev/skills/gepa/
输入: reflection_report + 当前 agent prompt
输出: candidate_configs[]（每个含 system_prompt_patch + tool_constraints）
"""
import json
import os
import random
import re
import textwrap
from typing import Dict, Any, List

from .prompt_candidate import PromptCandidate, DIMENSIONS


class MutationEngine:
    """
    基于反射反馈生成 prompt 变异。
    Mode = 'rule' 使用模板变异; mode = 'llm' 调用生成模型（预留接口）。
    """

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
        """基于反射反馈从父代生成子代候选 prompt"""
        if self.mode == "llm" and self._client:
            return self._mutate_llm(parent, reflective_feedback, population_size)
        return self._mutate_rule(parent, reflective_feedback, population_size)

    def _mutate_rule(self, parent: PromptCandidate, reflective_feedback: str, population_size: int) -> List[PromptCandidate]:
        children = []
        weak_dims = []
        for dim in DIMENSIONS:
            if dim.replace("_", " ") in reflective_feedback.lower() or dim in reflective_feedback:
                weak_dims.append(dim)
        weak_dims = list(dict.fromkeys(weak_dims))[:3]
        if not weak_dims:
            weak_dims = ["planning", "reflection", "goal_understanding"]

        base_prompt = parent.prompt
        for i in range(population_size):
            target_dim = random.choice(weak_dims)
            patches = self.PATCH_LIBRARY.get(target_dim, ["Improve clarity and specificity."])
            patch = random.choice(patches)
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

    def generate_configs(self, parent: PromptCandidate, reflection_report: Dict[str, Any], population_size: int = 3) -> List[Dict[str, Any]]:
        """生成结构化的 candidate_configs（含 system_prompt_patch + tool_constraints）"""
        candidates = self.mutate(parent, json.dumps(reflection_report, ensure_ascii=False), population_size)
        configs = []
        for c in candidates:
            configs.append({
                "prompt_candidate_id": c.id,
                "system_prompt": c.prompt,
                "system_prompt_patch": c.prompt[len(parent.prompt):] if c.prompt.startswith(parent.prompt) else c.prompt,
                "tool_constraints": {},  # 可扩展
                "parent_id": c.parent_id,
                "mutation_type": c.mutation_type,
                "generation": c.generation,
            })
        return configs
