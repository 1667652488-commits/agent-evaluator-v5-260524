#!/usr/bin/env python3
"""
GEPA 共享数据结构
"""
import textwrap
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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
