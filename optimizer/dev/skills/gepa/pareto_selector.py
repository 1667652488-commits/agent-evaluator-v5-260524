#!/usr/bin/env python3
"""
GEPA ParetoSelector — 双目标 Pareto 前沿选择
位置: optimizer/dev/skills/gepa/
输入: candidate_configs[] + 每个 config 的评估分数
输出: 最优 config 列表（fitness 最大化 + prompt 长度最小化）
"""
from typing import List

from .prompt_candidate import PromptCandidate


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

    def select_best(self, candidates: List[PromptCandidate]) -> PromptCandidate:
        """返回 fitness 最高的单一候选"""
        if not candidates:
            raise ValueError("Empty candidate list")
        return max(candidates, key=lambda c: c.fitness)

    def pareto_report(self, candidates: List[PromptCandidate]) -> dict:
        """输出 Pareto 前沿的分析报告"""
        frontier = self.select(candidates, elite_size=0, max_pop=len(candidates))
        return {
            "frontier_size": len(frontier),
            "frontier_ids": [c.id for c in frontier],
            "best_fitness": max(c.fitness for c in candidates) if candidates else 0,
            "best_eff_fitness": max(self.effective_fitness(c) for c in candidates) if candidates else 0,
            "shortest_prompt": min(c.prompt_length for c in candidates) if candidates else 0,
            "longest_prompt": max(c.prompt_length for c in candidates) if candidates else 0,
        }
