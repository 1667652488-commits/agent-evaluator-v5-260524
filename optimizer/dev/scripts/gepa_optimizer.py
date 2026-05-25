#!/usr/bin/env python3
"""
GEPA (Genetic Pareto) Prompt Optimizer — 主循环
位置: optimizer/dev/scripts/gepa_optimizer.py
功能: 协调 ReflectionEngine → MutationEngine → ParetoSelector 完成进化优化
"""
import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.dev.skills.gepa.prompt_candidate import PromptCandidate, SEED_PROMPT, DIMENSIONS
from optimizer.dev.skills.gepa.fitness_evaluator import evaluate_batch, compute_fitness
from evaluator.dev.skills.gepa.reflection_engine import ReflectionEngine
from optimizer.dev.skills.gepa.mutation_engine import MutationEngine
from optimizer.dev.skills.gepa.pareto_selector import ParetoSelector


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


def save_json(data: Any, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    def run(self, dataset: List[Dict[str, Any]], batch_size: int = 5, output_dir: str = "./gepa_output") -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 初始种群: seed + 2 随机扰动
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

            # 1. 采样 mini-batch
            batch = random.sample(dataset, min(batch_size, len(dataset)))
            print(f"[GEPA] Mini-batch size: {len(batch)}")

            # 2. 评估种群（fitness 赋值）
            population = compute_fitness(population, batch)
            for c in population:
                print(f"  {c.id} | fitness={c.fitness:.1f} | len={c.prompt_length} | type={c.mutation_type}")

            # 追踪最优
            gen_best = max(population, key=lambda c: c.fitness)
            if gen_best.fitness > best_overall:
                best_overall = gen_best.fitness
                best_candidate = gen_best
                print(f"[GEPA] New best: {gen_best.id} fitness={gen_best.fitness:.1f}")

            # 3. 反射失败
            _, dim_avg, scored = evaluate_batch(batch)
            reflective_feedback = self.reflection.reflect(scored, dim_avg)
            reflect_path = output_dir / f"gen{gen}_reflection.txt"
            reflect_path.write_text(reflective_feedback, encoding="utf-8")
            print(f"[GEPA] Reflection saved → {reflect_path}")

            # 4. 变异生成子代
            children = []
            for parent in population:
                kids = self.mutation.mutate(parent, reflective_feedback, population_size=self.population_size)
                children.extend(kids)
            print(f"[GEPA] Generated {len(children)} children")

            # 5. 评估子代
            children = compute_fitness(children, batch)

            # 6. Pareto 选择
            combined = population + children
            population = self.selector.select(combined, elite_size=self.elite_size, max_pop=self.max_population)
            print(f"[GEPA] Selected {len(population)} for next generation")
            for c in population:
                print(f"  → {c.id} fitness={c.fitness:.1f} len={c.prompt_length} type={c.mutation_type}")

            # 记录历史
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

        # 最终总结
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
    parser = argparse.ArgumentParser(description="GEPA Prompt Optimizer")
    parser.add_argument("--input", required=True, help="Input JSONL trajectory file")
    parser.add_argument("--output", default="./gepa_output", help="Output directory")
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
