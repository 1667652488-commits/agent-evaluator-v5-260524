#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Agent Evaluator 统一入口 (v5.1)
========================================
根据子命令分发到各模块。

Usage:
    python pipeline/scripts/main.py phase1  --data-dir /path/to/agentboard_data
    python pipeline/scripts/main.py run     --input data/agent_traces.jsonl --output output/traces_scored.jsonl
    python pipeline/scripts/main.py llm     --input output/traces_scored.jsonl --output output/llm_scores.jsonl
    python pipeline/scripts/main.py hybrid  --rule-input output/traces_scored.jsonl --llm-input output/llm_scores.jsonl --output output/hybrid_scores.jsonl
    python pipeline/scripts/main.py align   --rule output/traces_scored.jsonl --llm output/llm_scores.jsonl
    python pipeline/scripts/main.py optimize --input output/hybrid_scores.jsonl --output output/optimized_prompt.json
"""

import argparse
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(prog="agent-evaluator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- phase1 ---
    p1 = sub.add_parser("phase1", help="Phase 1: AgentBoard 数据预处理")
    p1.add_argument("--data-dir", default="./agentboard_data", help="原始数据根目录")
    p1.add_argument("--output-dir", default="./data", help="输出目录")

    # --- run (agent runner) ---
    pr = sub.add_parser("run", help="运行 Agent 生成轨迹")
    pr.add_argument("--input", required=True, help="输入 JSONL (含 goal)")
    pr.add_argument("--output", required=True, help="输出 JSONL (轨迹)")
    pr.add_argument("--model", default="mock", help="模型名: mock | siliconflow | ...")
    pr.add_argument("--api-key", default="", help="API Key (硅基流动等)")
    pr.add_argument("--max-steps", type=int, default=10, help="最大步数")

    # --- rule grade ---
    pg = sub.add_parser("rule", help="RuleGrader: 8维规则评估")
    pg.add_argument("--input", required=True, help="输入轨迹 JSONL")
    pg.add_argument("--output", required=True, help="输出评分 JSONL")

    # --- llm grade ---
    pl = sub.add_parser("llm", help="LLMGrader: LLM-as-Judge 评估")
    pl.add_argument("--input", required=True, help="输入轨迹 JSONL")
    pl.add_argument("--output", required=True, help="输出评分 JSONL")
    pl.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    pl.add_argument("--api-key", default="")

    # --- hybrid ---
    ph = sub.add_parser("hybrid", help="HybridGrader: 融合评估")
    ph.add_argument("--rule-input", required=True, help="Rule 评分 JSONL")
    ph.add_argument("--llm-input", required=True, help="LLM 评分 JSONL")
    ph.add_argument("--output", required=True, help="输出融合评分 JSONL")

    # --- align ---
    pa = sub.add_parser("align", help="GraderAligner: Rule vs LLM 对齐分析")
    pa.add_argument("--rule", required=True)
    pa.add_argument("--llm", required=True)
    pa.add_argument("--output-dir", default="./output/alignment")

    # --- optimize (GEPA) ---
    po = sub.add_parser("optimize", help="GEPA Optimizer: 遗传进化优化 Prompt")
    po.add_argument("--input", required=True, help="评分结果 JSONL (作为 fitness)")
    po.add_argument("--output", required=True, help="优化后的 prompt 输出路径")
    po.add_argument("--generations", type=int, default=5)
    po.add_argument("--population", type=int, default=6)
    po.add_argument("--api-key", default="")

# --- acebench (ACEBench cold-start + iteration pipeline) ---
    p_ac = sub.add_parser("acebench", help="ACEBench 飞轮: 冷启动→评估器→优化器→逐轮迭代")
    p_ac.add_argument("--phase", default="all", choices=["all", "coldstart", "evalopt", "iteration", "summary"], help="执行阶段: all=全部, coldstart=冷启动, evalopt=生成评估器+优化器, iteration=逐轮迭代, summary=汇总报告")
    p_ac.add_argument("--dataset", default="ACEBench", help="数据集名称 (默认 ACEBench)")
    p_ac.add_argument("--ratio", type=float, default=0.05, help="冷启动抽样比例 (默认 0.05)")
    p_ac.add_argument("--rounds", type=int, default=10, help="迭代轮数 (默认 10)")
    p_ac.add_argument("--iter-ratio", type=float, default=0.10, help="每轮迭代抽样比例 (默认 0.10)")
    p_ac.add_argument("--gepa", action="store_true", help="开启 GEPA (默认关闭)")
    p_ac.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="模型名称")
    p_ac.add_argument("--api-key", default="", help="SiliconFlow API Key")
    p_ac.add_argument("--output-dir", default="./results/no_gepa", help="结果输出目录")
    p_ac.add_argument("--coldstart-results", default="", help="Phase 2/3/4 时指定冷启动结果路径 (默认自动查找)")

    args = parser.parse_args()

    if args.cmd == "phase1":
        from data_preprocessing.scripts.phase1_preprocess import main as _main
        _main()
    elif args.cmd == "run":
        from agent.models.agent_runner import main as _main
        _main()
    elif args.cmd == "rule":
        from evaluator.dev.scripts.quick_rule_grade import main as _main
        _main()
    elif args.cmd == "llm":
        from evaluator.dev.scripts.llm_grader import main as _main
        _main()
    elif args.cmd == "hybrid":
        from evaluator.dev.scripts.hybrid_grader import main as _main
        _main()
    elif args.cmd == "align":
        from evaluator.dev.scripts.grader_aligner import main as _main
        _main()
    elif args.cmd == "optimize":
        from optimizer.dev.scripts.gepa_optimizer import main as _main
        _main()
    elif args.cmd == "acebench":
        # 设置环境变量并调用 ACEBench 飞轮脚本
        import os
        if args.api_key:
            os.environ["SILICONFLOW_API_KEY"] = args.api_key
        os.environ["ACEBENCH_PHASE"] = args.phase
        os.environ["ACEBENCH_ROUNDS"] = str(args.rounds)
        os.environ["ACEBENCH_ITER_RATIO"] = str(args.iter_ratio)
        os.environ["ACEBENCH_OUTPUT_DIR"] = args.output_dir
        if args.coldstart_results:
            os.environ["ACEBENCH_COLDSTART_RESULTS"] = args.coldstart_results
        # 导入并执行
        import runpy
        script_path = Path(__file__).parent / "run_acebench_pipeline_no_gepa.py"
        runpy.run_path(str(script_path), run_name="__main__")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
