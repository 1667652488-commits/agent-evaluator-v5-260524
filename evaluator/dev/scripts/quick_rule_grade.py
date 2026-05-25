#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quick_rule_grade.py — 快速 RuleGrader 批量评分"""
import json, sys
from pathlib import Path

# 从 hybrid_grader.py 导入 rule_grade
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_grader import rule_grade, load_jsonl, save_jsonl

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/root/.openclaw/workspace/0522task/dumb_agent_traces/all_traces.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/root/.openclaw/workspace/0523task/rule_20.jsonl"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    records = load_jsonl(input_path)
    if limit > 0:
        records = records[:limit]
    print(f"[INFO] Rule grading {len(records)} records...")

    results = []
    for i, rec in enumerate(records):
        if not rec.get("goal") and not rec.get("question"):
            rec["goal"] = f"[{rec.get('category','unknown')}] task {rec.get('id','?')}"
        graded = rule_grade(rec)
        graded["id"] = rec.get("id", i)
        graded["goal"] = rec.get("goal", "")
        results.append(graded)
        if (i+1) % 10 == 0:
            print(f"[Progress] {i+1}/{len(records)}")

    save_jsonl(results, output_path)
    print(f"[INFO] Saved to {output_path} | total_score_avg={sum(r['overall'] for r in results)/len(results):.1f}")

if __name__ == "__main__":
    main()
