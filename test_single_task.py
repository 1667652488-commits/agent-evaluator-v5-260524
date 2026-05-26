#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单条测试脚本：调试 ACEBench Agent 样本的多步沙箱执行
===============================================================

用途：
  1. 指定单条样本，执行完整多步沙箱流程
  2. 打印每步详细轨迹（thoughts / tool_calls / observations / state / latency / tokens）
  3. 输出 5 维评估得分明细
  4. 支持加载自定义 prompt（如优化后的 prompt）进行对比测试

运行方式：
  # 测试第 1 条 agent_multi_step 样本（默认）
  python test_single_task.py

  # 测试指定 task_id
  python test_single_task.py --task-id agent_multi_step_0

  # 使用自定义 prompt 文件测试
  python test_single_task.py --task-id agent_multi_turn_5 --prompt-file ../results/gepa/phase2_optimizer/optimized_agent_prompt.txt

  # 指定模型和 API Key
  python test_single_task.py --model Qwen/Qwen2.5-14B-Instruct --api-key sk-xxx
"""

import json
import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from agent.sandbox.sandbox_runner import ACEBenchSandboxRunner

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ujjwatckhsqtmptlfzwkazagayqbjosmgknyftutiqdjnfgw")

DATA_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh"
GT_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh/possible_answer"

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_task(task_id=None, index=0, subset="agent_multi_step"):
    """查找样本：按 task_id 或按索引"""
    data_file = DATA_DIR / f"{subset}.json"
    gt_file = GT_DIR / f"{subset}.json"
    
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        return None
    
    tasks = load_jsonl(data_file)
    gts = {gt["id"]: gt for gt in load_jsonl(gt_file)} if gt_file.exists() else {}
    
    if task_id:
        for task in tasks:
            if task["id"] == task_id:
                task["_ground_truth"] = gts.get(task_id, {})
                return task
        print(f"错误: 找不到 task_id={task_id}")
        return None
    else:
        if index < len(tasks):
            task = tasks[index]
            task["_ground_truth"] = gts.get(task["id"], {})
            return task
        else:
            print(f"错误: 索引 {index} 超出范围 (共 {len(tasks)} 条)")
            return None


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def create_llm_client(api_key, model):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")


def call_llm(client, system_prompt, user_prompt, model, max_tokens=512, temperature=0.1, timeout=30):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [LLM 错误] {e}")
        return ""


# ---------------------------------------------------------------------------
# 评估器（复制自飞轮脚本，用于本地测试）
# ---------------------------------------------------------------------------

def acebench_to_json(acebench_str):
    import re
    if not acebench_str or acebench_str == "[]":
        return None, {}
    inner = acebench_str.strip().strip("[]").strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", inner, re.DOTALL)
    if not match:
        return None, {}
    tool_name = match.group(1)
    args_str = match.group(2)
    args = {}
    if args_str.strip():
        pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('[^']*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\}|[^,)]+)"
        for m in re.finditer(pattern, args_str):
            k = m.group(1)
            v_str = m.group(2).strip()
            if (v_str.startswith("'") and v_str.endswith("'")) or \
               (v_str.startswith('"') and v_str.endswith('"')):
                v = v_str[1:-1]
            elif v_str == "True":
                v = True
            elif v_str == "False":
                v = False
            elif v_str == "None":
                v = None
            else:
                try:
                    v = int(v_str)
                except:
                    try:
                        v = float(v_str)
                    except:
                        try:
                            v = eval(v_str)
                        except:
                            v = v_str
            args[k] = v
    return tool_name, args


def evaluate_trajectory(result, gt_entry):
    """5维评估"""
    steps = result.get("steps", [])
    final_state = result.get("final_state", {})
    gt_state = gt_entry.get("ground_truth", {}) if gt_entry else {}
    mile_stone = gt_entry.get("mile_stone", []) if gt_entry else []
    
    # 1. tool_selection
    tool_scores = []
    ideal_first_tool = None
    if mile_stone:
        first_ms = mile_stone[0]
        if isinstance(first_ms, str):
            ideal_first_tool, _ = acebench_to_json(first_ms)
        elif isinstance(first_ms, list) and first_ms:
            ideal_first_tool, _ = acebench_to_json(first_ms[0])
    
    for step in steps:
        actual_tool = step.get("tool_calls", {}).get("tool")
        if ideal_first_tool and actual_tool == ideal_first_tool:
            tool_scores.append(1.0)
        elif ideal_first_tool:
            tool_scores.append(0.0)
        else:
            tool_scores.append(1.0)
    tool_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0
    
    # 2. argument_generation
    arg_score = 0.0
    if steps and ideal_first_tool and mile_stone:
        first_step = steps[0]
        actual_args = first_step.get("tool_calls", {}).get("arguments", {})
        _, ideal_args = acebench_to_json(mile_stone[0] if isinstance(mile_stone[0], str) else mile_stone[0][0])
        if ideal_args:
            matched = sum(1 for k, v in ideal_args.items() if actual_args.get(k) == v)
            arg_score = matched / len(ideal_args)
    
    # 3. execution_order (LCS)
    actual_tools = [s["tool_calls"]["tool"] for s in steps if s["tool_calls"].get("tool")]
    ideal_tools = []
    for ms in mile_stone:
        if isinstance(ms, str):
            t, _ = acebench_to_json(ms)
            if t: ideal_tools.append(t)
        elif isinstance(ms, list) and ms:
            t, _ = acebench_to_json(ms[0])
            if t: ideal_tools.append(t)
    
    def lcs_ratio(seq1, seq2):
        if not seq2: return 1.0
        m, n = len(seq1), len(seq2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / max(m, n) if max(m, n) > 0 else 0.0
    
    order_score = lcs_ratio(actual_tools, ideal_tools)
    
    # 4. state_tracking
    state_score = 0.0
    if gt_state and final_state:
        mismatches = 0
        def count_mismatches(actual, expected):
            count = 0
            if isinstance(expected, dict):
                for k, v in expected.items():
                    av = actual.get(k) if isinstance(actual, dict) else None
                    if av is None and k not in (actual or {}):
                        count += 1
                    else:
                        count += count_mismatches(av, v)
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    count += max(len(actual or []), len(expected))
                else:
                    for a, e in zip(actual, expected):
                        count += count_mismatches(a, e)
            else:
                if actual != expected:
                    count += 1
            return count
        
        def count_total(obj):
            if isinstance(obj, dict):
                return sum(1 + count_total(v) for v in obj.values())
            elif isinstance(obj, list):
                return sum(count_total(item) for item in obj)
            else:
                return 1
        
        mismatches = count_mismatches(final_state, gt_state)
        total = count_total(gt_state)
        state_score = max(0, 1.0 - mismatches / max(total, 1))
    
    # 5. termination_control
    ideal_steps = len(ideal_tools) if ideal_tools else 3
    actual_steps = len([s for s in steps if s["tool_calls"].get("tool")])
    if actual_steps == 0:
        term_score = 0.0
    elif abs(actual_steps - ideal_steps) <= 1:
        term_score = 1.0
    elif abs(actual_steps - ideal_steps) <= 2:
        term_score = 0.5
    else:
        term_score = 0.0
    
    total_score = (
        tool_score * 0.25 +
        arg_score * 0.25 +
        order_score * 0.20 +
        state_score * 0.15 +
        term_score * 0.15
    )
    
    if total_score >= 0.8:
        status = "pass"
    elif total_score >= 0.4:
        status = "partial"
    else:
        status = "fail"
    
    return {
        "status": status,
        "score": round(total_score, 2),
        "dimensions": {
            "tool_selection": round(tool_score, 2),
            "argument_generation": round(arg_score, 2),
            "execution_order": round(order_score, 2),
            "state_tracking": round(state_score, 2),
            "termination_control": round(term_score, 2)
        }
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="单条 ACEBench Agent 样本调试")
    parser.add_argument("--task-id", default=None, help="指定 task_id")
    parser.add_argument("--index", type=int, default=0, help="样本索引（默认0）")
    parser.add_argument("--subset", default="data_agent_multi_step", help="子集文件名（默认 data_agent_multi_step）")
    parser.add_argument("--prompt-file", default=None, help="自定义 prompt 文件路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--max-steps", type=int, default=8, help="最大执行步数")
    args = parser.parse_args()
    
    # 1. 加载样本
    print("=" * 70)
    print("单条样本调试")
    print("=" * 70)
    
    task = find_task(args.task_id, args.index, args.subset)
    if not task:
        return
    
    print(f"\n样本信息:")
    print(f"  ID: {task['id']}")
    print(f"  Question: {task['question'][:100]}...")
    print(f"  涉及类: {task.get('involved_classes', [])}")
    print(f"  工具数: {len(task.get('function', []))}")
    
    # 2. 加载 prompt
    system_prompt = ""
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if prompt_path.exists():
            system_prompt = prompt_path.read_text(encoding="utf-8")
            print(f"  加载自定义 prompt: {prompt_path}")
        else:
            print(f"  警告: prompt 文件不存在: {prompt_path}")
    else:
        system_prompt = "你是一个智能助手，需要根据用户请求调用工具完成任务。"
        print(f"  使用默认 prompt")
    
    # 3. 创建 LLM 客户端
    client = create_llm_client(args.api_key, args.model)
    
    def llm_call_fn(system, prompt):
        return call_llm(client, system, prompt, args.model, max_tokens=512, timeout=20)
    
    # 4. 执行
    print(f"\n{'=' * 70}")
    print("开始多步沙箱执行...")
    print(f"{'=' * 70}")
    
    start_time = time.time()
    runner = ACEBenchSandboxRunner(language="zh")
    result = runner.run_task(
        task=task,
        llm_call_fn=llm_call_fn,
        max_steps=args.max_steps,
        system_prompt=system_prompt
    )
    total_time = time.time() - start_time
    
    # 5. 打印详细轨迹
    print(f"\n{'=' * 70}")
    print("执行轨迹详情")
    print(f"{'=' * 70}")
    
    for step in result["steps"]:
        print(f"\n--- Step {step['step_id']} ---")
        print(f"  thoughts: {step['thoughts'][:80]}..." if step['thoughts'] else "  thoughts: (无)")
        print(f"  tool_calls:")
        print(f"    tool: {step['tool_calls']['tool']}")
        print(f"    arguments: {json.dumps(step['tool_calls']['arguments'], ensure_ascii=False)}")
        print(f"  observations: {step['observations'][:150]}...")
        print(f"  latency: {step['latency_ms']}ms")
        print(f"  tokens: prompt={step['tokens']['prompt']}, completion={step['tokens']['completion']}")
        print(f"  state_after keys: {list(step['state_after'].keys()) if step['state_after'] else []}")
    
    # 6. 评估
    print(f"\n{'=' * 70}")
    print("评估结果")
    print(f"{'=' * 70}")
    
    gt_entry = task.get("_ground_truth", {})
    eval_result = evaluate_trajectory(result, gt_entry)
    
    print(f"  最终状态: {json.dumps(result['final_state'], ensure_ascii=False, indent=2)[:300]}...")
    print(f"  Ground Truth: {json.dumps(result.get('ground_truth', {}), ensure_ascii=False, indent=2)[:300]}...")
    print(f"\n  Outcome: {eval_result['status'].upper()}")
    print(f"  Total Score: {eval_result['score']:.2f}")
    print(f"\n  5维得分:")
    for dim, score in eval_result['dimensions'].items():
        print(f"    {dim}: {score:.2f}")
    
    # 7. 时间统计
    print(f"\n{'=' * 70}")
    print(f"总执行时间: {total_time:.1f}s | 步数: {len(result['steps'])}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
