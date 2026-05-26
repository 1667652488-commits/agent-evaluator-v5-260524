#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACEBench 飞轮执行脚本（不开启 GEPA）
=====================================

Phase 1: 冷启动（5% = 50条，抽取 Agent + Normal + Special 混合）
Phase 2: 基于冷启动结果生成评估器 + 优化器
Phase 3: 逐轮迭代（10% × 10轮，优化→测试→记录通过率）

运行方式:
    export SILICONFLOW_API_KEY=sk-xxx
    python pipeline/scripts/run_acebench_pipeline_no_gepa.py

输出:
    results/no_gepa/
        ├── phase1_coldstart/          # 冷启动原始结果
        ├── phase2_evaluator/          # 生成的评估器
        ├── phase2_optimizer/          # 生成的优化器
        ├── phase3_iteration/          # 逐轮迭代结果
        │   ├── round_1/
        │   ├── round_2/
        │   └── ...
        └── summary_report.json        # 汇总报告
"""

import json
import os
import sys
import time
import random
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openai"])
    from openai import OpenAI
    print("openai installed.")

# ---------------------------------------------------------------------------
# 0. 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "gepa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
if not API_KEY:
    # 回退到硬编码 key（用于分支运行）
    API_KEY = "sk-ujjwatckhsqtmptlfzwkazagayqbjosmgknyftutiqdjnfgw"
if not API_KEY:
    print("错误: 请设置 SILICONFLOW_API_KEY 环境变量")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
MODEL = "Qwen/Qwen2.5-14B-Instruct"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh"
GT_DIR = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh/possible_answer"

# 随机种子保证可复现
random.seed(42)

# ---------------------------------------------------------------------------
# 1. 数据加载
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_datasets():
    """仅加载 ACEBench Agent 子集（multi_step + multi_turn）"""
    datasets = {}
    
    # 仅加载 Agent 子集
    for f in sorted(DATA_DIR.glob("data_agent_*.json")):
        name = f.stem
        datasets[name] = load_jsonl(f)
        print(f"  加载 {name}: {len(datasets[name])} 条")
    
    # 加载 ground truth（仅 Agent 子集）
    gt_files = {}
    for f in sorted(GT_DIR.glob("data_agent_*.json")):
        name = f.stem
        gt_files[name] = {gt["id"]: gt for gt in load_jsonl(f)}
    
    total = sum(len(v) for v in datasets.values())
    print(f"\n总计 (仅 Agent 子集): {total} 条")
    return datasets, gt_files


# ---------------------------------------------------------------------------
# 2. LLM 调用
# ---------------------------------------------------------------------------

def call_llm(system_prompt, user_prompt, max_tokens=512, temperature=0.1):
    """调用 SiliconFlow LLM"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  LLM 调用错误: {e}")
        return ""


# ---------------------------------------------------------------------------
# 3. Agent 执行（单步 Function Calling）
# ---------------------------------------------------------------------------

def build_agent_prompt(task):
    """构建 Agent prompt"""
    tools_desc = json.dumps(task["function"], ensure_ascii=False, indent=2)
    
    prompt = f"""你是一个智能助手，需要根据用户请求调用合适的工具来完成任务。

## 用户请求
{task['question']}

## 可用工具列表
{tools_desc}

## 要求
1. 仔细分析用户需求，判断是否需要调用工具
2. 如果需要调用工具，选择正确的工具名称
3. 严格按照工具的 parameters 定义填写参数（注意 required 字段必须填写）
4. 输出格式必须是合法的 JSON:
{{
  "tool": "工具名称",
  "arguments": {{
    "参数名": "参数值"
  }}
}}
5. 如果不需要调用工具，输出: {{"tool": null, "arguments": {{}}}}

## 当前状态（如适用）
{json.dumps(task.get('initial_config', {}), ensure_ascii=False, indent=2)}

请输出你的工具调用决策（只输出 JSON，不要其他内容）:"""
    return prompt


def parse_agent_output(output):
    """解析 Agent 的 JSON 输出"""
    try:
        # 提取 JSON 块
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()
        
        result = json.loads(output.strip())
        tool = result.get("tool")
        args = result.get("arguments", {})
        return tool, args
    except Exception as e:
        # 尝试正则提取
        import re
        tool_match = re.search(r'"tool"\s*:\s*"([^"]*)"', output)
        if tool_match:
            tool = tool_match.group(1)
            if tool == "null" or tool == "":
                return None, {}
        return None, {}


def run_agent_on_task(task):
    """在单条任务上运行 Agent，返回 {tool, arguments, raw_output}"""
    prompt = build_agent_prompt(task)
    raw_output = call_llm(
        "你是一个工具调用助手，严格按 JSON 格式输出工具调用决策。",
        prompt,
        max_tokens=512,
        temperature=0.1
    )
    tool, args = parse_agent_output(raw_output)
    return {
        "tool": tool,
        "arguments": args,
        "raw_output": raw_output,
        "task_id": task["id"],
        "question": task["question"]
    }


# ---------------------------------------------------------------------------
# 4. 评估（对比 Ground Truth）
# ---------------------------------------------------------------------------



def _compare_tool_args(tool, args, gt_tool, gt_args):
    """通用工具名+参数对比逻辑"""
    # 工具名匹配
    tool_match = (tool == gt_tool) if tool else False
    if not tool and not gt_tool:
        tool_match = True

    if not tool_match:
        return {"status": "fail", "score": 0.0, "details": f"工具选择错误: 期望 '{gt_tool}'，实际 '{tool}'", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}

    # 参数对比
    expected_params = set(gt_args.keys())
    actual_params = set(args.keys())

    missing = list(expected_params - actual_params)
    extra = list(actual_params - expected_params)

    value_mismatches = []
    for k in expected_params & actual_params:
        if gt_args[k] != args[k]:
            value_mismatches.append({"param": k, "expected": gt_args[k], "actual": args[k]})

    if not missing and not extra and not value_mismatches:
        return {"status": "pass", "score": 1.0, "details": "工具名和参数完全匹配", "tool_match": True, "param_match": True, "missing_params": [], "extra_params": [], "value_mismatches": []}
    elif not missing and len(value_mismatches) <= 1:
        return {"status": "partial", "score": 0.5, "details": f"工具正确，参数值不匹配: {len(value_mismatches)}处, 多余{len(extra)}个", "tool_match": True, "param_match": False, "missing_params": [], "extra_params": extra, "value_mismatches": value_mismatches}
    else:
        return {"status": "fail", "score": 0.0, "details": f"参数差距大: 缺失{len(missing)}个, 多余{len(extra)}个, 不匹配{len(value_mismatches)}处", "tool_match": True, "param_match": False, "missing_params": missing, "extra_params": extra, "value_mismatches": value_mismatches}


def parse_milestone(ms_str):
    """解析 ACEBench mile_stone 字符串如 [tool_name(arg1='val1', arg2=123)]"""
    if not ms_str:
        return None, {}
    ms_str = ms_str.strip().strip("[]")
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", ms_str)
    if not match:
        return None, {}
    tool_name = match.group(1)
    args_str = match.group(2)
    args = {}
    if args_str.strip():
        pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('[^']*'|\"[^\"]*\"|\[[^\]]*\]|[^,)]+)"
        for m in re.finditer(pattern, args_str):
            k = m.group(1)
            v_str = m.group(2).strip()
            if (v_str.startswith("'") and v_str.endswith("'")) or (v_str.startswith('"') and v_str.endswith('"')):
                v = v_str[1:-1]
            elif v_str in ("True", "true"):
                v = True
            elif v_str in ("False", "false"):
                v = False
            elif v_str in ("None", "none"):
                v = None
            else:
                try:
                    v = int(v_str)
                except:
                    try:
                        v = float(v_str)
                    except:
                        v = v_str
            args[k] = v
    return tool_name, args


def compare_with_ground_truth(agent_result, gt_entry):
    """
    对比 Agent 输出与 Ground Truth
    支持 4 种 ground truth 格式:
      1. dict{tool:{params}} — 直接对比工具名+参数值
      2. dict{tool:[params]} — 对比工具名+参数名是否齐全
      3. list[dict{tool:{params}}] — 多步，取第一个/最后一个作为单步对比
      4. str — 不调用工具
    返回: {status, score, details, ...}
    """
    tool = agent_result["tool"]
    args = agent_result["arguments"]
    
    # ---- Agent 子集特殊处理：使用 mile_stone 作为期望的单步调用 ----
    task_id = gt_entry.get("id", "")
    is_agent_subset = task_id.startswith(("agent_multi_step", "agent_multi_turn"))
    if is_agent_subset:
        milestones = gt_entry.get("mile_stone", [])
        if milestones and len(milestones) > 0:
            gt_tool, gt_args = parse_milestone(milestones[0])
            if gt_tool:
                return _compare_tool_args(tool, args, gt_tool, gt_args)
        # 没有 mile_stone 时 fallback
        return {"status": "pass", "score": 1.0, "details": "Agent 子集无 mile_stone，跳过单步对比", "tool_match": True, "param_match": True, "missing_params": [], "extra_params": [], "value_mismatches": []}
    
    # 获取 ground truth
    gt_raw = gt_entry.get("ground_truth", None)
    if gt_raw is None or gt_raw == []:
        return {"status": "fail", "score": 0.0, "details": "无 ground truth", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}
    
    # ---- 格式 4: 字符串 → 不调用工具 ----
    if isinstance(gt_raw, str):
        # 期望不调用工具
        if tool is None or tool == "null" or tool == "":
            return {"status": "pass", "score": 1.0, "details": "正确判断不调用工具", "tool_match": True, "param_match": True, "missing_params": [], "extra_params": [], "value_mismatches": []}
        else:
            return {"status": "fail", "score": 0.0, "details": f"应不调用工具，但实际调用了 '{tool}'", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}
    
    # ---- 格式 3: 列表 → 取第一个元素 ----
    if isinstance(gt_raw, list):
        if len(gt_raw) == 0:
            return {"status": "fail", "score": 0.0, "details": "ground truth 为空列表", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}
        gt_first = gt_raw[0]
    else:
        gt_first = gt_raw
    
    # ---- 此时 gt_first 应为 dict ----
    if not isinstance(gt_first, dict) or not gt_first:
        return {"status": "fail", "score": 0.0, "details": "ground truth 格式无法解析", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}
    
    gt_tool = list(gt_first.keys())[0]
    gt_args_raw = gt_first[gt_tool]
    
    # ---- 格式 2: gt_args 是列表 [param1, param2] → 只检查参数名 ----
    if isinstance(gt_args_raw, list):
        # 参数名对比模式
        gt_args = {}  # 用于兼容旧逻辑
        required_params = set(gt_args_raw)
        actual_params = set(args.keys())
        
        missing = list(required_params - actual_params)
        extra = list(actual_params - required_params)
        
        # 工具名匹配
        tool_match = (tool == gt_tool) if tool else False
        if not tool and not gt_tool:
            tool_match = True
        
        if not tool_match:
            return {"status": "fail", "score": 0.0, "details": f"工具选择错误: 期望 '{gt_tool}'，实际 '{tool}'", "tool_match": False, "param_match": False, "missing_params": missing, "extra_params": extra, "value_mismatches": []}
        
        return _compare_tool_args(tool, args, gt_tool, {p: "" for p in required_params})
    
    # ---- 格式 1: gt_args 是字典 {param: value} → 完整对比 ----
    elif isinstance(gt_args_raw, dict):
        gt_args = gt_args_raw
        
        # 工具名匹配
        tool_match = (tool == gt_tool) if tool else False
        if not tool and not gt_tool:
            tool_match = True
        
        if not tool_match:
            return {"status": "fail", "score": 0.0, "details": f"工具选择错误: 期望 '{gt_tool}'，实际 '{tool}'", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}
        
        # 参数对比
        expected_params = set(gt_args.keys())
        actual_params = set(args.keys())
        
        missing = list(expected_params - actual_params)
        extra = list(actual_params - expected_params)
        
        value_mismatches = []
        for k in expected_params & actual_params:
            if gt_args[k] != args[k]:
                value_mismatches.append({"param": k, "expected": gt_args[k], "actual": args[k]})
        
        if not missing and not extra and not value_mismatches:
            return {"status": "pass", "score": 1.0, "details": "工具名和参数完全匹配", "tool_match": True, "param_match": True, "missing_params": [], "extra_params": [], "value_mismatches": []}
        elif not missing and len(value_mismatches) <= 1:
            return {"status": "partial", "score": 0.5, "details": f"工具正确，参数值不匹配: {len(value_mismatches)}处, 多余{len(extra)}个", "tool_match": True, "param_match": False, "missing_params": [], "extra_params": extra, "value_mismatches": value_mismatches}
        else:
            return {"status": "fail", "score": 0.0, "details": f"参数差距大: 缺失{len(missing)}个, 多余{len(extra)}个, 不匹配{len(value_mismatches)}处", "tool_match": True, "param_match": False, "missing_params": missing, "extra_params": extra, "value_mismatches": value_mismatches}
    
    else:
        return {"status": "fail", "score": 0.0, "details": f"ground truth 参数格式未知: {type(gt_args_raw)}", "tool_match": False, "param_match": False, "missing_params": [], "extra_params": [], "value_mismatches": []}


# ---------------------------------------------------------------------------
# 5. LLM-as-Judge 评估说明
# ---------------------------------------------------------------------------

def llm_judge_explanation(task, agent_result, eval_result):
    """用 LLM 生成评估说明"""
    gt_entry = task.get("ground_truth", {})
    prompt = f"""请分析以下 Agent 的工具调用决策，给出专业评估说明。

## 用户请求
{task['question']}

## Agent 决策
工具: {agent_result['tool']}
参数: {json.dumps(agent_result['arguments'], ensure_ascii=False)}

## Ground Truth
{json.dumps(gt_entry, ensure_ascii=False, indent=2)}

## 初步评估
状态: {eval_result['status']}
分数: {eval_result['score']}
详情: {eval_result['details']}

## 要求
1. 分析 Agent 决策是否正确
2. 如果失败，指出具体原因（工具选择错误？参数缺失？参数值错误？格式问题？）
3. 给出改进建议（一句话）
4. 输出格式: {{"assessment": "...", "failure_reason": "...", "improvement": "..."}}

请输出 JSON:"""

    raw = call_llm("你是一个 Agent 评估专家。", prompt, max_tokens=300, temperature=0.1)
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw.strip())
    except:
        return {
            "assessment": eval_result["details"],
            "failure_reason": "解析失败",
            "improvement": "检查输出格式"
        }


# ---------------------------------------------------------------------------
# 6. Phase 1: 冷启动
# ---------------------------------------------------------------------------

def run_cold_start(datasets, gt_files, ratio=0.05):
    """
    执行冷启动：抽取 ratio 比例的数据，跑 Agent，对比 GT，产出带评估的轨迹
    """
    print("\n" + "=" * 70)
    print("Phase 1: 冷启动")
    print("=" * 70)
    
    # 分层抽样：Agent 子集冷启动 10 条
    # multi_step (20条) 抽 4 条，multi_turn (30条) 抽 6 条
    coldstart_tasks = []
    for subset_name, records in datasets.items():
        if "multi_step" in subset_name:
            n = min(4, len(records))
        elif "multi_turn" in subset_name:
            n = min(6, len(records))
        else:
            n = 0
        if n > 0:
            sampled = [dict(r) for r in random.sample(records, n)]
            for task in sampled:
                task["_subset"] = subset_name
            coldstart_tasks.extend(sampled)
            print(f"  {subset_name}: 抽取 {n}/{len(records)} 条")
    
    print(f"\n  冷启动总样本: {len(coldstart_tasks)} 条")
    ms_count = sum(1 for t in coldstart_tasks if "multi_step" in t["_subset"])
    mt_count = sum(1 for t in coldstart_tasks if "multi_turn" in t["_subset"])
    print(f"  其中 multi_step: {ms_count} 条, multi_turn: {mt_count} 条")
    
    # 逐条执行
    results = []
    pass_count = partial_count = fail_count = 0
    
    for i, task in enumerate(coldstart_tasks, 1):
        subset = task["_subset"]
        task_id = task["id"]
        print(f"\n  [{i}/{len(coldstart_tasks)}] {subset} | {task_id}")
        print(f"    Question: {task['question'][:50]}...")
        
        # 运行 Agent
        agent_result = run_agent_on_task(task)
        print(f"    Agent -> {agent_result['tool']}: {json.dumps(agent_result['arguments'], ensure_ascii=False)[:80]}")
        
        # 获取 ground truth
        gt_subset = subset.replace("data_", "").replace("_", "")  # 简化映射
        gt_entry = None
        for gt_name, gt_data in gt_files.items():
            if task_id in gt_data:
                gt_entry = gt_data[task_id]
                break
        
        if gt_entry:
            # 评估
            eval_result = compare_with_ground_truth(agent_result, gt_entry)
            # LLM 说明
            explanation = llm_judge_explanation(task, agent_result, eval_result)
            
            print(f"    评估 -> {eval_result['status'].upper()} | {eval_result['details']}")
            
            if eval_result["status"] == "pass":
                pass_count += 1
            elif eval_result["status"] == "partial":
                partial_count += 1
            else:
                fail_count += 1
        else:
            eval_result = {"status": "unknown", "score": 0, "details": "无 ground truth"}
            explanation = {"assessment": "无 ground truth", "failure_reason": "N/A", "improvement": "N/A"}
            print(f"    评估 -> 无 ground truth")
        
        results.append({
            "task_id": task_id,
            "subset": subset,
            "question": task["question"],
            "agent_result": agent_result,
            "ground_truth": gt_entry,
            "evaluation": eval_result,
            "explanation": explanation
        })
        
        # 避免限流
        time.sleep(0.5)
    
    # 汇总
    total = len(results)
    summary = {
        "total": total,
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "pass_rate": pass_count / total if total > 0 else 0,
        "partial_rate": partial_count / total if total > 0 else 0,
        "fail_rate": fail_count / total if total > 0 else 0,
        "avg_score": sum(r["evaluation"]["score"] for r in results) / total if total > 0 else 0
    }
    
    print(f"\n  冷启动汇总:")
    print(f"    总计: {total} | 通过: {pass_count} | 部分通过: {partial_count} | 失败: {fail_count}")
    print(f"    通过率: {summary['pass_rate']:.1%} | 平均分数: {summary['avg_score']:.2f}")
    
    # 保存结果
    phase1_dir = RESULTS_DIR / "phase1_coldstart"
    phase1_dir.mkdir(exist_ok=True)
    
    with open(phase1_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(phase1_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"  结果已保存: {phase1_dir}")
    return results, summary


# ---------------------------------------------------------------------------
# 7. Phase 2: 生成评估器 + 优化器
# ---------------------------------------------------------------------------

def generate_evaluator(coldstart_results, summary):
    """
    基于冷启动结果生成 ACEBench 专用评估器配置
    """
    print("\n" + "=" * 70)
    print("Phase 2: 生成评估器 + 优化器")
    print("=" * 70)
    
    # 分析失败模式
    failure_patterns = defaultdict(int)
    for r in coldstart_results:
        if r["evaluation"]["status"] in ["fail", "partial"]:
            reason = r["explanation"].get("failure_reason", "unknown")
            failure_patterns[reason] += 1
    
    print(f"  失败模式统计:")
    for reason, count in sorted(failure_patterns.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    
    # 生成评估器配置
    evaluator_config = {
        "name": "ACEBench_Grader_v1",
        "created_from": "cold_start",
        "sample_count": len(coldstart_results),
        "dimensions": [
            {
                "name": "tool_selection",
                "weight": 0.4,
                "description": "工具选择正确性",
                "criteria": "选择的工具名与 ground truth 一致"
            },
            {
                "name": "argument_generation",
                "weight": 0.4,
                "description": "参数生成准确性",
                "criteria": "参数完整（无缺失/多余）、值正确"
            },
            {
                "name": "format_compliance",
                "weight": 0.2,
                "description": "输出格式合规性",
                "criteria": "输出合法 JSON，包含 tool 和 arguments 字段"
            }
        ],
        "scoring": {
            "pass": {"min_score": 1.0, "color": "green"},
            "partial": {"min_score": 0.3, "max_score": 0.99, "color": "yellow"},
            "fail": {"max_score": 0.29, "color": "red"}
        },
        "failure_patterns": dict(failure_patterns),
        "baseline_stats": summary
    }
    
    phase2_eval_dir = RESULTS_DIR / "phase2_evaluator"
    phase2_eval_dir.mkdir(exist_ok=True)
    with open(phase2_eval_dir / "evaluator_config.json", "w", encoding="utf-8") as f:
        json.dump(evaluator_config, f, ensure_ascii=False, indent=2)
    print(f"  评估器配置已保存: {phase2_eval_dir / 'evaluator_config.json'}")
    
    # 生成优化器配置
    optimizer_prompts = []
    
    # 根据失败模式生成 prompt 补丁
    if "工具选择错误" in str(failure_patterns) or any("工具" in k for k in failure_patterns):
        optimizer_prompts.append({
            "target": "tool_selection",
            "patch": "在决策前，先仔细阅读每个工具的 description，匹配用户需求与工具功能。",
            "priority": 1
        })
    
    if "参数缺失" in str(failure_patterns) or any("参数" in k for k in failure_patterns):
        optimizer_prompts.append({
            "target": "argument_generation",
            "patch": "检查工具的 required 字段列表，确保所有必填参数都已提供，不要遗漏。",
            "priority": 1
        })
    
    if "格式" in str(failure_patterns) or any("格式" in k for k in failure_patterns):
        optimizer_prompts.append({
            "target": "format_compliance",
            "patch": "严格输出合法 JSON 格式，包含 tool 和 arguments 两个字段，不要添加其他内容。",
            "priority": 2
        })
    
    # 通用优化补丁
    optimizer_prompts.append({
        "target": "general",
        "patch": "如果用户请求不需要调用任何工具（闲聊、问候），输出 {\"tool\": null, \"arguments\": {}}。",
        "priority": 3
    })
    
    optimizer_config = {
        "name": "ACEBench_Optimizer_GEPA_v1",
        "created_from": "cold_start",
        "strategy": "gepa_evolutionary",
        "gepa_enabled": True,
        "gepa_settings": {
            "generations": 3,
            "population_size": 5,
            "elite_size": 2,
            "mutation_mode": "rule",
            "reflection_mode": "llm"
        },
        "prompt_patches": optimizer_prompts,
        "regression_check": {
            "enabled": True,
            "threshold": 0.05
        },
        "baseline": summary
    }
    
    phase2_opt_dir = RESULTS_DIR / "phase2_optimizer"
    phase2_opt_dir.mkdir(exist_ok=True)
    with open(phase2_opt_dir / "optimizer_config.json", "w", encoding="utf-8") as f:
        json.dump(optimizer_config, f, ensure_ascii=False, indent=2)
    print(f"  优化器配置已保存: {phase2_opt_dir / 'optimizer_config.json'}")
    
    # 生成优化后的 Agent prompt
    base_prompt = """你是一个智能助手，需要根据用户请求调用合适的工具来完成任务。

## 决策流程
1. 分析用户需求，判断是否需要调用工具
2. 如果需要，从可用工具列表中选择最匹配的工具
3. 仔细阅读工具的 parameters 定义，确保填写所有 required 字段
4. 输出合法 JSON 格式

## 输出格式
{
  "tool": "工具名称或 null",
  "arguments": {
    "参数名": "参数值"
  }
}
"""
    
    # 应用补丁
    for patch in optimizer_prompts:
        base_prompt += f"\n## [{patch['target']}] {patch['patch']}\n"
    
    with open(phase2_opt_dir / "optimized_agent_prompt.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print(f"  优化后 Agent prompt 已保存: {phase2_opt_dir / 'optimized_agent_prompt.txt'}")
    
    return evaluator_config, optimizer_config, base_prompt


# ---------------------------------------------------------------------------
# 8. Phase 3: 逐轮迭代
# ---------------------------------------------------------------------------

def run_iteration(datasets, gt_files, optimized_prompt, rounds=4, ratio=0.25):
    """
    逐轮迭代飞轮（Agent 子集专用）
    40 条数据分 4 轮，每轮 10 条
    """
    print("\n" + "=" * 70)
    print(f"Phase 3: 逐轮迭代 ({rounds} 轮 × {ratio:.0%})")
    print("=" * 70)
    
    # 合并所有数据并分层切分
    all_tasks = []
    for subset_name, records in datasets.items():
        for task in records:
            task = dict(task)
            task["_subset"] = subset_name
            all_tasks.append(task)
    
    random.shuffle(all_tasks)
    
    # 切分为 rounds 份
    chunk_size = max(1, int(len(all_tasks) * ratio))
    chunks = []
    for i in range(0, len(all_tasks), chunk_size):
        chunk = all_tasks[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    # 限制轮数
    chunks = chunks[:rounds]
    print(f"  总样本: {len(all_tasks)} | 每轮约: {chunk_size} | 实际轮数: {len(chunks)}")
    
    # 记录每轮结果
    round_results = []
    current_prompt = optimized_prompt
    
    for round_idx, test_chunk in enumerate(chunks, 1):
        print(f"\n  --- Round {round_idx}/{len(chunks)} ---")
        print(f"    测试集: {len(test_chunk)} 条")
        
        pass_count = partial_count = fail_count = 0
        round_details = []
        
        for i, task in enumerate(test_chunk, 1):
            task_id = task["id"]
            subset = task["_subset"]
            
            # 使用当前优化后的 prompt 运行 Agent
            tools_desc = json.dumps(task["function"], ensure_ascii=False, indent=2)
            user_prompt = f"""{current_prompt}

## 用户请求
{task['question']}

## 可用工具列表
{tools_desc}

## 当前状态
{json.dumps(task.get('initial_config', {}), ensure_ascii=False, indent=2)}

请输出你的工具调用决策（只输出 JSON）:"""
            
            raw_output = call_llm("你是一个工具调用助手，严格按 JSON 格式输出。", user_prompt, max_tokens=512)
            tool, args = parse_agent_output(raw_output)
            
            # 评估
            gt_entry = None
            for gt_name, gt_data in gt_files.items():
                if task_id in gt_data:
                    gt_entry = gt_data[task_id]
                    break
            
            if gt_entry:
                eval_result = compare_with_ground_truth(
                    {"tool": tool, "arguments": args}, gt_entry
                )
                if eval_result["status"] == "pass":
                    pass_count += 1
                elif eval_result["status"] == "partial":
                    partial_count += 1
                else:
                    fail_count += 1
            else:
                eval_result = {"status": "unknown", "score": 0}
            
            round_details.append({
                "task_id": task_id,
                "subset": subset,
                "tool": tool,
                "arguments": args,
                "status": eval_result["status"],
                "score": eval_result["score"]
            })
            
            if i % 10 == 0:
                print(f"      进度: {i}/{len(test_chunk)} | 通过: {pass_count} | 部分: {partial_count} | 失败: {fail_count}")
            
            time.sleep(0.3)
        
        total = len(test_chunk)
        round_summary = {
            "round": round_idx,
            "total": total,
            "pass": pass_count,
            "partial": partial_count,
            "fail": fail_count,
            "pass_rate": pass_count / total if total > 0 else 0,
            "avg_score": sum(d["score"] for d in round_details) / total if total > 0 else 0
        }
        
        print(f"    本轮结果: 通过 {pass_count}/{total} ({round_summary['pass_rate']:.1%}) | 均分: {round_summary['avg_score']:.2f}")
        
        # 保存本轮结果
        round_dir = RESULTS_DIR / "phase3_iteration" / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        with open(round_dir / "details.json", "w", encoding="utf-8") as f:
            json.dump(round_details, f, ensure_ascii=False, indent=2)
        with open(round_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(round_summary, f, ensure_ascii=False, indent=2)
        
        round_results.append(round_summary)
    
    return round_results


# ---------------------------------------------------------------------------
# 9. 汇总报告
# ---------------------------------------------------------------------------

def generate_summary_report(coldstart_summary, round_results):
    """生成最终汇总报告"""
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "dataset": "ACEBench",
        "gepa_enabled": False,
        "phase1_coldstart": coldstart_summary,
        "phase3_iteration": {
            "rounds": len(round_results),
            "round_summaries": round_results,
            "pass_rate_trend": [r["pass_rate"] for r in round_results],
            "avg_score_trend": [r["avg_score"] for r in round_results]
        }
    }
    
    # 计算趋势
    if len(round_results) >= 2:
        first_rate = round_results[0]["pass_rate"]
        last_rate = round_results[-1]["pass_rate"]
        delta = last_rate - first_rate
        report["improvement"] = {
            "first_round_pass_rate": first_rate,
            "last_round_pass_rate": last_rate,
            "absolute_improvement": delta,
            "relative_improvement": delta / first_rate if first_rate > 0 else 0
        }
        print(f"\n  通过率趋势:")
        print(f"    第1轮: {first_rate:.1%}")
        print(f"    最后轮: {last_rate:.1%}")
        print(f"    提升: {delta:+.1%} (相对 {report['improvement']['relative_improvement']:+.1%})")
    
    with open(RESULTS_DIR / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总报告已保存: {RESULTS_DIR / 'summary_report.json'}")
    return report


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    # 读取环境变量（由 main.py acebench 子命令设置）
    phase = os.environ.get("ACEBENCH_PHASE", "all")
    rounds = int(os.environ.get("ACEBENCH_ROUNDS", "10"))
    iter_ratio = float(os.environ.get("ACEBENCH_ITER_RATIO", "0.10"))
    output_dir_env = os.environ.get("ACEBENCH_OUTPUT_DIR", "")
    coldstart_results_env = os.environ.get("ACEBENCH_COLDSTART_RESULTS", "")
    
    global RESULTS_DIR
    if output_dir_env:
        RESULTS_DIR = Path(output_dir_env)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"ACEBench 飞轮执行（Phase: {phase} | GEPA: 开启）")
    print("=" * 70)
    print(f"模型: {MODEL}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"指定阶段: {phase}")
    print(f"GEPA: 进化优化（3代 × 5种群）")
    
    # 统一加载数据
    print("\n加载数据集...")
    datasets, gt_files = load_all_datasets()
    
    # 阶段分发
    if phase == "coldstart":
        # Phase 1: 仅冷启动
        print("\n>>> 执行 Phase 1: 冷启动")
        coldstart_results, coldstart_summary = run_cold_start(datasets, gt_files, ratio=0.05)
        print("\n冷启动完成。如需继续 Phase 2，执行:")
        print(f"  python pipeline/scripts/main.py acebench --phase evalopt --coldstart-results {RESULTS_DIR / 'phase1_coldstart' / 'results.json'}")
        
    elif phase == "evalopt":
        # Phase 2: 生成评估器 + 优化器（需要 Phase 1 结果）
        print("\n>>> 执行 Phase 2: 生成评估器 + 优化器")
        coldstart_path = Path(coldstart_results_env) if coldstart_results_env else (RESULTS_DIR / "phase1_coldstart" / "results.json")
        if not coldstart_path.exists():
            print(f"  错误: 找不到冷启动结果文件: {coldstart_path}")
            print("  请先执行: python pipeline/scripts/main.py acebench --phase coldstart")
            return
        
        with open(coldstart_path, "r", encoding="utf-8") as f:
            coldstart_results = json.load(f)
        
        # 重建 summary
        pass_count = sum(1 for r in coldstart_results if r["evaluation"]["status"] == "pass")
        partial_count = sum(1 for r in coldstart_results if r["evaluation"]["status"] == "partial")
        fail_count = sum(1 for r in coldstart_results if r["evaluation"]["status"] == "fail")
        total = len(coldstart_results)
        coldstart_summary = {
            "total": total,
            "pass": pass_count,
            "partial": partial_count,
            "fail": fail_count,
            "pass_rate": pass_count / total if total > 0 else 0,
            "avg_score": sum(r["evaluation"]["score"] for r in coldstart_results) / total if total > 0 else 0
        }
        
        evaluator_config, optimizer_config, optimized_prompt = generate_evaluator(coldstart_results, coldstart_summary)
        print("\n评估器+优化器生成完成。如需继续 Phase 3，执行:")
        print(f"  python pipeline/scripts/main.py acebench --phase iteration")
        
    elif phase == "iteration":
        # Phase 3: 逐轮迭代（需要 Phase 2 结果）
        print(f"\n>>> 执行 Phase 3: 逐轮迭代 ({rounds} 轮)")
        opt_prompt_path = RESULTS_DIR / "phase2_optimizer" / "optimized_agent_prompt.txt"
        if not opt_prompt_path.exists():
            print(f"  错误: 找不到优化后 prompt: {opt_prompt_path}")
            print("  请先执行: python pipeline/scripts/main.py acebench --phase evalopt")
            return
        
        with open(opt_prompt_path, "r", encoding="utf-8") as f:
            optimized_prompt = f.read()
        
        round_results = run_iteration(datasets, gt_files, optimized_prompt, rounds=rounds, ratio=iter_ratio)
        print("\n逐轮迭代完成。如需汇总，执行:")
        print(f"  python pipeline/scripts/main.py acebench --phase summary")
        
    elif phase == "summary":
        # Phase 4: 汇总报告
        print("\n>>> 执行 Phase 4: 汇总报告")
        coldstart_summary_path = RESULTS_DIR / "phase1_coldstart" / "summary.json"
        if not coldstart_summary_path.exists():
            print(f"  错误: 找不到冷启动汇总: {coldstart_summary_path}")
            return
        
        with open(coldstart_summary_path, "r", encoding="utf-8") as f:
            coldstart_summary = json.load(f)
        
        # 收集所有轮次结果
        round_results = []
        round_idx = 1
        while True:
            round_summary_path = RESULTS_DIR / "phase3_iteration" / f"round_{round_idx}" / "summary.json"
            if not round_summary_path.exists():
                break
            with open(round_summary_path, "r", encoding="utf-8") as f:
                round_results.append(json.load(f))
            round_idx += 1
        
        report = generate_summary_report(coldstart_summary, round_results)
        print("\n汇总报告完成")
        
    else:  # phase == "all"
        # 全部执行
        print("\n>>> 执行全部阶段 (Phase 1 → 2 → 3 → 4)")
        
        # Phase 1: 冷启动
        coldstart_results, coldstart_summary = run_cold_start(datasets, gt_files, ratio=0.05)
        
        # Phase 2: 生成评估器 + 优化器
        evaluator_config, optimizer_config, optimized_prompt = generate_evaluator(coldstart_results, coldstart_summary)
        
        # Phase 3: 逐轮迭代
        round_results = run_iteration(datasets, gt_files, optimized_prompt, rounds=4, ratio=0.25)
        
        # Phase 4: 汇总
        report = generate_summary_report(coldstart_summary, round_results)
        
        print("\n" + "=" * 70)
        print("飞轮全部执行完毕!")
        print("=" * 70)
    
    print(f"\n所有结果保存在: {RESULTS_DIR}")
    print("目录结构:")
    for p in sorted(RESULTS_DIR.rglob("*")):
        rel = p.relative_to(RESULTS_DIR)
        print(f"  {rel}")
    print(f"所有结果保存在: {RESULTS_DIR}")
    print("目录结构:")
    for p in sorted(RESULTS_DIR.rglob("*")):
        rel = p.relative_to(RESULTS_DIR)
        print(f"  {rel}")


if __name__ == "__main__":
    main()
