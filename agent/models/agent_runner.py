#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acebench_agent_runner.py v1 — 真实 ACEBench 工具 + 硅基流动 LLM
===========================================================

取代 dumb_agent_runner.py，接入:
1. ACEBenchSandbox — 真实工具执行环境
2. SiliconFlowLLM  — 真实模型调用

输出格式保持与评估器兼容的 step-based 轨迹。

Usage:
    export SILICONFLOW_API_KEY=sk-xxx
    python agent/models/agent_runner.py \
        --input data/agent_tasks.jsonl \
        --output output/agent_traces.jsonl \
        --model Qwen/Qwen2.5-14B-Instruct \
        --max-steps 15
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.acebench_sandbox import ACEBenchSandbox


# ──────────────────────────────────────────
# SiliconFlow LLM 客户端
# ──────────────────────────────────────────

class SiliconFlowLLM:
    """硅基流动 API 客户端"""
    
    def __init__(self, model: str = "Qwen/Qwen2.5-14B-Instruct", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.call_count = 0
        self.total_tokens = 0
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2048) -> Tuple[str, int, int]:
        """
        调用硅基流动 API。
        
        Args:
            messages: OpenAI 格式消息列表 [{role, content}]
            temperature: 采样温度
            max_tokens: 最大输出长度
        
        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        if not self.api_key:
            raise ValueError("缺少 API Key。请设置 SILICONFLOW_API_KEY 环境变量或通过 --api-key 传入")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        start_time = time.time()
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 请求失败: {e}")
        
        latency_ms = int((time.time() - start_time) * 1000)
        data = resp.json()
        
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"API 返回异常: {data}")
        
        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        self.call_count += 1
        self.total_tokens += prompt_tokens + completion_tokens
        
        return content, prompt_tokens, completion_tokens


# ──────────────────────────────────────────
# Agent: ReAct 循环 + 真实工具
# ──────────────────────────────────────────

class ACEBenchAgent:
    """
    基于 ReAct 的 Agent，调用 ACEBenchSandbox 真实工具。
    """
    
    def __init__(self, llm: SiliconFlowLLM, max_steps: int = 15):
        self.llm = llm
        self.max_steps = max_steps
    
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个 ACEBench 任务。
        
        Args:
            task: 包含 id, question, initial_config, function 的任务定义
        
        Returns:
            step-based 轨迹字典
        """
        task_id = task.get("id", "unknown")
        question = task.get("question", "")
        available_functions = task.get("function", [])
        
        # 初始化沙箱
        sandbox = ACEBenchSandbox(task)
        
        # 构建系统提示
        system_prompt = self._build_system_prompt(available_functions)
        
        steps = []
        final_output = ""
        
        for step_idx in range(self.max_steps):
            # 构建当前步骤的用户提示
            user_prompt = self._build_step_prompt(question, steps, sandbox)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # 调用 LLM
            start_time = time.time()
            try:
                response_text, prompt_tokens, completion_tokens = self.llm.chat(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
            except Exception as e:
                steps.append({
                    "step": step_idx + 1,
                    "thought": f"[LLM 调用失败: {str(e)}]",
                    "tool_call": "",
                    "observation": f"Error: {str(e)}",
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "tokens": 0,
                })
                break
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # 解析 Thought + Action
            thought, action_name, action_args = self._parse_response(response_text)
            
            # 执行工具
            if action_name:
                observation = sandbox.execute(action_name, action_args)
                obs_text = json.dumps(observation, ensure_ascii=False)
            else:
                obs_text = "无操作"
            
            steps.append({
                "step": step_idx + 1,
                "thought": thought,
                "tool_call": f"{action_name}({json.dumps(action_args, ensure_ascii=False)})" if action_name else "",
                "observation": obs_text,
                "latency_ms": latency_ms,
                "tokens": prompt_tokens + completion_tokens,
            })
            
            # 终止条件
            if action_name == "finish" or "任务已完成" in thought or "结束" in thought:
                final_output = action_args.get("answer", thought)
                break
        
        # 评估最终状态
        # 如果有 ground_truth 传入，做对比（但通常 task 里不含 ground_truth，需外部加载）
        state_score = 0.0
        
        return {
            "id": task_id,
            "category": task.get("category", "acebench_agent"),
            "goal": question,
            "steps": steps,
            "output": final_output,
            "ground_truth": "",  # 由外部评估时填入
            "outcome": "待评估",  # 由评估器填入
            "num_steps": len(steps),
            "final_state": sandbox.get_state_snapshot(),
            "llm_calls": self.llm.call_count,
            "total_tokens": self.llm.total_tokens,
        }
    
    def _build_system_prompt(self, functions: List[Dict[str, Any]]) -> str:
        """构建系统提示，包含所有可用工具定义"""
        lines = [
            "你是一个智能助手，需要帮用户完成多步任务。",
            "",
            "你可以使用以下工具:",
        ]
        
        for fn in functions:
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            
            # 格式化参数
            param_str = self._format_params(params)
            lines.append(f"- {name}({param_str}): {desc}")
        
        lines.extend([
            "",
            "规则:",
            "1. 每步先给出思考过程 (Thought)，再给出工具调用 (Action)",
            "2. Action 格式: 工具名(参数JSON)，例如: login_food_platform({\"username\": \"Eve\", \"password\": \"password123\"})",
            "3. 如果任务完成，调用 finish({\"answer\": \"结果描述\"})",
            "4. 注意检查前置条件（如 WiFi 是否开启、是否已登录）",
            "5. 参数必须从任务描述中提取，不要编造",
        ])
        
        return "\n".join(lines)
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """格式化参数定义"""
        if not params:
            return ""
        props = params.get("properties", {})
        required = params.get("required", [])
        parts = []
        for k, v in props.items():
            req_mark = "*" if k in required else ""
            ptype = v.get("type", "any")
            parts.append(f"{k}{req_mark}: {ptype}")
        return ", ".join(parts)
    
    def _build_step_prompt(self, question: str, steps: List[Dict], sandbox: ACEBenchSandbox) -> str:
        """构建当前步骤的用户提示"""
        lines = [
            f"任务: {question}",
            "",
        ]
        
        # 当前环境状态摘要
        state = sandbox.get_state_snapshot()
        lines.append("当前状态:")
        lines.append(f"- WiFi: {'开启' if state['BaseApi']['wifi'] else '关闭'}")
        lines.append(f"- 设备登录: {'已登录' if state['BaseApi']['logged_in'] else '未登录'}")
        lines.append(f"- 外卖平台登录用户: {state['FoodPlatform']['logged_in_users']}")
        lines.append(f"- 订单数: {len(state['FoodPlatform']['orders'])}")
        lines.append(f"- 消息数: {len(state['MessageApi']['inbox'])}")
        lines.append(f"- 提醒数: {len(state['ReminderAPI']['reminder_list'])}")
        lines.append("")
        
        # 历史步骤
        if steps:
            lines.append("历史操作:")
            for s in steps[-5:]:  # 只显示最近 5 步
                lines.append(f"Step {s['step']}: Thought: {s['thought']}")
                lines.append(f"  Action: {s['tool_call']}")
                lines.append(f"  Observation: {s['observation'][:200]}")  # 截断
                lines.append("")
        
        lines.append("请给出下一步的 Thought 和 Action。格式:")
        lines.append("Thought: ...")
        lines.append("Action: 工具名({参数})")
        
        return "\n".join(lines)
    
    def _parse_response(self, text: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        解析 LLM 响应，提取 Thought 和 Action。
        
        Returns:
            (thought, action_name, action_args)
        """
        thought = ""
        action_name = ""
        action_args = {}
        
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', text, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip()
            action_name, action_args = self._parse_action(action_str)
        
        return thought, action_name, action_args
    
    def _parse_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析 Action 字符串。
        格式: tool_name({...}) 或 tool_name(key=value, ...)
        """
        # 尝试匹配 tool_name({json})
        m = re.match(r'(\w+)\s*\((\{.*\})\)', action_str)
        if m:
            name = m.group(1)
            try:
                args = json.loads(m.group(2))
                return name, args
            except json.JSONDecodeError:
                pass
        
        # 尝试匹配 tool_name(key=value, ...)
        m = re.match(r'(\w+)\s*\((.*)\)', action_str)
        if m:
            name = m.group(1)
            inner = m.group(2).strip()
            args = {}
            # 简单解析 key=value 对
            for pair in re.findall(r'(\w+)\s*=\s*([^,]+)', inner):
                k, v = pair
                v = v.strip().strip('"').strip("'")
                # 尝试转数字、布尔值
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.isdigit():
                    v = int(v)
                args[k] = v
            return name, args
        
        # fallback: 只提取工具名
        m = re.match(r'(\w+)', action_str)
        if m:
            return m.group(1), {}
        
        return "", {}


# ──────────────────────────────────────────
# IO 工具
# ──────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
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
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="运行 ACEBench Agent（真实工具 + 真实 LLM）")
    parser.add_argument("--input", required=True, help="输入任务 JSONL")
    parser.add_argument("--output", required=True, help="输出轨迹 JSONL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="模型名")
    parser.add_argument("--api-key", default="", help="API Key")
    parser.add_argument("--max-steps", type=int, default=15, help="每任务最大步数")
    parser.add_argument("--limit", type=int, default=0, help="限制任务数")
    parser.add_argument("--ground-truth", default="", help="ground_truth JSON/JSONL 路径（可选，用于状态评估）")
    args = parser.parse_args()
    
    # 加载任务
    tasks = load_jsonl(args.input)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"[INFO] 加载 {len(tasks)} 个任务")
    
    # 加载 ground_truth（如果有）
    ground_truths = {}
    if args.ground_truth:
        gt_data = load_jsonl(args.ground_truth)
        for gt in gt_data:
            gid = gt.get("id")
            if gid:
                ground_truths[gid] = gt.get("ground_truth", [])
        print(f"[INFO] 加载 {len(ground_truths)} 条 ground_truth")
    
    # 初始化 LLM 和 Agent
    llm = SiliconFlowLLM(model=args.model, api_key=args.api_key)
    agent = ACEBenchAgent(llm, max_steps=args.max_steps)
    
    results = []
    for i, task in enumerate(tasks):
        print(f"\n[Progress] 任务 {i+1}/{len(tasks)}: {task.get('id', 'unknown')}")
        print(f"  Question: {task.get('question', '')[:80]}...")
        
        result = agent.run(task)
        
        # 如果有 ground_truth，做状态对比
        gt = ground_truths.get(task.get("id"))
        if gt:
            # 重建沙箱来对比（因为 agent.run 后沙箱被销毁了，需要重新执行或保存状态）
            # 简化：这里只在轨迹中标记有 ground_truth，评估阶段再对比
            result["ground_truth"] = "loaded"
        
        results.append(result)
        print(f"  完成: {result['num_steps']} 步, LLM 调用: {llm.call_count}, Tokens: {llm.total_tokens}")
    
    save_jsonl(results, args.output)
    print(f"\n[OK] 保存 {len(results)} 条轨迹 -> {args.output}")
    
    # 汇总
    total_steps = sum(r["num_steps"] for r in results)
    print(f"[Summary] 总步数: {total_steps}, 平均步数: {total_steps/len(results):.1f}")
    print(f"[Summary] 总 LLM 调用: {llm.call_count}, 总 Tokens: {llm.total_tokens}")


if __name__ == "__main__":
    main()
