#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dumb_agent_runner.py v5 — step-based 结构 + question 字段
每步记录: {step, thought, tool_call, observation, latency_ms, tokens}
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List

MOCK_TOOLS = {
    "search": {"desc": "搜索引擎", "params": {"query": "string"}},
    "click": {"desc": "点击元素", "params": {"element": "string"}},
    "type": {"desc": "输入文本", "params": {"text": "string", "field": "string"}},
    "scroll": {"desc": "滚动页面", "params": {"direction": "up|down"}},
    "finish": {"desc": "完成任务", "params": {"answer": "string"}},
    "ask": {"desc": "向用户追问", "params": {"goal": "string"}},
}


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


class MockLLM:
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.call_count = 0

    def chat(self, prompt: str) -> str:
        self.call_count += 1
        task = ""
        if "任务:" in prompt:
            task = prompt.split("任务:")[1].split("\n")[0].strip()
        action = self._heuristic_action(task)
        thoughts = [
            "我需要完成这个任务。", "让我看看应该怎么做。", "先搜索一下相关信息。",
            "直接尝试操作。", "不太确定，试试看。",
        ]
        thought = random.choice(thoughts)
        return thought + "\nAction: " + action

    def _heuristic_action(self, task: str) -> str:
        if random.random() < 0.5:
            tool_name = random.choice(list(MOCK_TOOLS.keys()))
        else:
            if "搜索" in task or "查" in task or "找" in task:
                tool_name = "search"
            elif "点击" in task or "打开" in task:
                tool_name = "click"
            elif "输入" in task or "填" in task:
                tool_name = "type"
            else:
                tool_name = random.choice(list(MOCK_TOOLS.keys()))

        if tool_name == "search":
            q = task[:20] if random.random() > 0.3 else ""
            return 'search({"query": "' + str(q) + '"})'
        elif tool_name == "click":
            element = random.choice(["button", "link", "", "unknown"])
            return 'click({"element": "' + str(element) + '"})'
        elif tool_name == "type":
            text = task[:10] if random.random() > 0.3 else ""
            field = random.choice(["input", "search_box", "", "field1"])
            return 'type({"text": "' + str(text) + '", "field": "' + str(field) + '"})'
        elif tool_name == "scroll":
            direction = random.choice(["up", "down", "left"])
            return 'scroll({"direction": "' + str(direction) + '"})'
        elif tool_name == "finish":
            answer = "已完成" if random.random() > 0.5 else ""
            return 'finish({"answer": "' + str(answer) + '"})'
        elif tool_name == "ask":
            return 'ask({"goal": "请问更多信息"})'
        return 'finish({"answer": ""})'


class DumbAgent:
    def __init__(self, llm: MockLLM, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max_steps

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        question = task.get("goal", "") or task.get("query", "")
        ground_truth = task.get("ground_truth", "")
        task_id = task.get("id", "unknown")
        category = task.get("category", "unknown")

        steps = []
        final_output = ""
        outcome = "失败"

        for step_idx in range(self.max_steps):
            prompt = self._build_prompt(question, steps)
            response = self.llm.chat(prompt)
            thought, action = self._parse_response(response)
            observation = self._execute_action(action, question)

            # 记录 step (v5 新结构)
            start_time = time.time()
            steps.append({
                "step": step_idx + 1,
                "thought": thought,
                "tool_call": action,
                "observation": observation,
                "latency_ms": int((time.time() - start_time) * 1000),
                "tokens": random.randint(20, 100),  # 模拟 token 消耗
            })

            if "finish" in action.lower():
                final_output = self._extract_answer(action)
                break

        if final_output and ground_truth and final_output in str(ground_truth):
            outcome = "通过"
        elif final_output:
            outcome = "部分通过"
        else:
            outcome = "失败"

        return {
            "id": task_id,
            "category": category,
            "goal": question,
            "steps": steps,
            "output": final_output,
            "ground_truth": ground_truth,
            "outcome": outcome,
            "num_steps": len(steps),
        }

    def _build_prompt(self, goal: str, steps: List[Dict]) -> str:
        history = ""
        for s in steps[-3:]:
            history += "Step " + str(s['step']) + ": Thought: " + s['thought'] + "\nAction: " + s['tool_call'] + "\nObservation: " + s['observation'] + "\n\n"
        return """你正在完成一个任务。

任务: """ + goal + """

可用工具:
- search(query): 搜索引擎
- click(element): 点击元素
- type(text, field): 输入文本
- scroll(direction): 滚动页面
- finish(answer): 完成任务
- ask(question): 向用户追问

当前步骤: """ + str(len(steps)) + """
历史:
""" + history + """

请给出你的思考（Thought）和下一步操作（Action）。格式:
Thought: ...
Action: ...
"""

    def _parse_response(self, response: str) -> tuple:
        thought = ""
        action = ""
        if "Thought:" in response:
            parts = response.split("Action:")
            thought = parts[0].replace("Thought:", "").strip()
            if len(parts) > 1:
                action = parts[1].strip()
        else:
            lines = response.strip().split("\n")
            thought = lines[0] if lines else ""
            action = lines[-1] if len(lines) > 1 else ""
        return thought, action

    def _execute_action(self, action: str, goal: str) -> str:
        if random.random() < 0.4:
            errors = ["Error: 元素未找到", "Error: 网络超时", "Error: 参数无效", "Error: 页面加载失败", ""]
            return random.choice(errors)
        if "search" in action.lower():
            return "搜索结果: 关于 '" + goal[:10] + "' 找到 3 条记录（模拟数据）"
        elif "click" in action.lower():
            return "点击成功，页面跳转（模拟）"
        elif "type" in action.lower():
            return "输入成功（模拟）"
        elif "scroll" in action.lower():
            return "滚动完成（模拟）"
        elif "finish" in action.lower():
            return "任务结束"
        return "无操作"

    def _extract_answer(self, action: str) -> str:
        try:
            import re as re_local
            m = re_local.search(r'"answer"\s*:\s*"([^"]*)"', action)
            if m:
                return m.group(1)
        except:
            pass
        return action


def main():
    parser = argparse.ArgumentParser(description="运行 dumb agent 生成轨迹 (v5 step-based)")
    parser.add_argument("--tasks", required=True, help="任务定义 jsonl 文件")
    parser.add_argument("--output", default="./dumb_agent_traces.jsonl", help="输出轨迹文件")
    parser.add_argument("--model", default="mock", help="模型名称（默认mock）")
    parser.add_argument("--max-steps", type=int, default=5, help="每任务最大步数")
    parser.add_argument("--limit", type=int, default=0, help="限制任务数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    tasks = load_jsonl(args.tasks)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"[INFO] 加载 {len(tasks)} 个任务")

    llm = MockLLM(model_name=args.model)
    agent = DumbAgent(llm, max_steps=args.max_steps)

    results = []
    for i, task in enumerate(tasks):
        print(f"[Progress] 任务 {i+1}/{len(tasks)}: {task.get('id', 'unknown')}")
        result = agent.run(task)
        results.append(result)

    save_jsonl(results, args.output)
    print(f"[OK] 保存 {len(results)} 条轨迹 -> {args.output}")

    outcomes = {"通过": 0, "部分通过": 0, "失败": 0}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"[Summary] 通过: {outcomes['通过']}, 部分通过: {outcomes['部分通过']}, 失败: {outcomes['失败']}")
    print(f"[Summary] LLM 调用次数: {llm.call_count}")


if __name__ == "__main__":
    main()
