#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACEBench 沙箱集成 Runner
========================

将 Agent 的 JSON 格式工具调用转译为 ACEBench 沙箱格式，
执行多步沙箱模拟，返回 JSON 格式的 observation，
记录每步的 latency_ms 和 tokens（估算）。

使用方式:
    from sandbox_runner import ACEBenchSandboxRunner
    
    runner = ACEBenchSandboxRunner()
    result = runner.run_task(task, llm_client, max_steps=8)
"""

import json
import os
import sys
import time
import re
import ast
import copy
import importlib
import types
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_ROOT / "agent/sandbox/acebench"

sys.path.insert(0, str(SANDBOX_DIR))

# ---------------------------------------------------------------------------
# 注册 model_inference 包（沙箱代码需要这个导入路径）
# ---------------------------------------------------------------------------
def _register_pkg(name):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    return sys.modules[name]

_register_pkg("model_inference")

for prefix in ["multi_step", "multi_turn"]:
    _register_pkg(f"model_inference.{prefix}")
    _register_pkg(f"model_inference.{prefix}.scenarioszh")
    _register_pkg(f"model_inference.{prefix}.scenarioszh.phone_platform")

# 注册 base_api
_base_api = importlib.import_module("multi_step.phone_platform.base_api")
for prefix in ["multi_step", "multi_turn"]:
    sys.modules[f"model_inference.{prefix}.scenarioszh.phone_platform.base_api"] = _base_api

# 注册其他 phone_platform 模块
for mod_name in ["food_services", "message", "reminder"]:
    _mod = importlib.import_module(f"multi_step.phone_platform.{mod_name}")
    for prefix in ["multi_step", "multi_turn"]:
        sys.modules[f"model_inference.{prefix}.scenarioszh.phone_platform.{mod_name}"] = _mod

# 注册 travel
_travel = importlib.import_module("multi_step.scenarioszh.travel")
for prefix in ["multi_step", "multi_turn"]:
    sys.modules[f"model_inference.{prefix}.scenarioszh.travel"] = _travel

# 注册顶层工具模块
for mod_name in ["multi_step_utils", "execution_role_step", "multi_step_scene"]:
    _mod = importlib.import_module(f"multi_step.{mod_name}")
    for prefix in ["multi_step", "multi_turn"]:
        sys.modules[f"model_inference.{prefix}.{mod_name}"] = _mod

# 修改 CLASS_FILE_PATH_MAPPING
import multi_step.multi_step_utils as _utils
_utils.CLASS_FILE_PATH_MAPPING_ZH = {
    "BaseApi": "model_inference.multi_step.scenarioszh.phone_platform.base_api",
    "MessageApi": "model_inference.multi_step.scenarioszh.phone_platform.message",
    "ReminderApi": "model_inference.multi_step.scenarioszh.phone_platform.reminder",
    "FoodPlatform": "model_inference.multi_step.scenarioszh.phone_platform.food_services",
    "Travel": "model_inference.multi_step.scenarioszh.travel",
}

# ---------------------------------------------------------------------------
# 导入沙箱核心模块
# ---------------------------------------------------------------------------
from multi_step.multi_step_utils import execute_agent_func_call
from multi_step.execution_role_step import EXECUTION_STEP


# ---------------------------------------------------------------------------
# JSON ↔ ACEBench 格式转译
# ---------------------------------------------------------------------------

def json_to_acebench_format(tool_name: str, arguments: dict) -> str:
    if not tool_name or tool_name == "null":
        return "[]"
    
    def _serialize(value):
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            items = [_serialize(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            items = [f"{_serialize(k)}: {_serialize(v)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return str(value)
    
    args_str = ", ".join(f"{k}={_serialize(v)}" for k, v in arguments.items())
    return f"[{tool_name}({args_str})]"


def acebench_to_json(acebench_str: str) -> Tuple[Optional[str], dict]:
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


# ---------------------------------------------------------------------------
# 沙箱 Runner
# ---------------------------------------------------------------------------

class ACEBenchSandboxRunner:
    def __init__(self, language: str = "zh"):
        self.language = language
        self.execution_role = EXECUTION_STEP(
            agent_model_name="sandbox_runner",
            initial_config={},
            involved_classes=[],
            test_id="",
            language=language
        )
    
    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
    
    def execute_step(self, tool_name: str, arguments: dict,
                     initial_config: dict, involved_classes: list,
                     test_id: str) -> Tuple[str, float, int, int, dict]:
        """执行单步，返回 (observation, latency_ms, prompt_tokens, completion_tokens, involved_instances)"""
        
        acebench_format = json_to_acebench_format(tool_name, arguments)
        start_time = time.time()
        involved_instances = {}
        
        try:
            execution_list = self.execution_role.decode_function_list(acebench_format)
        except Exception:
            # Fallback
            tool, args = acebench_to_json(acebench_format)
            if tool:
                args_parts = []
                for k, v in args.items():
                    if isinstance(v, str):
                        args_parts.append(f"{k}='{v}'")
                    else:
                        args_parts.append(f"{k}={v}")
                execution_list = [f"{tool}({', '.join(args_parts)})"]
            else:
                execution_list = []
        
        try:
            if execute_agent_func_call and execution_list:
                results, involved_instances = execute_agent_func_call(
                    func_call_list=execution_list,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name="sandbox_runner",
                    test_entry_id=test_id,
                    language=self.language
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    try:
                        parsed = json.loads(result)
                        observation = json.dumps(parsed, ensure_ascii=False)
                    except:
                        observation = json.dumps({"message": result}, ensure_ascii=False)
                else:
                    observation = json.dumps({"message": "No result"}, ensure_ascii=False)
            else:
                observation = json.dumps({"message": "No execution"}, ensure_ascii=False)
        except Exception as e:
            observation = json.dumps({"error": str(e)}, ensure_ascii=False)
        
        latency_ms = (time.time() - start_time) * 1000
        prompt_tokens = self._estimate_tokens(json.dumps(arguments, ensure_ascii=False))
        completion_tokens = self._estimate_tokens(observation)
        
        return observation, latency_ms, prompt_tokens, completion_tokens, involved_instances
    
    def get_sandbox_state(self, involved_instances: dict) -> dict:
        state = {}
        for class_name, instance in involved_instances.items():
            try:
                if hasattr(instance, 'state'):
                    state[class_name] = copy.deepcopy(instance.state)
                elif hasattr(instance, '_state'):
                    state[class_name] = copy.deepcopy(instance._state)
                else:
                    state[class_name] = {k: v for k, v in instance.__dict__.items() if not k.startswith('_')}
            except Exception as e:
                state[class_name] = {"error": str(e)}
        return state
    
    def reset_sandbox(self, involved_classes: list, test_id: str):
        import multi_step.multi_step_utils as utils_module
        for class_name in involved_classes:
            instance_name = f"sandbox_runner_{test_id}_{class_name.lower()}_instance"
            if hasattr(utils_module, instance_name):
                delattr(utils_module, instance_name)
    
    def run_task(self, task: dict, llm_call_fn, max_steps: int = 8,
                 system_prompt: str = "") -> dict:
        task_id = task.get("id", "unknown")
        question = task.get("question", "")
        initial_config = task.get("initial_config", {})
        functions = task.get("function", [])
        involved_classes = task.get("involved_classes", [])
        gt_entry = task.get("_ground_truth", {})
        
        self.reset_sandbox(involved_classes, task_id)
        
        tools_desc = json.dumps(functions, ensure_ascii=False, indent=2)
        conversation = [
            {"role": "system", "content": system_prompt or "你是一个智能助手，需要根据用户请求调用工具完成任务。"},
            {"role": "user", "content": question}
        ]
        
        steps = []
        all_instances = {}
        
        for step_id in range(1, max_steps + 1):
            current_prompt = self._build_step_prompt(question, tools_desc, conversation, functions)
            
            llm_start = time.time()
            raw_output = llm_call_fn(conversation[0]["content"], current_prompt)
            llm_latency = (time.time() - llm_start) * 1000
            
            thought, tool_name, arguments = self._parse_llm_output(raw_output)
            state_before = self.get_sandbox_state(all_instances)
            
            if tool_name and tool_name != "null":
                observation, exec_latency, prompt_tokens, completion_tokens, step_instances = \
                    self.execute_step(tool_name, arguments, initial_config, involved_classes, task_id)
                
                all_instances.update(step_instances)
                state_after = self.get_sandbox_state(all_instances)
                
                steps.append({
                    "step_id": step_id,
                    "thoughts": thought,
                    "tool_calls": {"tool": tool_name, "arguments": arguments},
                    "observations": observation,
                    "latency_ms": round(llm_latency + exec_latency, 2),
                    "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                    "state_before": state_before,
                    "state_after": state_after
                })
                
                conversation.append({"role": "assistant", "content": raw_output})
                conversation.append({"role": "user", "content": f"工具执行结果: {observation}\n如果任务已完成，请输出 {{\"tool\": null, \"arguments\": {{}}}}。"})
            else:
                steps.append({
                    "step_id": step_id,
                    "thoughts": thought,
                    "tool_calls": {"tool": None, "arguments": {}},
                    "observations": json.dumps({"message": "Task finished"}, ensure_ascii=False),
                    "latency_ms": round(llm_latency, 2),
                    "tokens": {
                        "prompt": self._estimate_tokens(current_prompt),
                        "completion": self._estimate_tokens(raw_output)
                    },
                    "state_before": state_before,
                    "state_after": state_before
                })
                break
        
        final_state = self.get_sandbox_state(all_instances)
        outcome, outcome_reason = self._evaluate_final_state(
            final_state, gt_entry, steps, task.get("mile_stone", [])
        )
        
        return {
            "task_id": task_id,
            "question": question,
            "steps": steps,
            "final_state": final_state,
            "ground_truth": gt_entry.get("ground_truth", {}) if isinstance(gt_entry, dict) else {},
            "outcome": outcome,
            "outcome_reason": outcome_reason
        }
    
    def _build_step_prompt(self, question, tools_desc, conversation, functions):
        history = ""
        for msg in conversation[1:]:
            if msg["role"] == "user":
                history += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history += f"Agent: {msg['content']}\n"
        
        return f"""你是一个智能助手，根据用户请求调用工具完成任务。

## 用户原始请求
{question}

## 可用工具列表
{tools_desc}

## 对话历史
{history}

## 当前状态
请分析当前情况，决定下一步行动：
1. 如果需要调用工具，输出 JSON: {{"tool": "工具名", "arguments": {{"参数名": "值"}}}}
2. 在 JSON 前，先用一句话说明思考过程
3. 如果任务已完成，输出: {{"tool": null, "arguments": {{}}}}

思考过程: ...
{{"tool": "...", "arguments": {{...}}}}"""
    
    def _parse_llm_output(self, raw_output: str):
        thought = ""
        tool_name = None
        arguments = {}
        
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                tool_name = result.get("tool")
                arguments = result.get("arguments", {})
                thought = re.sub(r'^思考过程[:：]\s*', '', raw_output[:json_match.start()].strip()).strip()
            except:
                pass
        
        if not tool_name:
            tool_match = re.search(r'"tool"\s*:\s*"([^"]*)"', raw_output)
            if tool_match:
                tool_name = tool_match.group(1)
                if tool_name == "null":
                    tool_name = None
        
        return thought, tool_name, arguments
    
    def _evaluate_final_state(self, final_state, gt_entry, steps, mile_stone):
        gt_state = gt_entry.get("ground_truth", {}) if isinstance(gt_entry, dict) else {}
        if not gt_state:
            return "fail", "无 ground_truth"
        
        mismatches = self._deep_compare(final_state, gt_state)
        path_match = self._compare_path(steps, mile_stone)
        
        if len(mismatches) == 0 and path_match >= 0.8:
            return "pass", f"状态完全匹配，路径匹配{path_match:.0%}"
        elif len(mismatches) <= 2 and path_match >= 0.5:
            return "partial", f"{len(mismatches)}处不匹配，路径匹配{path_match:.0%}"
        else:
            return "fail", f"{len(mismatches)}处不匹配，路径匹配{path_match:.0%}"
    
    def _deep_compare(self, actual, expected, path=""):
        mismatches = []
        if isinstance(expected, dict):
            for k, v in expected.items():
                av = actual.get(k) if isinstance(actual, dict) else None
                if av is None and k not in (actual or {}):
                    mismatches.append(f"{path}.{k}: 缺失")
                else:
                    mismatches.extend(self._deep_compare(av, v, f"{path}.{k}"))
            if isinstance(actual, dict):
                for k in actual:
                    if k not in expected:
                        mismatches.append(f"{path}.{k}: 多余")
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                mismatches.append(f"{path}: 长度不匹配")
            else:
                for i, (a, e) in enumerate(zip(actual, expected)):
                    mismatches.extend(self._deep_compare(a, e, f"{path}[{i}]"))
        else:
            if actual != expected:
                mismatches.append(f"{path}: 值不匹配 (实际 {actual} vs 期望 {expected})")
        return mismatches
    
    def _compare_path(self, steps, mile_stone):
        if not mile_stone:
            return 1.0
        
        actual = [s["tool_calls"]["tool"] for s in steps if s["tool_calls"].get("tool")]
        ideal = []
        for ms in mile_stone:
            if isinstance(ms, str):
                t, _ = acebench_to_json(ms)
                if t: ideal.append(t)
            elif isinstance(ms, list) and ms:
                t, _ = acebench_to_json(ms[0])
                if t: ideal.append(t)
        
        if not ideal:
            return 1.0
        
        m, n = len(actual), len(ideal)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if actual[i-1] == ideal[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / max(m, n) if max(m, n) > 0 else 0.0


def run_single_task(task: dict, llm_call_fn, max_steps: int = 8) -> dict:
    runner = ACEBenchSandboxRunner()
    return runner.run_task(task, llm_call_fn, max_steps=max_steps)
