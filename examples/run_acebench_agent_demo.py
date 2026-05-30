#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACEBench Agent 运行示例 —— 展示完整输入、执行、评估流程
===============================================================

本脚本演示如何：
1. 加载 ACEBench Agent 子集数据
2. 初始化沙箱状态（基于 initial_config）
3. 用 LLM 生成工具调用决策
4. 在沙箱中执行工具（修改状态）
5. 对比 ground_truth 输出评估结果

运行方式：
    export SILICONFLOW_API_KEY=sk-xxx
    python examples/run_acebench_agent_demo.py
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 需要安装: pip install openai
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. 加载 ACEBench 数据
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh/data_agent_multi_step.json"
GT_PATH   = PROJECT_ROOT / "data_preprocessing/raw_datasets/ACEBench/data_zh/possible_answer/data_agent_multi_step.json"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


print("=" * 70)
print("步骤 1: 加载 ACEBench 数据")
print("=" * 70)

tasks = load_jsonl(DATA_PATH)
ground_truths = {gt["id"]: gt for gt in load_jsonl(GT_PATH)}

print(f"  加载任务数: {len(tasks)}")
print(f"  Ground truth 数: {len(ground_truths)}")

# 取第一条示例
task = tasks[0]
task_id = task["id"]
print(f"\n  示例任务 ID: {task_id}")

# ---------------------------------------------------------------------------
# 2. 展示输入数据
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("步骤 2: 输入数据")
print("=" * 70)

print(f"\n【用户请求 (question)】")
print(f"  {task['question']}")

print(f"\n【初始状态 (initial_config)】")
print(json.dumps(task["initial_config"], ensure_ascii=False, indent=2))

print(f"\n【可用工具 (function)】")
for i, func in enumerate(task["function"], 1):
    print(f"  {i}. {func['name']}: {func['description']}")
    # 参数简要展示
    params = func.get("parameters", {})
    if "properties" in params:
        for pname, pdef in params["properties"].items():
            required = "(required)" if pname in params.get("required", []) else ""
            print(f"      - {pname}: {pdef.get('type', 'any')} {required}")

# ---------------------------------------------------------------------------
# 3. 初始化沙箱（基于 initial_config）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("步骤 3: 初始化沙箱")
print("=" * 70)


class SimpleSandbox:
    """
    简化版 ACEBench 沙箱模拟器
    基于 initial_config 初始化，通过工具调用修改状态
    """

    def __init__(self, initial_config):
        # 深拷贝初始状态
        self.state = json.loads(json.dumps(initial_config))
        self.execution_log = []

    def execute(self, tool_name, arguments):
        """执行工具调用，修改沙箱状态，返回 observation"""
        obs = ""

        if tool_name == "turn_on_wifi":
            self.state.setdefault("BaseApi", {})["wifi"] = True
            obs = "WiFi 已开启"

        elif tool_name == "login_device":
            self.state.setdefault("BaseApi", {})["logged_in"] = True
            obs = "设备已登录"

        elif tool_name == "login_food_platform":
            username = arguments.get("username")
            password = arguments.get("password")
            # 从 state 中验证密码
            users = self.state.get("FoodPlatform", {}).get("users", {})
            user = users.get(username)
            if user and user.get("password") == password:
                self.state.setdefault("FoodPlatform", {}).setdefault("logged_in_users", [])
                if username not in self.state["FoodPlatform"]["logged_in_users"]:
                    self.state["FoodPlatform"]["logged_in_users"].append(username)
                obs = f"用户 {username} 登录成功"
            else:
                obs = f"登录失败: 用户名或密码错误"

        elif tool_name == "check_balance":
            user_name = arguments.get("user_name")
            users = self.state.get("FoodPlatform", {}).get("users", {})
            balance = users.get(user_name, {}).get("balance", 0)
            obs = f"用户 {user_name} 余额: {balance} 元"

        elif tool_name == "add_food_delivery_order":
            username = arguments.get("username")
            merchant_name = arguments.get("merchant_name")
            items = arguments.get("items", [])
            # 计算总价（简化：假设每个商品价格固定）
            total = 88.0  # 简化处理，实际应从内置菜单查询
            # 扣款
            users = self.state.setdefault("FoodPlatform", {}).setdefault("users", {})
            if username in users:
                users[username]["balance"] = users[username].get("balance", 0) - total
            # 添加订单
            orders = self.state.setdefault("FoodPlatform", {}).setdefault("orders", [])
            orders.append({
                "user_name": username,
                "merchant_name": merchant_name,
                "items": items,
                "total_price": total
            })
            obs = f"订单已提交: {merchant_name}，总价 {total} 元"

        elif tool_name == "add_reminder":
            title = arguments.get("title")
            description = arguments.get("description")
            time = arguments.get("time")
            reminders = self.state.setdefault("ReminderApi", {}).setdefault("reminder_list", {})
            rid = len(reminders) + 1
            reminders[str(rid)] = {
                "reminder_id": rid,
                "title": title,
                "description": description,
                "time": time,
                "notified": False
            }
            obs = f"提醒已添加: {title}"

        elif tool_name == "view_reminder_by_title":
            title = arguments.get("title")
            reminders = self.state.get("ReminderApi", {}).get("reminder_list", {})
            found = [r for r in reminders.values() if r.get("title") == title]
            obs = f"查询结果: {found}"

        elif tool_name == "send_message":
            sender = arguments.get("sender_name")
            receiver = arguments.get("receiver_name")
            message = arguments.get("message")
            inbox = self.state.setdefault("MessageApi", {}).setdefault("inbox", {})
            msg_id = len(inbox) + 1
            inbox[str(msg_id)] = {
                "sender_id": sender,
                "receiver_id": receiver,
                "message": message,
                "time": "2024-07-15"
            }
            obs = f"消息已发送: {sender} -> {receiver}"

        elif tool_name == "get_products":
            merchant = arguments.get("merchant_name")
            # 简化返回
            obs = f"商家 {merchant} 菜单: [超级至尊披萨 88元, 玛格丽特披萨 78元]"

        elif tool_name == "get_latest_message_id":
            inbox = self.state.get("MessageApi", {}).get("inbox", {})
            if inbox:
                latest = max(int(k) for k in inbox.keys())
                obs = f"最新消息 ID: {latest}"
            else:
                obs = "收件箱为空"

        elif tool_name == "delete_message":
            msg_id = arguments.get("message_id")
            inbox = self.state.get("MessageApi", {}).get("inbox", {})
            if str(msg_id) in inbox:
                del inbox[str(msg_id)]
                obs = f"消息 {msg_id} 已删除"
            else:
                obs = f"消息 {msg_id} 不存在"

        elif tool_name == "search_messages":
            user_name = arguments.get("user_name")
            keyword = arguments.get("keyword")
            inbox = self.state.get("MessageApi", {}).get("inbox", {})
            found = [m for m in inbox.values() if keyword in m.get("message", "")]
            obs = f"搜索结果: {len(found)} 条"

        else:
            obs = f"未知工具: {tool_name}"

        self.execution_log.append({
            "tool": tool_name,
            "args": arguments,
            "observation": obs
        })
        return obs

    def get_state(self):
        return json.loads(json.dumps(self.state))


sandbox = SimpleSandbox(task["initial_config"])
print("\n  沙箱已初始化，当前状态:")
print(json.dumps(sandbox.get_state(), ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# 4. LLM 决策 + 沙箱执行（模拟单步执行）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("步骤 4: LLM 决策 + 沙箱执行")
print("=" * 70)

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
if not API_KEY:
    print("\n  ⚠️ 未设置 SILICONFLOW_API_KEY，使用模拟输出演示")
    # 模拟 LLM 输出（正确的执行路径）
    llm_calls = [
        {"tool": "login_food_platform", "args": {"username": "Eve", "password": "password123"}},
        {"tool": "add_food_delivery_order", "args": {"username": "Eve", "merchant_name": "达美乐", "items": [{"product": "超级至尊披萨", "quantity": 1}]}},
        {"tool": "add_reminder", "args": {"title": "今日花费", "description": "今日花费88.0元", "time": "2024-07-15 09:30"}},
    ]
else:
    print("\n  ✅ 检测到 API Key，将调用 SiliconFlow LLM")
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")

    # 构建 prompt
    tools_desc = json.dumps(task["function"], ensure_ascii=False, indent=2)
    prompt = f"""你是一个智能助手，需要根据用户请求调用合适的工具完成任务。

用户请求: {task['question']}

可用工具:
{tools_desc}

当前系统状态:
{json.dumps(task['initial_config'], ensure_ascii=False, indent=2)}

请分析用户需求，选择正确的工具并填写参数。
输出格式（严格遵循）:
[tool_name(key1='value1', key2='value2', ...)]

如果一步无法完成，只输出当前这一步的工具调用。"""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct",
            messages=[
                {"role": "system", "content": "你是一个工具调用助手，严格按格式输出工具调用。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        raw_output = response.choices[0].message.content
        print(f"\n  LLM 原始输出:\n{raw_output}")
        # 简化解析（实际需用 execution_role_step.py 的 AST 解析）
        llm_calls = [{"tool": "login_food_platform", "args": {"username": "Eve", "password": "password123"}}]
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        llm_calls = []

# 执行工具调用
print(f"\n  执行计划（共 {len(llm_calls)} 步）:")
for i, call in enumerate(llm_calls, 1):
    print(f"\n  Step {i}: 调用 {call['tool']}")
    print(f"    参数: {json.dumps(call['args'], ensure_ascii=False)}")
    obs = sandbox.execute(call["tool"], call["args"])
    print(f"    返回: {obs}")
    print(f"    沙箱状态更新后:")
    print(json.dumps(sandbox.get_state(), ensure_ascii=False, indent=2)[:500] + "...")

# ---------------------------------------------------------------------------
# 5. 对比 Ground Truth
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("步骤 5: 对比 Ground Truth")
print("=" * 70)

gt = ground_truths.get(task_id, {})
print(f"\n【Ground Truth 理想最终状态】")
print(json.dumps(gt.get("ground_truth", {}), ensure_ascii=False, indent=2)[:1500])

print(f"\n【模型实际达到的最终状态】")
print(json.dumps(sandbox.get_state(), ensure_ascii=False, indent=2)[:1500])

print(f"\n【理想执行路径 (mile_stone)】")
for i, step in enumerate(gt.get("mile_stone", []), 1):
    print(f"  Step {i}: {step}")

print(f"\n【模型实际执行路径】")
for i, log in enumerate(sandbox.execution_log, 1):
    print(f"  Step {i}: {log['tool']}({json.dumps(log['args'], ensure_ascii=False)})")

# ---------------------------------------------------------------------------
# 6. 评估结论
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("步骤 6: 评估结论")
print("=" * 70)


def compare_states(actual, expected, path=""):
    """递归对比两个状态字典"""
    mismatches = []
    if isinstance(expected, dict):
        for k, v in expected.items():
            if k not in actual:
                mismatches.append(f"{path}.{k}: 缺失 (期望 {v})")
            else:
                mismatches.extend(compare_states(actual[k], v, f"{path}.{k}"))
        for k in actual:
            if k not in expected:
                mismatches.append(f"{path}.{k}: 多余 (实际 {actual[k]})")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            mismatches.append(f"{path}: 长度不匹配 (实际{len(actual)} vs 期望{len(expected)})")
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                mismatches.extend(compare_states(a, e, f"{path}[{i}]"))
    else:
        if actual != expected:
            mismatches.append(f"{path}: 值不匹配 (实际 {actual} vs 期望 {expected})")
    return mismatches


actual_state = sandbox.get_state()
expected_state = gt.get("ground_truth", [{}])[0] if gt.get("ground_truth") else {}

mismatches = compare_states(actual_state, expected_state)

if not mismatches:
    result = "✅ 通过"
    explanation = "最终状态与 Ground Truth 完全匹配。"
elif len(mismatches) <= 2:
    result = "⚠️ 部分通过"
    explanation = f"主要目标达成，但有 {len(mismatches)} 处细节不匹配。"
else:
    result = "❌ 失败"
    explanation = f"最终状态与 Ground Truth 差距较大，共 {len(mismatches)} 处不匹配。"

print(f"\n  评估结果: {result}")
print(f"  评估说明: {explanation}")

if mismatches:
    print(f"\n  不匹配详情:")
    for mm in mismatches[:5]:
        print(f"    - {mm}")
    if len(mismatches) > 5:
        print(f"    ... 还有 {len(mismatches) - 5} 处")

# 执行路径对比
print(f"\n  路径评估:")
ideal_path = gt.get("mile_stone", [])
actual_path = [f"[{log['tool']}({json.dumps(log['args'], ensure_ascii=False)})]" for log in sandbox.execution_log]
print(f"    理想步数: {len(ideal_path)}")
print(f"    实际步数: {len(actual_path)}")
if len(actual_path) == len(ideal_path):
    print(f"    ✅ 步数匹配")
else:
    print(f"    ⚠️ 步数不匹配（可能遗漏或多执行）")

print("\n" + "=" * 70)
print("示例运行完毕")
print("=" * 70)
