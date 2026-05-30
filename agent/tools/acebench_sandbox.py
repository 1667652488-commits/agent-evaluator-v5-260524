#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACEBench Sandbox — 真实工具执行环境
=====================================
模拟 ACEBench 数据集中涉及的所有 API 行为，让 Agent 调用真实 function 而非 mock。

支持的 API 类别:
- BaseApi: turn_on_wifi, login_device
- FoodPlatform: login_food_platform, check_balance, add_food_delivery_order, get_products
- MessageApi: send_message, view_messages_between_users, delete_message, search_messages, get_latest_message_id, get_earliest_message_id
- ReminderApi: add_reminder, view_reminder_by_title, search_reminders

每个任务初始化时根据 initial_config 和 ground_truth 构建初始状态，
Agent 的工具调用会修改状态，最终状态与 ground_truth 对比评估。
"""

import json
import re
import copy
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class ACEBenchSandbox:
    """
    ACEBench 沙箱环境。
    
    每个任务独立初始化一个沙箱实例，维护:
    - base_state: {wifi: bool, logged_in: bool}
    - food_platform_state: {users, logged_in_users, orders}
    - message_state: {inbox, next_message_id}
    - reminder_state: {reminder_list, next_reminder_id}
    """
    
    def __init__(self, task: Dict[str, Any]):
        """
        从 task 定义初始化沙箱状态。
        
        Args:
            task: ACEBench 任务对象，包含 initial_config, function, ground_truth(可选)
        """
        self.task_id = task.get("id", "unknown")
        self.question = task.get("question", "")
        self.available_functions = task.get("function", [])
        self.initial_config = task.get("initial_config", {})
        
        # 初始化各模块状态
        self._init_base_state()
        self._init_food_platform_state()
        self._init_message_state()
        self._init_reminder_state()
        
        # 执行轨迹记录
        self.execution_log: List[Dict[str, Any]] = []
        
    def _init_base_state(self):
        """初始化 BaseApi 状态"""
        base = self.initial_config.get("BaseApi", {})
        self.base_state = {
            "wifi": base.get("wifi", False),
            "logged_in": base.get("logged_in", False),
        }
    
    def _init_food_platform_state(self):
        """初始化 FoodPlatform 状态 — 从 ground_truth 推断或默认值"""
        # 默认用户表（根据数据集常见用户）
        self.food_platform_state = {
            "users": {
                "Eve": {"user_id": "U100", "password": "password123", "balance": 500.0},
                "Frank": {"user_id": "U101", "password": "password456", "balance": 300.0},
                "Grace": {"user_id": "U102", "password": "password789", "balance": 150.0},
                "Helen": {"user_id": "U103", "password": "password321", "balance": 800.0},
                "Isaac": {"user_id": "U104", "password": "password654", "balance": 400.0},
                "Jack": {"user_id": "U105", "password": "password654", "balance": 120.0},
            },
            "logged_in_users": [],
            "orders": [],
        }
        # 如果 initial_config 暗示已登录，同步
        if self.base_state.get("logged_in", False):
            # 从 question 中提取当前用户
            user = self._extract_user_from_question()
            if user and user in self.food_platform_state["users"]:
                self.food_platform_state["logged_in_users"].append(user)
    
    def _init_message_state(self):
        """初始化 MessageApi 状态 — 预置 ACEBench 标准短信记录"""
        self.message_state = {
            "inbox": {
                "1": {
                    "sender_id": "USR100",
                    "receiver_id": "USR101",
                    "message": "Hey Frank, don't forget about our meeting on 2024-06-11 at 4 PM in Conference Room 1.",
                    "time": "2024-06-09",
                },
                "2": {
                    "sender_id": "USR101",
                    "receiver_id": "USR102",
                    "message": '你能帮我点一个"玛格丽特披萨"的外卖吗,商家是达美乐。',
                    "time": "2024-03-09",
                },
                "3": {
                    "sender_id": "USR102",
                    "receiver_id": "USR103",
                    "message": "帮我查一些喜茶有哪些奶茶外卖，买一杯便宜些的奶茶。买完以后记得回复我,回复的内容是（已经买好了）",
                    "time": "2023-12-05",
                },
                "4": {
                    "sender_id": "USR103",
                    "receiver_id": "USR102",
                    "message": "No problem Helen, I can assist you.",
                    "time": "2024-09-09",
                },
                "5": {
                    "sender_id": "USR104",
                    "receiver_id": "USR105",
                    "message": "Isaac, are you available for a call?",
                    "time": "2024-06-06",
                },
                "6": {
                    "sender_id": "USR105",
                    "receiver_id": "USR104",
                    "message": "Yes Jack, let's do it in 30 minutes.",
                    "time": "2024-01-15",
                },
            },
            "next_message_id": 7,
        }
    
    def _init_reminder_state(self):
        """初始化 ReminderApi 状态"""
        self.reminder_state = {
            "reminder_list": {},
            "next_reminder_id": 1001,
        }
    
    def _extract_user_from_question(self) -> Optional[str]:
        """从 question 中提取用户名"""
        patterns = [
            r'我是(\w+)',
            r'我(\w+)，',
            r'username["\']?\s*[:=]\s*["\']?(\w+)',
        ]
        for pat in patterns:
            m = re.search(pat, self.question)
            if m:
                return m.group(1)
        return None
    
    # ──────────────────────────────────────────
    # 工具执行入口
    # ──────────────────────────────────────────
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一个工具调用，返回 observation。
        
        Args:
            tool_name: 工具名称
            arguments: 参数字典
        
        Returns:
            {status: ok|error, result: ..., message: ...}
        """
        self.execution_log.append({
            "tool": tool_name,
            "args": arguments,
            "base_before": copy.deepcopy(self.base_state),
            "food_before": copy.deepcopy(self.food_platform_state),
            "msg_before": copy.deepcopy(self.message_state),
            "reminder_before": copy.deepcopy(self.reminder_state),
        })
        
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            result = {"status": "error", "message": f"未知工具: {tool_name}"}
        else:
            try:
                result = handler(**arguments)
            except Exception as e:
                result = {"status": "error", "message": f"执行异常: {str(e)}"}
        
        self.execution_log[-1]["result"] = result
        return result
    
    # ──────────────────────────────────────────
    # BaseApi 工具实现
    # ──────────────────────────────────────────
    
    def _tool_turn_on_wifi(self) -> Dict[str, Any]:
        self.base_state["wifi"] = True
        return {"status": "ok", "result": {"wifi": True}, "message": "WiFi 已开启"}
    
    def _tool_login_device(self) -> Dict[str, Any]:
        self.base_state["logged_in"] = True
        return {"status": "ok", "result": {"logged_in": True}, "message": "设备登录成功"}
    
    # ──────────────────────────────────────────
    # FoodPlatform 工具实现
    # ──────────────────────────────────────────
    
    def _tool_login_food_platform(self, username: str, password: str) -> Dict[str, Any]:
        user = self.food_platform_state["users"].get(username)
        if not user:
            return {"status": "error", "message": f"用户 {username} 不存在"}
        if user["password"] != password:
            return {"status": "error", "message": "密码错误"}
        if username not in self.food_platform_state["logged_in_users"]:
            self.food_platform_state["logged_in_users"].append(username)
        return {"status": "ok", "result": {"user": username, "logged_in": True}, "message": f"{username} 登录外卖平台成功"}
    
    def _tool_check_balance(self, user_name: str) -> Dict[str, Any]:
        user = self.food_platform_state["users"].get(user_name)
        if not user:
            return {"status": "error", "message": f"用户 {user_name} 不存在"}
        return {"status": "ok", "result": {"user": user_name, "balance": user["balance"]}, "message": f"{user_name} 余额: {user['balance']} 元"}
    
    def _tool_get_products(self, merchant_name: str) -> Dict[str, Any]:
        # 模拟商家商品列表
        menus = {
            "达美乐": [
                {"product": "玛格丽特披萨", "price": 68.0},
                {"product": "超级至尊披萨", "price": 88.0},
            ],
            "米村拌饭": [
                {"product": "石锅拌饭", "price": 35.0},
                {"product": "韩式牛肉拌饭", "price": 45.0},
            ],
            "海底捞": [
                {"product": "牛肉卷", "price": 68.0},
                {"product": "海鲜拼盘", "price": 88.0},
            ],
            "喜茶": [
                {"product": "芝士奶茶", "price": 25.0},
                {"product": "四季春奶茶", "price": 22.0},
            ],
            "盒马生鲜": [
                {"product": "有机蔬菜包", "price": 15.0},
                {"product": "生鲜大礼包", "price": 99.0},
            ],
            "九田家烤肉": [
                {"product": "韩式烤牛肉", "price": 128.0},
                {"product": "烤五花肉", "price": 78.0},
            ],
        }
        products = menus.get(merchant_name, [])
        return {"status": "ok", "result": {"merchant": merchant_name, "products": products}, "message": f"{merchant_name} 商品列表: {products}"}
    
    def _tool_add_food_delivery_order(self, username: str, merchant_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        user = self.food_platform_state["users"].get(username)
        if not user:
            return {"status": "error", "message": f"用户 {username} 不存在"}
        if username not in self.food_platform_state["logged_in_users"]:
            return {"status": "error", "message": f"用户 {username} 未登录外卖平台"}
        
        # 计算总价
        menus = {
            "达美乐": {"玛格丽特披萨": 68.0, "超级至尊披萨": 88.0},
            "米村拌饭": {"石锅拌饭": 35.0, "韩式牛肉拌饭": 45.0},
            "海底捞": {"牛肉卷": 68.0, "海鲜拼盘": 88.0},
            "喜茶": {"芝士奶茶": 25.0, "四季春奶茶": 22.0},
            "盒马生鲜": {"有机蔬菜包": 15.0, "生鲜大礼包": 99.0},
            "九田家烤肉": {"韩式烤牛肉": 128.0, "烤五花肉": 78.0},
        }
        merchant_menu = menus.get(merchant_name, {})
        total = 0.0
        order_items = []
        for it in items:
            pname = it.get("product", "")
            qty = it.get("quantity", 1)
            price = merchant_menu.get(pname, 0.0)
            total += price * qty
            order_items.append({"product": pname, "quantity": qty, "price_per_unit": price})
        
        if user["balance"] < total:
            return {"status": "error", "message": f"余额不足: {user['balance']} < {total}"}
        
        user["balance"] -= total
        order = {
            "user_name": username,
            "merchant_name": merchant_name,
            "items": order_items,
            "total_price": total,
        }
        self.food_platform_state["orders"].append(order)
        return {"status": "ok", "result": order, "message": f"订单创建成功，总价 {total} 元"}
    
    # ──────────────────────────────────────────
    # MessageApi 工具实现
    # ──────────────────────────────────────────
    
    def _tool_send_message(self, sender_name: str, receiver_name: str, message: str) -> Dict[str, Any]:
        # 获取 sender_id / receiver_id（简化映射）
        name_to_id = {"Eve": "USR100", "Frank": "USR101", "Grace": "USR102", "Helen": "USR103", "Isaac": "USR104", "Jack": "USR105"}
        sid = name_to_id.get(sender_name, sender_name)
        rid = name_to_id.get(receiver_name, receiver_name)
        
        msg_id = self.message_state["next_message_id"]
        self.message_state["next_message_id"] += 1
        
        msg = {
            "sender_id": sid,
            "receiver_id": rid,
            "message": message,
            "time": "2024-07-15",  # 简化
        }
        self.message_state["inbox"][str(msg_id)] = msg
        return {"status": "ok", "result": {"message_id": msg_id}, "message": f"消息发送成功，ID={msg_id}"}
    
    def _tool_view_messages_between_users(self, sender_name: str, receiver_name: str) -> Dict[str, Any]:
        name_to_id = {"Eve": "USR100", "Frank": "USR101", "Grace": "USR102", "Helen": "USR103", "Isaac": "USR104", "Jack": "USR105"}
        sid = name_to_id.get(sender_name, sender_name)
        rid = name_to_id.get(receiver_name, receiver_name)
        
        msgs = []
        for mid, m in self.message_state["inbox"].items():
            if m["sender_id"] == sid and m["receiver_id"] == rid:
                msgs.append({"message_id": int(mid), **m})
        return {"status": "ok", "result": {"messages": msgs}, "message": f"找到 {len(msgs)} 条消息"}
    
    def _tool_delete_message(self, message_id: int) -> Dict[str, Any]:
        mid = str(message_id)
        if mid not in self.message_state["inbox"]:
            return {"status": "error", "message": f"消息 {message_id} 不存在"}
        del self.message_state["inbox"][mid]
        return {"status": "ok", "result": {"deleted": message_id}, "message": f"消息 {message_id} 已删除"}
    
    def _tool_search_messages(self, user_name: str, keyword: str) -> Dict[str, Any]:
        name_to_id = {"Eve": "USR100", "Frank": "USR101", "Grace": "USR102", "Helen": "USR103", "Isaac": "USR104", "Jack": "USR105"}
        uid = name_to_id.get(user_name, user_name)
        
        msgs = []
        for mid, m in self.message_state["inbox"].items():
            if m["sender_id"] == uid or m["receiver_id"] == uid:
                if keyword in m.get("message", ""):
                    msgs.append({"message_id": int(mid), **m})
        return {"status": "ok", "result": {"messages": msgs}, "message": f"找到 {len(msgs)} 条含 '{keyword}' 的消息"}
    
    def _tool_get_latest_message_id(self) -> Dict[str, Any]:
        if not self.message_state["inbox"]:
            return {"status": "ok", "result": {"message_id": None}, "message": "没有消息"}
        max_id = max(int(k) for k in self.message_state["inbox"].keys())
        return {"status": "ok", "result": {"message_id": max_id}, "message": f"最新消息 ID={max_id}"}
    
    def _tool_get_earliest_message_id(self) -> Dict[str, Any]:
        if not self.message_state["inbox"]:
            return {"status": "ok", "result": {"message_id": None}, "message": "没有消息"}
        min_id = min(int(k) for k in self.message_state["inbox"].keys())
        return {"status": "ok", "result": {"message_id": min_id}, "message": f"最早消息 ID={min_id}"}
    
    # ──────────────────────────────────────────
    # ReminderApi 工具实现
    # ──────────────────────────────────────────
    
    def _tool_add_reminder(self, title: str, description: str, time: str) -> Dict[str, Any]:
        rid = self.reminder_state["next_reminder_id"]
        self.reminder_state["next_reminder_id"] += 1
        reminder = {
            "reminder_id": rid,
            "title": title,
            "description": description,
            "time": time,
            "notified": False,
        }
        self.reminder_state["reminder_list"][str(rid)] = reminder
        return {"status": "ok", "result": {"reminder_id": rid}, "message": f"提醒添加成功，ID={rid}"}
    
    def _tool_view_reminder_by_title(self, title: str) -> Dict[str, Any]:
        for rid, r in self.reminder_state["reminder_list"].items():
            if r["title"] == title:
                return {"status": "ok", "result": {"reminder": r}, "message": f"找到提醒: {r}"}
        return {"status": "error", "message": f"未找到标题为 '{title}' 的提醒"}
    
    def _tool_search_reminders(self, keyword: str) -> Dict[str, Any]:
        results = []
        for rid, r in self.reminder_state["reminder_list"].items():
            if keyword in r.get("title", "") or keyword in r.get("description", ""):
                results.append(r)
        return {"status": "ok", "result": {"reminders": results}, "message": f"找到 {len(results)} 条提醒"}
    
    # ──────────────────────────────────────────
    # 状态快照与评估
    # ──────────────────────────────────────────
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取当前完整状态快照"""
        return {
            "BaseApi": copy.deepcopy(self.base_state),
            "FoodPlatform": copy.deepcopy(self.food_platform_state),
            "MessageApi": copy.deepcopy(self.message_state),
            "ReminderAPI": copy.deepcopy(self.reminder_state),
        }
    
    def compare_with_ground_truth(self, ground_truth: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        将当前状态与 ground_truth 对比，返回相似度分数和差异详情。
        
        Returns:
            (score: 0.0-1.0, details: dict)
        """
        if not ground_truth:
            return 0.0, {"error": "无 ground_truth"}
        
        gt_final = ground_truth[-1] if isinstance(ground_truth, list) else ground_truth
        current = self.get_state_snapshot()
        
        total_checks = 0
        passed_checks = 0
        details = {}
        
        for api_name, gt_state in gt_final.items():
            if api_name not in current:
                details[api_name] = {"status": "missing_in_current", "score": 0.0}
                continue
            
            cur_state = current[api_name]
            api_score, api_detail = self._compare_state(api_name, gt_state, cur_state)
            passed_checks += api_score
            total_checks += 1
            details[api_name] = api_detail
        
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        return score, details
    
    def _compare_state(self, api_name: str, gt: Dict[str, Any], current: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """对比单个 API 模块的状态"""
        # 简化对比：用 json 序列化后比较结构相似度
        try:
            gt_json = json.dumps(gt, sort_keys=True, ensure_ascii=False)
            cur_json = json.dumps(current, sort_keys=True, ensure_ascii=False)
            
            # 对 FoodPlatform 做关键字段对比
            if api_name == "FoodPlatform":
                return self._compare_food_platform(gt, current)
            elif api_name == "MessageApi":
                return self._compare_message_api(gt, current)
            elif api_name == "ReminderAPI":
                return self._compare_reminder_api(gt, current)
            else:
                # BaseApi 直接比较
                match = gt == current
                return (1.0 if match else 0.0, {"match": match})
        except Exception as e:
            return 0.0, {"error": str(e)}
    
    def _compare_food_platform(self, gt: Dict[str, Any], current: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """FoodPlatform 关键字段: orders, logged_in_users"""
        score = 0.0
        checks = 0
        detail = {}
        
        # 检查 logged_in_users
        gt_logged = set(gt.get("logged_in_users", []))
        cur_logged = set(current.get("logged_in_users", []))
        if gt_logged == cur_logged:
            score += 1.0
            detail["logged_in_users"] = {"match": True}
        else:
            detail["logged_in_users"] = {"match": False, "gt": list(gt_logged), "cur": list(cur_logged)}
        checks += 1
        
        # 检查 orders（数量和内容）
        gt_orders = gt.get("orders", [])
        cur_orders = current.get("orders", [])
        
        if len(gt_orders) == len(cur_orders):
            score += 0.5
            detail["order_count"] = {"match": True, "count": len(gt_orders)}
        else:
            detail["order_count"] = {"match": False, "gt_count": len(gt_orders), "cur_count": len(cur_orders)}
        checks += 1
        
        # 检查 order 内容（简化：比较用户名、商家、商品名列表）
        order_match_count = 0
        for gto in gt_orders:
            for co in cur_orders:
                if (gto.get("user_name") == co.get("user_name") and 
                    gto.get("merchant_name") == co.get("merchant_name")):
                    gt_items = {i.get("product"): i.get("quantity") for i in gto.get("items", [])}
                    cur_items = {i.get("product"): i.get("quantity") for i in co.get("items", [])}
                    if gt_items == cur_items:
                        order_match_count += 1
                        break
        
        if len(gt_orders) > 0:
            order_ratio = order_match_count / len(gt_orders)
            score += order_ratio * 0.5
            detail["order_content"] = {"match_count": order_match_count, "gt_count": len(gt_orders)}
            checks += 1
        
        final_score = score / checks if checks > 0 else 0.0
        return final_score, detail
    
    def _compare_message_api(self, gt: Dict[str, Any], current: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """MessageApi 关键字段: inbox 内容"""
        gt_inbox = gt.get("inbox", {})
        cur_inbox = current.get("inbox", {})
        
        if not gt_inbox and not cur_inbox:
            return 1.0, {"empty": True}
        
        # 对比消息数量
        if len(gt_inbox) == len(cur_inbox):
            count_score = 0.5
        else:
            count_score = 0.0
        
        # 对比消息内容（简化：只对比最后一条的 message 内容）
        content_matches = 0
        for mid, gtm in gt_inbox.items():
            if mid in cur_inbox:
                if gtm.get("message") == cur_inbox[mid].get("message"):
                    content_matches += 1
        
        gt_len = len(gt_inbox)
        content_score = (content_matches / gt_len * 0.5) if gt_len > 0 else 0.0
        
        total = count_score + content_score
        return total, {
            "count_match": len(gt_inbox) == len(cur_inbox),
            "content_matches": content_matches,
            "gt_count": len(gt_inbox),
            "cur_count": len(cur_inbox),
        }
    
    def _compare_reminder_api(self, gt: Dict[str, Any], current: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """ReminderApi 关键字段: reminder_list"""
        gt_list = gt.get("reminder_list", {})
        cur_list = current.get("reminder_list", {})
        
        if not gt_list and not cur_list:
            return 1.0, {"empty": True}
        
        # 对比数量
        if len(gt_list) == len(cur_list):
            count_score = 0.3
        else:
            count_score = 0.0
        
        # 对比标题和内容
        match_count = 0
        for rid, gtr in gt_list.items():
            if rid in cur_list:
                cur = cur_list[rid]
                if (gtr.get("title") == cur.get("title") and 
                    gtr.get("description") == cur.get("description")):
                    match_count += 1
        
        gt_len = len(gt_list)
        content_score = (match_count / gt_len * 0.7) if gt_len > 0 else 0.0
        
        total = count_score + content_score
        return total, {
            "count_match": len(gt_list) == len(cur_list),
            "content_matches": match_count,
            "gt_count": len(gt_list),
            "cur_count": len(cur_list),
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "task_id": self.task_id,
            "total_steps": len(self.execution_log),
            "final_state": self.get_state_snapshot(),
            "execution_log": self.execution_log,
        }


def load_acebench_data(path: str) -> List[Dict[str, Any]]:
    """加载 ACEBench JSON 数据"""
    records = []
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]
    return records


if __name__ == "__main__":
    # 简单测试
    task = {
        "id": "test_task",
        "question": "我是Eve，需要在达美乐点超级至尊披萨",
        "initial_config": {"BaseApi": {"wifi": False, "logged_in": True}},
        "function": [{"name": "turn_on_wifi"}, {"name": "login_food_platform"}, {"name": "add_food_delivery_order"}],
    }
    
    sandbox = ACEBenchSandbox(task)
    print("初始状态:", sandbox.get_state_snapshot())
    
    # 模拟 Agent 执行
    print("\n→ turn_on_wifi()")
    print(sandbox.execute("turn_on_wifi", {}))
    
    print("\n→ login_food_platform(username='Eve', password='password123')")
    print(sandbox.execute("login_food_platform", {"username": "Eve", "password": "password123"}))
    
    print("\n→ add_food_delivery_order(...)")
    print(sandbox.execute("add_food_delivery_order", {
        "username": "Eve",
        "merchant_name": "达美乐",
        "items": [{"product": "超级至尊披萨", "quantity": 1}]
    }))
    
    print("\n最终状态:", json.dumps(sandbox.get_state_snapshot(), ensure_ascii=False, indent=2))
