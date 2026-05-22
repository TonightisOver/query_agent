"""
用户反馈处理 Agent (Phase 4)
收集、存储和分析用户对分析报告的反馈，驱动持续优化。
"""
import os
import json
import threading
import uuid
from typing import Dict, List, Optional
from datetime import datetime

# 持久化路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEEDBACK_STORE_PATH = os.path.join(DATA_DIR, "feedback_store.json")
_feedback_lock = threading.Lock()


def _ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_feedback_store() -> List[Dict]:
    """加载反馈存储"""
    _ensure_data_dir()
    if os.path.exists(FEEDBACK_STORE_PATH):
        try:
            with open(FEEDBACK_STORE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_feedback_store(data: List[Dict]):
    """保存反馈存储"""
    _ensure_data_dir()
    with open(FEEDBACK_STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_feedback(report_id: str, rating: int, comments: str, vendor_list: List[str] = None) -> Dict:
    """保存用户反馈"""
    with _feedback_lock:
        store = _load_feedback_store()
        feedback_entry = {
            "feedback_id": str(uuid.uuid4())[:8],
            "report_id": report_id,
            "rating": max(1, min(5, rating)),
            "comments": comments,
            "vendor_list": vendor_list or [],
            "created_at": datetime.utcnow().isoformat()
        }
        store.append(feedback_entry)
        _save_feedback_store(store)
        print(f"[FeedbackAgent] 保存反馈: report={report_id}, rating={rating}")
        return feedback_entry


def get_feedback_summary() -> Dict:
    """获取反馈统计摘要"""
    store = _load_feedback_store()
    if not store:
        return {"total_feedback": 0, "average_rating": 0, "rating_distribution": {}}
    
    ratings = [f["rating"] for f in store]
    distribution = {}
    for r in range(1, 6):
        distribution[str(r)] = ratings.count(r)
    
    return {
        "total_feedback": len(store),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "rating_distribution": distribution,
        "recent_comments": [f.get("comments", "") for f in store[-5:] if f.get("comments")]
    }


def get_average_satisfaction() -> float:
    """获取平均满意度评分"""
    summary = get_feedback_summary()
    return summary.get("average_rating", 0)


def feedback_agent_node(state: dict) -> dict:
    """反馈 Agent 节点 — 注入历史反馈统计到报告上下文"""
    trace_logs = list(state.get("trace_logs", []))
    
    try:
        summary = get_feedback_summary()
        trace_msg = f"[FeedbackAgent] 历史反馈统计: {summary['total_feedback']} 条, 平均评分 {summary['average_rating']}"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"feedback_summary": summary, "trace_logs": trace_logs}
    except Exception as e:
        trace_msg = f"[FeedbackAgent] 加载反馈数据异常: {e}"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"feedback_summary": {}, "trace_logs": trace_logs}
