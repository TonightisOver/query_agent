"""
数据时效性审计 Agent (Phase 3)
评估各厂商数据的新鲜度，对过期数据进行降权处理和告警。
"""
from typing import Dict, List, Any
from datetime import datetime, timedelta
from agents.engine import call_llm

# 时效性评估规则
FRESHNESS_RULES = [
    {"max_days": 7, "score": 1.0, "label": "新鲜", "color": "green"},
    {"max_days": 30, "score": 0.9, "label": "较新", "color": "blue"},
    {"max_days": 90, "score": 0.7, "label": "需关注", "color": "orange"},
    {"max_days": float('inf'), "score": 0.5, "label": "陈旧", "color": "red"}
]


def _evaluate_freshness(last_updated) -> Dict:
    """评估单条数据的时效性"""
    now = datetime.utcnow()
    
    # 解析 last_updated
    if isinstance(last_updated, str):
        try:
            update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00').replace('+00:00', ''))
        except:
            update_time = now  # 无法解析时视为最新
    elif isinstance(last_updated, datetime):
        update_time = last_updated
    else:
        update_time = now
    
    days_old = max(0, (now - update_time).days)
    
    # 匹配规则
    for rule in FRESHNESS_RULES:
        if days_old <= rule["max_days"]:
            warning = None
            if rule["score"] < 0.9:
                warning = f"数据已超过 {days_old} 天未更新，建议重新采集"
            return {
                "freshness_score": rule["score"],
                "freshness_label": rule["label"],
                "freshness_color": rule["color"],
                "last_updated": update_time.strftime("%Y-%m-%d"),
                "days_old": days_old,
                "warning": warning
            }
    
    # 默认
    return {
        "freshness_score": 0.5,
        "freshness_label": "陈旧",
        "freshness_color": "red",
        "last_updated": update_time.strftime("%Y-%m-%d"),
        "days_old": days_old,
        "warning": f"数据已超过 {days_old} 天未更新"
    }


def freshness_auditor_node(state: dict) -> dict:
    """时效性审计节点 — 评估所有厂商数据的新鲜度"""
    reports_archive = state.get("reports_archive", {})
    trace_logs = list(state.get("trace_logs", []))
    
    if not reports_archive:
        trace_msg = "[FreshnessAuditor] reports_archive 为空，跳过时效性审计。"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"freshness_results": {}, "trace_logs": trace_logs}
    
    trace_msg = f"[FreshnessAuditor] 启动数据时效性审计，厂商数: {len(reports_archive)}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    freshness_results = {}
    warnings_count = 0
    
    for vendor_name, vendor_data in reports_archive.items():
        last_updated = vendor_data.get("last_updated", datetime.utcnow().isoformat())
        result = _evaluate_freshness(last_updated)
        freshness_results[vendor_name] = result
        
        if result["warning"]:
            warnings_count += 1
            trace_msg = f"[FreshnessAuditor] ⚠️ {vendor_name}: {result['warning']}"
        else:
            trace_msg = f"[FreshnessAuditor] ✅ {vendor_name}: {result['freshness_label']} (更新于 {result['last_updated']})"
        print(trace_msg)
        trace_logs.append(trace_msg)
    
    # LLM 语义过时检测：对 days_old > 30 的厂商调用 LLM 判断是否语义过时
    for vendor_name, result in freshness_results.items():
        if result["days_old"] > 30:
            try:
                vendor_data = reports_archive.get(vendor_name, {})
                model_family = vendor_data.get("model_family", "未知")
                pricing = vendor_data.get("pricing", {})
                features = vendor_data.get("features", {})
                features_summary = ", ".join(list(features.keys())[:5]) if isinstance(features, dict) else str(features)

                semantic_system = "你是一个数据时效性分析专家。判断以下大模型厂商数据是否可能已经语义过时（即使时间戳较新，但市场可能已发生重大变化）。回答'过时'或'有效'，并给出一句话理由。"
                semantic_user = f"厂商: {vendor_name}\n模型系列: {model_family}\n定价: {pricing}\n特性摘要: {features_summary}"

                llm_response = call_llm(semantic_user, system_prompt=semantic_system)
                trace_msg = f"[FreshnessAuditor] LLM语义评估 {vendor_name}: {llm_response[:80]}"
                print(trace_msg)
                trace_logs.append(trace_msg)

                if "过时" in llm_response:
                    result["freshness_score"] = max(0.0, result["freshness_score"] - 0.2)
                    semantic_warning = f"LLM语义判定数据可能过时: {llm_response[:100]}"
                    if result["warning"]:
                        result["warning"] += f"; {semantic_warning}"
                    else:
                        result["warning"] = semantic_warning
                    warnings_count += 1
                    trace_msg = f"[FreshnessAuditor] ⚠️ {vendor_name} 语义过时，freshness_score 降权至 {result['freshness_score']:.2f}"
                    print(trace_msg)
                    trace_logs.append(trace_msg)
            except Exception as e:
                trace_msg = f"[FreshnessAuditor] LLM语义评估 {vendor_name} 异常: {e}"
                print(trace_msg)
                trace_logs.append(trace_msg)

    summary = f"[FreshnessAuditor] 审计完成: {len(freshness_results)} 家厂商, {warnings_count} 条时效性告警"
    print(summary)
    trace_logs.append(summary)
    
    return {"freshness_results": freshness_results, "trace_logs": trace_logs}
