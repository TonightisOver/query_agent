"""
流程编排 Agent (Phase 4)
根据需求复杂度动态决定执行策略，智能跳过非必要步骤。
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.engine import call_llm


def planner_agent_node(state: dict) -> dict:
    """编排 Agent 节点 — 根据需求复杂度制定执行计划"""
    parsed_req = state.get("parsed_requirement", {})
    competitor_list = state.get("competitor_list", [])
    trace_logs = list(state.get("trace_logs", []))
    
    raw_query = parsed_req.get("raw_query", "") if parsed_req else ""
    scenario = parsed_req.get("scenario", "general") if parsed_req else "general"
    
    trace_msg = f"[PlannerAgent] 分析执行策略，查询长度: {len(raw_query)}，场景: {scenario}，厂商数: {len(competitor_list)}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    # 尝试查找知识库缓存
    cache_hit = False
    try:
        from agents.learning_agent import lookup_cached_result
        cached = lookup_cached_result(competitor_list, scenario, max_age_hours=24)
        if cached:
            cache_hit = True
    except Exception:
        pass
    
    # 决策逻辑
    if cache_hit:
        plan_type = "cache_hit"
        skip_agents = ["collector", "sanitizer", "analyzer", "qc"]
        enable_rag = False
        max_qc_retries = 0
        reasoning = "知识库中发现新鲜的缓存结果（24小时内），直接复用历史分析"
    elif len(raw_query) < 20 and len(competitor_list) <= 3:
        plan_type = "standard"
        skip_agents = []
        enable_rag = False
        max_qc_retries = 2
        reasoning = "标准分析模式：查询简短，厂商数量少"
    elif len(competitor_list) > 8 or len(raw_query) > 100:
        plan_type = "deep_analysis"
        skip_agents = []
        enable_rag = True
        max_qc_retries = 3
        reasoning = "深度分析模式：大规模厂商对比或复杂需求，启用 RAG 增强"
    else:
        plan_type = "standard"
        skip_agents = []
        enable_rag = True  # 如果有 RAG 文档就启用
        max_qc_retries = 2
        reasoning = "标准分析模式，如有 RAG 文档则启用检索增强"

    # LLM-powered complexity assessment: potentially upgrade standard plans
    if plan_type == "standard":
        try:
            complexity_system_prompt = (
                "你是一个查询复杂度评估专家。请分析用户的竞品分析查询，"
                "评估其复杂度（1-10分）。只返回一个JSON对象，格式为: "
                '{\"complexity_score\": <int>, \"reason\": \"<简短原因>\"}'
            )
            complexity_user_prompt = (
                f"请评估以下竞品分析查询的复杂度:\n"
                f"查询内容: {raw_query}\n"
                f"分析场景: {scenario}\n"
                f"涉及厂商数量: {len(competitor_list)}\n"
                f"请返回1-10的复杂度评分。"
            )
            llm_response = call_llm(
                prompt=complexity_user_prompt,
                system_prompt=complexity_system_prompt
            )
            score_data = json.loads(llm_response)
            complexity_score = int(score_data.get("complexity_score", 0))
            if complexity_score >= 7:
                plan_type = "deep_analysis"
                enable_rag = True
                max_qc_retries = 3
                reasoning = f"LLM 复杂度评估升级：评分 {complexity_score}/10，{score_data.get('reason', '语义复杂度高')}"
                upgrade_msg = f"[PlannerAgent] LLM 复杂度评估将计划从 standard 升级为 deep_analysis (score={complexity_score})"
                print(upgrade_msg)
                trace_logs.append(upgrade_msg)
        except Exception:
            pass  # 静默回退到启发式结果

    execution_plan = {
        "plan_type": plan_type,
        "skip_agents": skip_agents,
        "enable_rag": enable_rag,
        "max_qc_retries": max_qc_retries,
        "reasoning": reasoning,
        "planned_at": datetime.utcnow().isoformat()
    }
    
    arch_thought = f"【系统架构师 · 管道执行编排思考】\n> 针对当前竞品分析诉求，系统评估得出的最佳运行模式为 [{plan_type}]。核心考量逻辑：{reasoning}。为了在确保100%可信分析质量的同时最大化系统的执行效能，本管道已动态配置了‘{'启用 RAG 本地知识检索以强化数据可信审计' if enable_rag else '直连大厂数据以获取最高实时吞吐'}’与‘最大质检打回重试数: {max_qc_retries}’的防报错容错参数。"
    trace_logs.append(f"⚙️ **{arch_thought}**")
    
    trace_msg = f"[PlannerAgent] 执行计划: {plan_type} — {reasoning}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    return {"execution_plan": execution_plan, "trace_logs": trace_logs}
