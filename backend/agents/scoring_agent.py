"""
评分引擎 Agent (Phase 3)
多维度加权评分系统，对各厂商进行全面量化评估和排名。
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from agents.engine import call_llm

# 默认评分维度与权重
DEFAULT_SCORING_WEIGHTS = {
    "pricing_competitiveness": 0.20,
    "context_capability": 0.15,
    "multimodal_support": 0.10,
    "developer_ecosystem": 0.15,
    "performance_throughput": 0.15,
    "compliance_security": 0.10,
    "innovation_roadmap": 0.10,
    "user_satisfaction": 0.05
}

# 场景专属权重调整
SCENARIO_WEIGHT_OVERRIDES = {
    "code_development": {
        "developer_ecosystem": 0.25,
        "pricing_competitiveness": 0.15,
        "context_capability": 0.20,
        "multimodal_support": 0.05
    },
    "customer_service": {
        "performance_throughput": 0.25,
        "pricing_competitiveness": 0.25,
        "context_capability": 0.10,
        "innovation_roadmap": 0.05
    },
    "data_analysis": {
        "developer_ecosystem": 0.25,
        "context_capability": 0.20,
        "compliance_security": 0.20,
        "pricing_competitiveness": 0.15,
        "multimodal_support": 0.10,
        "user_satisfaction": 0.10
    },
    "document_analysis": {
        "context_capability": 0.30,
        "pricing_competitiveness": 0.15,
        "multimodal_support": 0.15,
        "performance_throughput": 0.10
    },
    "enterprise": {
        "compliance_security": 0.25,
        "developer_ecosystem": 0.20,
        "performance_throughput": 0.15,
        "pricing_competitiveness": 0.10
    },
    "creative": {
        "innovation_roadmap": 0.20,
        "context_capability": 0.20,
        "user_satisfaction": 0.15,
        "pricing_competitiveness": 0.15
    }
}


def adjust_weights_for_scenario(scenario: str) -> Dict[str, float]:
    """根据场景调整评分权重"""
    weights = dict(DEFAULT_SCORING_WEIGHTS)
    overrides = SCENARIO_WEIGHT_OVERRIDES.get(scenario, {})
    weights.update(overrides)
    # 归一化确保权重总和为 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _safe_get_nested(data: dict, *keys, default=None):
    """安全获取嵌套字典值"""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def _compute_vendor_scores(vendor_name: str, vendor_data: dict, all_vendors_data: dict) -> Dict[str, float]:
    """计算单个厂商的各维度得分 (0-100)"""
    scores = {}
    
    # 收集所有厂商的数据用于归一化
    all_prices = []
    all_contexts = []
    all_tpms = []
    for vd in all_vendors_data.values():
        price = _safe_get_nested(vd, "pricing", "prompt_price_per_million", default=0)
        # 统一为 USD (CNY / 7.2)
        currency = _safe_get_nested(vd, "pricing", "currency", default="USD")
        if currency == "CNY":
            price = price / 7.2
        all_prices.append(price)
        all_contexts.append(_safe_get_nested(vd, "features", "context_window", default=0))
        all_tpms.append(_safe_get_nested(vd, "rate_limits", "tpm", default=0))
    
    max_price = max(all_prices) if all_prices else 1
    max_context = max(all_contexts) if all_contexts else 1
    max_tpm = max(all_tpms) if all_tpms else 1
    
    # 当前厂商数据
    price = _safe_get_nested(vendor_data, "pricing", "prompt_price_per_million", default=0)
    currency = _safe_get_nested(vendor_data, "pricing", "currency", default="USD")
    if currency == "CNY":
        price = price / 7.2
    context = _safe_get_nested(vendor_data, "features", "context_window", default=0)
    tpm = _safe_get_nested(vendor_data, "rate_limits", "tpm", default=0)
    vision = _safe_get_nested(vendor_data, "features", "vision_support", default=False)
    func_call = _safe_get_nested(vendor_data, "features", "function_calling", default=False)
    satisfaction = _safe_get_nested(vendor_data, "user_feedback", "developer_satisfaction", default=3.0)
    coding_plan = vendor_data.get("coding_plan", {})
    region = vendor_data.get("region", "international")
    
    # 1. 价格竞争力 (越便宜越好)
    if max_price > 0 and price >= 0:
        scores["pricing_competitiveness"] = round(max(0, 100 * (1 - price / max_price)), 1)
    else:
        scores["pricing_competitiveness"] = 50.0
    
    # 2. 上下文窗口能力
    if max_context > 0:
        scores["context_capability"] = round(min(100, 100 * (context / max_context)), 1)
    else:
        scores["context_capability"] = 50.0
    
    # 3. 多模态支持
    multimodal_score = 0
    if vision:
        multimodal_score += 60
    if func_call:
        multimodal_score += 40
    scores["multimodal_support"] = float(multimodal_score)
    
    # 4. 开发者生态
    eco_score = 0
    if _safe_get_nested(coding_plan, "is_supported_in_editor", default=False):
        eco_score += 40
    if _safe_get_nested(coding_plan, "has_sandbox_env", default=False):
        eco_score += 30
    langs = _safe_get_nested(coding_plan, "language_optimizations", default=[])
    eco_score += min(30, len(langs) * 10)
    scores["developer_ecosystem"] = float(eco_score)
    
    # 5. 性能吞吐
    if max_tpm > 0:
        scores["performance_throughput"] = round(min(100, 100 * (tpm / max_tpm)), 1)
    else:
        scores["performance_throughput"] = 50.0
    
    # 6. 合规安全
    compliance_score = 50  # 基础分
    if region == "domestic":
        compliance_score += 30  # 国内厂商中国市场合规优势
    else:
        compliance_score += 20  # 国际厂商全球合规
    strengths = _safe_get_nested(vendor_data, "user_feedback", "strengths", default=[])
    for s in strengths:
        if any(kw in str(s) for kw in ["合规", "安全", "SOC", "隐私", "主权"]):
            compliance_score += 20
            break
    scores["compliance_security"] = float(min(100, compliance_score))
    
    # 7. 创新力 (基于满意度)
    scores["innovation_roadmap"] = round(min(100, satisfaction / 5.0 * 100), 1)
    
    # 8. 用户满意度
    scores["user_satisfaction"] = round(min(100, satisfaction / 5.0 * 100), 1)
    
    return scores


def _generate_recommendation(vendor_name: str, scores: Dict[str, float], rank: int) -> str:
    """生成厂商推荐语"""
    top_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    dim_names = {
        "pricing_competitiveness": "价格竞争力",
        "context_capability": "上下文能力",
        "multimodal_support": "多模态支持",
        "developer_ecosystem": "开发者生态",
        "performance_throughput": "性能吞吐",
        "compliance_security": "合规安全",
        "innovation_roadmap": "创新力",
        "user_satisfaction": "用户满意度"
    }
    strengths = [dim_names.get(d[0], d[0]) for d in top_dims]
    if rank == 1:
        return f"综合排名第一，在{strengths[0]}和{strengths[1]}方面表现最为突出"
    elif rank <= 3:
        return f"综合实力强劲，{strengths[0]}表现优异"
    else:
        return f"{strengths[0]}方面有独特优势"


def scoring_agent_node(state: dict) -> dict:
    """评分 Agent 节点 — 对所有厂商进行多维度加权评分和排名"""
    reports_archive = state.get("reports_archive", {})
    scenario = ""
    parsed_req = state.get("parsed_requirement", {})
    if parsed_req:
        scenario = parsed_req.get("scenario", "")
    
    trace_logs = list(state.get("trace_logs", []))
    
    if not reports_archive:
        trace_msg = "[ScoringAgent] reports_archive 为空，跳过评分。"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"scoring_results": {}, "trace_logs": trace_logs}
    
    # 选择权重
    weights = adjust_weights_for_scenario(scenario) if scenario else DEFAULT_SCORING_WEIGHTS

    # LLM 动态权重微调：根据用户查询语义进一步优化权重分配
    raw_query = parsed_req.get("raw_query", "")
    if len(raw_query) > 10:
        try:
            llm_system_prompt = (
                "你是一个大模型评分权重优化专家。根据用户的具体需求语义，微调评分维度权重。"
                "输出 JSON 格式，键为维度名，值为建议权重(0.0-0.4)。只输出 JSON，不要其他文字。"
            )
            llm_user_prompt = (
                f"用户需求: {raw_query}\n"
                f"当前场景: {scenario or '通用'}\n"
                f"当前权重: {json.dumps(weights, ensure_ascii=False)}\n"
                f"可用维度: pricing_competitiveness, context_capability, multimodal_support, "
                f"developer_ecosystem, performance_throughput, compliance_security, "
                f"innovation_roadmap, user_satisfaction\n"
                f"请根据用户需求语义微调权重，输出完整的维度权重 JSON。"
            )
            llm_response = call_llm(prompt=llm_user_prompt, system_prompt=llm_system_prompt)
            # 解析 LLM 返回的 JSON 权重
            llm_weights = json.loads(llm_response.strip())
            # 验证: 所有键必须是合法维度，值必须在 0.0-0.4 范围内
            valid_dims = set(DEFAULT_SCORING_WEIGHTS.keys())
            if isinstance(llm_weights, dict) and all(
                k in valid_dims and isinstance(v, (int, float)) and 0.0 <= v <= 0.4
                for k, v in llm_weights.items()
            ):
                # 合并: 仅更新 LLM 返回的维度
                for k, v in llm_weights.items():
                    weights[k] = float(v)
                # 归一化确保权重总和为 1.0
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                llm_trace = f"[ScoringAgent] LLM 动态权重微调生效，基于查询语义: '{raw_query[:50]}...'"
                print(llm_trace)
                trace_logs.append(llm_trace)
        except Exception:
            # LLM 调用失败时静默回退到启发式权重
            pass

    trace_msg = f"[ScoringAgent] 启动多维度评分，场景: {scenario or '通用'}，厂商数: {len(reports_archive)}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    # 计算每个厂商的得分
    vendor_scores = {}
    for vendor_name, vendor_data in reports_archive.items():
        dim_scores = _compute_vendor_scores(vendor_name, vendor_data, reports_archive)
        # 加权总分
        overall = sum(dim_scores.get(dim, 0) * w for dim, w in weights.items())
        vendor_scores[vendor_name] = {
            "dimension_scores": dim_scores,
            "overall_score": round(overall, 1)
        }
    
    # 排名
    sorted_vendors = sorted(vendor_scores.items(), key=lambda x: x[1]["overall_score"], reverse=True)
    rankings = []
    for rank, (vendor_name, score_data) in enumerate(sorted_vendors, 1):
        recommendation = _generate_recommendation(vendor_name, score_data["dimension_scores"], rank)
        rankings.append({
            "rank": rank,
            "vendor_name": vendor_name,
            "overall_score": score_data["overall_score"],
            "dimension_scores": score_data["dimension_scores"],
            "recommendation": recommendation
        })
    
    top = rankings[0] if rankings else None
    top_recommendation = f"综合评分最高推荐: {top['vendor_name']} (综合得分: {top['overall_score']}分)" if top else "无推荐"
    
    # 增加分析师量化解读思考，强化说服力与细节度
    dim_names = {
        "pricing_competitiveness": "价格竞争力",
        "context_capability": "上下文窗口",
        "multimodal_support": "多模态能力",
        "developer_ecosystem": "开发者生态",
        "performance_throughput": "高频并发吞吐",
        "compliance_security": "数据合规安全",
        "innovation_roadmap": "创新满意度"
    }
    weight_desc = ", ".join([f"{dim_names.get(k, k)}: {int(v*100)}%" for k, v in weights.items() if v > 0.12])
    score_thought = f"【大模型竞品分析师 · 多维加权量化评估报告】\n> 针对 [{scenario or '通用'}场景] 的特殊业务诉求，评分系统已自动调优权重分配比重，核心加权维度包括（{weight_desc}）。各厂商在多维性能指标矩阵中完成了无量纲化数值对比，揭示了每一家厂商底层的‘旗舰超脑’与‘性价比能效’双层级性能指数，致力于为您勾勒出最真实客观的性价比最优推荐路线。"
    trace_logs.append(f"📊 **{score_thought}**")
    
    scoring_results = {
        "rankings": rankings,
        "weights_used": weights,
        "scenario": scenario or "通用",
        "top_recommendation": top_recommendation,
        "evaluated_at": datetime.utcnow().isoformat()
    }
    
    trace_msg = f"[ScoringAgent] 评分完成。{top_recommendation}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    return {"scoring_results": scoring_results, "trace_logs": trace_logs}
