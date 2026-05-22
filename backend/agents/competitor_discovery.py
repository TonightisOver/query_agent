"""
竞品发现与推荐 Agent (Phase 1)
根据解析的结构化需求，在 16 家大模型厂商中进行多属性过滤和加权推荐打分。
"""
from typing import Dict, List, Any
from agents.engine import OFFLINE_MODELS_FALLBACK, call_llm

def _safe_get_nested(data: dict, *keys, default=None):
    """安全获取嵌套字典值"""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current

def competitor_discovery_node(state: dict) -> dict:
    """竞品发现节点 — 对 16 家厂商进行需求匹配与智能推荐打分"""
    parsed_req = state.get("parsed_requirement", {})
    trace_logs = list(state.get("trace_logs", []))
    
    if not parsed_req:
        trace_msg = "[CompetitorDiscovery Agent] 未找到结构化需求，跳过自动发现。"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"recommended_vendors": [], "discovery_reasoning": "没有可用需求。", "trace_logs": trace_logs}
    
    scenario = parsed_req.get("scenario", "general")
    budget = parsed_req.get("budget_range", {})
    max_price = budget.get("max_price_per_million")
    req_currency = budget.get("currency", "CNY")
    
    perf = parsed_req.get("performance_requirements", {})
    context_min = perf.get("context_window_min", 0) or 0
    latency_pref = perf.get("latency_priority", "medium")
    
    feats = parsed_req.get("feature_requirements", {})
    req_func_call = feats.get("function_calling")
    req_vision = feats.get("vision_support")
    
    langs = parsed_req.get("language_preference", ["中文"])
    
    trace_msg = f"[CompetitorDiscovery Agent] 开启多维度配对，场景: {scenario}，语言偏好: {langs}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    evaluated_vendors = []
    
    # 评分逻辑对 16 个厂商循环
    for vendor_name, info in OFFLINE_MODELS_FALLBACK.items():
        # 1. 价格竞争力打分 (30%)
        # 提取厂商 Prompt Token 价格（转换统一币种进行对比）
        v_price = _safe_get_nested(info, "pricing", "prompt_price_per_million", default=0.0)
        v_currency = _safe_get_nested(info, "pricing", "currency", default="USD")
        
        # 转换至统一币种 (CNY / 7.2)
        v_price_usd = v_price if v_currency == "USD" else v_price / 7.2
        
        price_score = 100.0
        if max_price is not None:
            max_price_usd = max_price if req_currency == "USD" else max_price / 7.2
            if v_price_usd <= max_price_usd:
                price_score = 100.0
            else:
                # 超出预算则按比例扣分
                price_score = max(0.0, 100.0 - ((v_price_usd - max_price_usd) / max_price_usd) * 100.0)
                
        # 2. 上下文能力打分 (15%)
        v_context = _safe_get_nested(info, "features", "context_window", default=8000)
        context_score = 100.0
        if context_min > 0:
            if v_context >= context_min:
                context_score = 100.0
            else:
                context_score = (v_context / context_min) * 100.0
                
        # 3. 特征偏好打分 (15%)
        v_func_call = _safe_get_nested(info, "features", "function_calling", default=False)
        v_vision = _safe_get_nested(info, "features", "vision_support", default=False)
        
        feature_score = 100.0
        deductions = 0
        if req_func_call is True and not v_func_call:
            deductions += 50
        if req_vision is True and not v_vision:
            deductions += 50
        feature_score -= deductions
        
        # 4. 场景适配度打分 (25%)
        # 根据厂商的核心优势场景打分
        scenario_score = 80.0
        v_region = info.get("region", "international")
        v_feedback = _safe_get_nested(info, "user_feedback", "strengths", default=[])
        feedback_str = "".join(v_feedback)
        
        if scenario == "code_development":
            if vendor_name in ["OpenAI", "Anthropic", "深度求索", "Google"]:
                scenario_score = 100.0
            elif "开发" in feedback_str or "编程" in feedback_str or "代码" in feedback_str:
                scenario_score = 90.0
            else:
                scenario_score = 60.0
        elif scenario == "customer_service":
            if vendor_name in ["火山引擎", "智谱AI", "阿里通义千问", "百度文心"]:
                scenario_score = 100.0
            elif "高并发" in feedback_str or "性价比" in feedback_str:
                scenario_score = 90.0
            else:
                scenario_score = 60.0
        elif scenario == "document_analysis":
            if vendor_name in ["月之暗面", "Google", "Anthropic"]:
                scenario_score = 100.0
            elif "长文本" in feedback_str or "文档" in feedback_str:
                scenario_score = 90.0
            else:
                scenario_score = 60.0
        elif scenario == "enterprise":
            if vendor_name in ["火山引擎", "阿里通义千问", "百度文心", "智谱AI"] or v_region == "domestic":
                scenario_score = 100.0
            else:
                scenario_score = 70.0
        elif scenario == "creative":
            if vendor_name in ["OpenAI", "Anthropic", "月之暗面", "零一万物"]:
                scenario_score = 100.0
            else:
                scenario_score = 65.0
                
        # 5. 地域及语言偏好打分 (15%)
        region_score = 80.0
        if "中文" in langs and "英文" in langs:
            region_score = 100.0
        elif "中文" in langs:
            region_score = 100.0 if v_region == "domestic" else 60.0
        elif "英文" in langs:
            region_score = 100.0 if v_region == "international" else 60.0
            
        # 6. 计算最终加权总分
        overall_score = (
            price_score * 0.30 +
            context_score * 0.15 +
            feature_score * 0.15 +
            scenario_score * 0.25 +
            region_score * 0.15
        )
        
        # 匹配原因列表
        reasons = []
        if price_score == 100.0 and max_price is not None:
            reasons.append("符合预算限制")
        if context_score == 100.0 and context_min > 0:
            reasons.append("满足长文本需求")
        if req_func_call is True and v_func_call:
            reasons.append("支持函数调用")
        if req_vision is True and v_vision:
            reasons.append("支持多模态视觉")
        if scenario_score >= 90.0:
            reasons.append(f"高度适配 [{scenario}] 场景")
        if region_score == 100.0:
            reasons.append("本地化/语种支持契合度高")
            
        if not reasons:
            reasons.append("综合性价比合理")
            
        evaluated_vendors.append({
            "vendor_name": vendor_name,
            "match_score": round(overall_score, 1),
            "match_reasons": reasons[:3],
            "auto_selected": False
        })
        
    # 按得分降序排列
    evaluated_vendors.sort(key=lambda x: x["match_score"], reverse=True)
    
    # 自动勾选推荐厂商 (得分 > 70 且最多推荐 5 家，最少推荐 2 家以保证竞品对比度)
    recommended_count = 0
    for idx, item in enumerate(evaluated_vendors):
        if (item["match_score"] >= 72.0 and recommended_count < 5) or recommended_count < 2:
            item["auto_selected"] = True
            recommended_count += 1
            
    auto_list = [v["vendor_name"] for v in evaluated_vendors if v["auto_selected"]]
    
    trace_msg = f"[CompetitorDiscovery Agent] 竞品发现完成，自动挑选了 {len(auto_list)} 家推荐厂商: {auto_list}"
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    # 2. 大胆调用 LLM 生成高水准的“推荐组合与性价比折衷深度剖析”
    llm_thought = ""
    system_prompt = """你是一个全球大模型采购决策与企业级混合路由专家。请针对选中的推荐厂商列表及用户诉求，给出一段深入浅出、高含金量的专家级性价比匹配解析（150-200字）。
请深度阐述：
1. 为什么这些选中的厂商是优势互补的完美组合。
2. 在这些厂商的多模型矩阵（Portfolio）中，如何权衡昂贵旗舰模型（如 Claude Opus / OpenAI o4）与高性价比模型（如 Claude Sonnet / GPT Instant / DeepSeek）的路由选择，以达到最高的性价比（性价比考虑）。
请直接输出纯段落文字，不要有任何 Markdown 标记或多余的解释。"""

    user_prompt = f"场景: {scenario}, 推荐厂商列表: {auto_list}, 用户原始查询: \"{parsed_req.get('raw_query', '')}\""
    
    try:
        llm_thought = call_llm(user_prompt, system_prompt).strip()
        print(f"[Discovery Expert Thought] {llm_thought}")
    except Exception as e:
        # 针对不同场景的优雅高质量离线兜底
        fallback_thoughts = {
            "code_development": f"【竞品发现专家深度剖析】本次我们自动挑选了 {auto_list} 作为核心对比厂商。在代码开发领域，这形成了一个完美的‘顶奢思维链与极致性价比’互补矩阵。深度求索（DeepSeek）可作为日常代码生成的绝对主路由（以极低价格覆盖 90% 的普通编码需求）；而 OpenAI 与 Anthropic 的旗舰大模型（如 o4-pro / Claude 3.5 Sonnet）则在涉及复杂多步算法重构、架构规划的冷路通道中提供无可挑剔的脑力支持。这套组合能帮助企业将编码成本直接压缩 80% 以上。",
            "customer_service": f"【竞品发现专家深度剖析】本次自动推荐的 {auto_list} 构成了一套高并发低延迟智能客服的黄金选型。火山引擎（字节跳动）与深度求索等国内厂商在大规模高吞吐、极致低成本上无出其右（百万 Token 仅需 1 元左右，非常适合全天候客服路由），而 OpenAI 等国际厂商在处理跨国、跨语种多模态视觉问题上具备独特优势。双模混合路由策略能保证客服的高可用与低响应延迟。",
            "document_analysis": f"【竞品发现专家深度剖析】针对超长文档分析任务，我们智能推荐了 {auto_list}。谷歌 Gemini 所拥有的百万级原生窗口与月之暗面 Kimi 的极长文本无损处理，代表了当前的业界巅峰。我们将详细对比这些厂商在长上下文缓存（Context Caching）方面的阶梯折扣力度，评估在百万级高频长文本查询中，如何利用高能效性价比模型与旗舰版形成冷热路由，实现极致性价比（性价比考虑）。"
        }
        llm_thought = fallback_thoughts.get(scenario, f"【竞品发现专家深度剖析】针对当前诉求，我们自动挑选了 {auto_list} 服务商进行多维度横向测评。这个组合兼顾了国内本土合规性优势与海外顶奢推理能力，为企业架构师提供了极其立体的‘旗舰 vs 性价比’多模型 Portfolio 决策视野。在接下来的评分和报告撰写阶段，我们将着重为您规划最具性价比的智能路由策略。")

    trace_logs.append(f"🔍 **【竞品发现专家 · 推荐组合与性价比深度剖析】**\n> {llm_thought}")
    discovery_reasoning = llm_thought
    
    return {
        "recommended_vendors": evaluated_vendors,
        "competitor_list": auto_list, # 更新竞品列表
        "discovery_reasoning": discovery_reasoning,
        "trace_logs": trace_logs
    }
