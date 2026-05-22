"""
需求解析 Agent (Phase 1)
解析用户自然语言输入，提取结构化的竞品分析过滤条件。
支持基于大模型的智能解析与离线规则降级引擎。
"""
import os
import re
import json
from typing import Dict, List, Any
from agents.engine import call_llm

def parse_requirement_offline(query: str) -> dict:
    """基于关键词和正则的离线需求解析引擎"""
    query_lower = query.lower()
    
    # 1. 场景匹配
    scenario = "general"
    if any(k in query_lower for k in ["代码", "编程", "开发", "code", "coding", "programming", "agent开发助手", "修改文件", "修复报错", "运行测试"]):
        scenario = "code_development"
    elif any(k in query_lower for k in ["客服", "对话", "聊天", "chat", "customer", "咨询", "售后", "退款", "客服系统"]):
        scenario = "customer_service"
    elif any(k in query_lower for k in ["数据分析助手", "数据分析", "excel", "csv", "database", "数据库数据", "分析趋势", "生成图表", "解释异常", "输出报告"]):
        scenario = "data_analysis"
    elif any(k in query_lower for k in [
        "文档", "分析", "报告", "document", "analysis", 
        "论文", "学术", "文献", "大纲", "润色", "引用", "改写", "翻译", "段落", "写作助手", 
        "合同审查", "风险条款", "法律", "简历", "jd", "招聘", "知识库"
    ]):
        scenario = "document_analysis"
    elif any(k in query_lower for k in ["企业", "部署", "安全", "enterprise", "deploy", "合规", "私有", "本地化", "内网"]):
        scenario = "enterprise"
        scenario = "enterprise"
    elif any(k in query_lower for k in ["创意", "写作", "creative", "writing", "文案", "小说", "故事", "章节续写", "视频脚本", "口播", "营销"]):
        scenario = "creative"
        
    # 2. 预算匹配
    max_price = None
    currency = "CNY"
    
    # 尝试匹配价格，如 "10元", "10块", "$5"
    price_cny = re.search(r'(\d+(?:\.\d+)?)\s*(?:元|块|cny)', query_lower)
    price_usd = re.search(r'(?:\$|usd)\s*(\d+(?:\.\d+)?)', query_lower)
    price_generic = re.search(r'价格\D*(\d+(?:\.\d+)?)', query_lower)
    
    if price_usd:
        max_price = float(price_usd.group(1))
        currency = "USD"
    elif price_cny:
        max_price = float(price_cny.group(1))
        currency = "CNY"
    elif price_generic:
        max_price = float(price_generic.group(1))
        currency = "CNY"

    # 3. 语言偏好
    languages = []
    if any(k in query_lower for k in ["中文", "汉语", "chinese"]):
        languages.append("中文")
    if any(k in query_lower for k in ["英文", "英语", "english"]):
        languages.append("英文")
    if not languages:
        languages = ["中文"] # 默认中文环境

    # 4. 性能要求
    context_min = 0
    if "长文本" in query_lower or "long context" in query_lower or "128k" in query_lower:
        context_min = 128000
    elif "8k" in query_lower:
        context_min = 8000
    elif "32k" in query_lower:
        context_min = 32000
        
    latency = "medium"
    if any(k in query_lower for k in ["低延迟", "实时", "快", "fast", "low latency"]):
        latency = "high" # 延迟高优先级 = 越快越好
        
    # 5. 特性偏好
    vision = None
    if any(k in query_lower for k in ["多模态", "图片", "视觉", "vision", "image"]):
        vision = True
    
    func_call = None
    if any(k in query_lower for k in ["函数调用", "工具", "function", "tool", "plugin"]):
        func_call = True

    # 6. 关键词提取
    keywords = [k for k in re.split(r'[\s,，.。;；\-_]+', query) if len(k) > 1]
    
    return {
        "scenario": scenario,
        "budget_range": {"max_price_per_million": max_price, "currency": currency},
        "language_preference": languages,
        "performance_requirements": {
            "context_window_min": context_min,
            "latency_priority": latency,
            "throughput_priority": "medium"
        },
        "feature_requirements": {
            "function_calling": func_call,
            "vision_support": vision
        },
        "keywords": keywords[:5],
        "raw_query": query
    }


def requirement_parser_node(state: dict) -> dict:
    """需求解析 Agent 节点 — 将自然语言输入转化为结构化过滤契约并输出深度分析师思考"""
    query = state.get("raw_query", "") or state.get("query", "")
    trace_logs = list(state.get("trace_logs", []))
    
    trace_msg = f"[RequirementParser Agent] 启动需求理解，原始查询: \"{query}\""
    print(trace_msg)
    trace_logs.append(trace_msg)
    
    if not query:
        fallback = parse_requirement_offline("")
        fallback["analyst_thought"] = "【大模型竞品分析师】当前未检测到具体的有效自然语言诉求，系统已切换至默认通用场景，采用全局高吞吐低延迟性价比厂商组合进行冷备部署评估。"
        return {"parsed_requirement": fallback, "trace_logs": trace_logs}

    system_prompt = """你是一个顶级云原生大模型竞品分析与企业级架构设计专家。请将用户的自然语言竞品分析诉求，精确提炼为以下 JSON 格式：
{
    "scenario": "code_development|customer_service|data_analysis|document_analysis|enterprise|creative|general",
    "budget_range": {"max_price_per_million": null_or_float, "currency": "CNY|USD"},
    "language_preference": ["中文", "英文"],
    "performance_requirements": {
        "context_window_min": integer,
        "latency_priority": "high|medium|low",
        "throughput_priority": "high|medium|low"
    },
    "feature_requirements": {"function_calling": null_or_boolean, "vision_support": null_or_boolean},
    "keywords": ["关键词1", "关键词2"],
    "raw_query": "原始查询文本",
    "analyst_thought": "详细深入的竞品选型宏观思考（150字-250字中文）。请深度剖析用户该场景的核心技术挑战，并基于各厂商‘旗舰模型 vs 性价比模型’的 Portfolio 双层级产品现状，阐述在此场景下应该关注哪些核心指标，如何针对性地平衡性价比（性价比考虑），规避潜在的技术性陷阱。"
}

【极其重要解析原则】：
1. 用户的“当前过滤诉求（raw_query）”是绝对的最高优先级，提供的“已知背景知识/参考快讯”仅作通识参考。
2. 严禁将背景知识中关于特定场景的限制（如其他代码开发任务的预算不超过10元上限、必须支持函数调用或视觉多模态等硬性要求）直接抄袭或带入到当前需求的解析 JSON 字段中！
3. 如果用户在当前的 raw_query 中没有明确提出价格预算上限，则 budget_range 中的 max_price_per_million 必须解析为 null。
4. 如果用户在当前的 raw_query 中没有明确提到必须支持函数调用或视觉多模态，则 feature_requirements 中的对应字段 must be parsed as null.
5. 场景(scenario)智能归类规则：
   - 小说、故事、章节续写、视频脚本、口播、内容创作、营销文案 -> creative (创意写作)
   - 上传Excel、CSV或数据库数据、自动分析趋势、生成图表、解释异常、输出报告、数据分析助手、BI报表 -> data_analysis (数据分析)
   - 论文、学术写作、文献、合同审查、简历、长文档处理、知识库问答 -> document_analysis (文档分析)
   - 企业合规、私有部署、内网、本地推理 -> enterprise (企业合规)
   - 代码、编程、开发、Agent 助手 -> code_development (代码开发)
   - 客服、聊天对话 -> customer_service (客服系统)
   - 成本优化、多模型协作、工作流调度 -> general (通用/成本)

请严格只返回该 JSON，不能有任何 Markdown 标记或多余文字。"""

    prompt = f"请解析以下竞品过滤诉求：\n\"{query}\""
    
    pre_context = state.get("pre_retrieved_context", "")
    if pre_context:
        prompt = f"【已知行业快讯与历史分析背景知识】\n{pre_context}\n\n请结合上述背景知识，解析以下竞品过滤诉求：\n\"{query}\"\n并请在生成的 `analyst_thought` 中结合这些行业背景信息，给出深度定制的架构分析。"
        trace_msg = "[RequirementParser Agent] RAG context injected successfully."
        print(trace_msg)
        trace_logs.append(trace_msg)

    parsed = None
    try:
        llm_res = call_llm(prompt, system_prompt)
        # 清理可能存在的 markdown 标记
        cleaned = re.sub(r'```json\s*|\s*```', '', llm_res).strip()
        parsed = json.loads(cleaned)
        
        # 字段校验与修复
        if "scenario" not in parsed: parsed["scenario"] = "general"
        if "budget_range" not in parsed: parsed["budget_range"] = {"max_price_per_million": None, "currency": "CNY"}
        if "language_preference" not in parsed: parsed["language_preference"] = ["中文"]
        
        # 提取分析师思考并展示在并发控制台
        thought = parsed.get("analyst_thought", "")
        if thought:
            trace_logs.append(f"🧠 **【大模型竞品分析师 · 业务场景与痛点洞察】**\n> {thought}")
            print(f"[Analyst Thought] {thought}")
            
        trace_msg = f"[RequirementParser Agent] LLM 智能解析成功，提取场景: {parsed.get('scenario')}"
        print(trace_msg)
        trace_logs.append(trace_msg)
    except Exception as e:
        trace_msg = f"[RequirementParser Agent] LLM 解析异常 ({e})，降级为离线规则引擎匹配。"
        print(trace_msg)
        trace_logs.append(trace_msg)
        parsed = parse_requirement_offline(query)
        
        # 针对不同场景生成极高质量的离线专家分析师思考
        offline_thoughts = {
            "code_development": "【离线专家分析】检测到代码编程与开发场景。该场景对上下文内代码依赖的完整性（如AST树解析）、复杂算法规划有极高准度要求。虽然传统上多偏向顶奢模型如 Claude 3 Opus，但其极高单价难以应对高并发。建议重点关注深度求索（DeepSeek）和 OpenAI Instant 等兼备极高性价比（性价比考虑）与强大代码补全的模型组合，实施高频编码智能路由路由策略。",
            "customer_service": "【离线专家分析】检测到智能客服场景。客服对话对于首字响应延迟（TTFB）与处理吞吐有着近乎严苛的要求，且属于日请求量极大的高并发领域。此时盲目选择高溢价旗舰模型（如 claude-opus）将导致高昂的企业账单。应当重点路由至支持高吞吐、价格在 1 元/M tokens 左右的国内厂商（如火山引擎、深度求索），并将旗舰模型作为高疑难解答的兜底冷通道。",
            "data_analysis": "【离线专家分析】检测到 AI 数据分析与自动化报表场景。该场景要求极高的数据/表格理解、代码执行（Python/SQL）以及图表生成与异常解释能力。针对此场景，评估应重点关注模型在统计分析、结构化输出、Python/SQL 代码生成与执行环境等方面的能力表现，以及产品级的文件上传、图表配置、报表模板、导出和数据脱敏安全保障。",
            "document_analysis": "【离线专家分析】检测到长文档或报告分析场景。长文本分析的胜负手在于长程注意力（Long-Context Attention）与精准的信息抽取（针锋相对测试）。谷歌 Gemini 的百万级窗口与月之暗面 Kimi 的极长文本无损处理是无可争议的第一梯队。在此场景中，应设计以长文本吞吐为优先的策略，并在报告中权衡高能效模型与旗舰版的多阶梯折扣优势。",
            "enterprise": "【离线专家分析】检测到企业级合规部署场景。企业级业务的重中之重是数据隐私合规与高稳定性。国内厂商在政企合规优势、本地化私有云部署方面领先，而国际前沿厂商则能满足出海及多模态高难推理。分析应重点对国内外厂商的合规安全与 SLA 表现进行多维度打分加权。",
            "creative": "【离线专家分析】检测到创意写作与文案创意场景。该场景对大模型的创新力（创新表现）和发散联想能力要求较高。OpenAI 与 Anthropic 的最新版模型在中文及英文创意语篇输出中各有千秋。在此场景中，我们将调高用户满意度与创新路线图维度的权重，以捕获最有文学灵性的生成厂商。"
        }
        parsed["analyst_thought"] = offline_thoughts.get(parsed.get("scenario"), "【离线专家分析】通用场景分析启动。评估策略将优先采用性价比（性价比考虑）最优排名曲线，推荐具备全能通识表现的厂商矩阵，规避单价过高的技术供应商，提供稳健的企业多模型路由配置方案。")
        trace_logs.append(f"🧠 **【大模型竞品分析师 · 业务场景与痛点洞察 (离线降级版)】**\n> {parsed['analyst_thought']}")

    return {"parsed_requirement": parsed, "trace_logs": trace_logs}
