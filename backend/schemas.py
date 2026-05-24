from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict
from datetime import datetime

class PricingSchema(BaseModel):
    prompt_price_per_million: float = Field(..., description="每百万 Prompt Token 价格（单位为 currency）")
    completion_price_per_million: float = Field(..., description="每百万 Completion Token 价格（单位为 currency）")
    currency: str = Field("USD", description="计价货币，如 USD, CNY")

class RateLimitsSchema(BaseModel):
    rpm: int = Field(..., description="每分钟最大请求数 (Requests Per Minute)")
    tpm: int = Field(..., description="每分钟最大 Token 数 (Tokens Per Minute)")

class FeaturesSchema(BaseModel):
    context_window: int = Field(..., description="支持的最大上下文 Token 长度")
    function_calling: bool = Field(..., description="是否支持函数调用 (Tools API)")
    vision_support: bool = Field(..., description="是否支持视觉/多模态输入")

class SourceTrace(BaseModel):
    snippet: str = Field(..., description="抽取的源文本片段/证据")
    url: str = Field(..., description="提取的源网页 URL")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

class EvidenceItem(BaseModel):
    """
    证据条目 Schema。实现 Evidence-first 竞品可信分析。
    示例:
        evidence = EvidenceItem(
            evidence_id="ev_001",
            source_type="web",
            title="OpenAI Pricing Policy",
            url="https://openai.com/pricing",
            snippet="gpt-4o input is $5 per million tokens",
            collected_at=datetime.utcnow(),
            competitor="OpenAI",
            reliability_score=0.95
        )
    """
    evidence_id: str = Field(..., description="证据条目的唯一标识符")
    source_type: str = Field(..., description="数据来源类型，如 web, pdf, api_doc, official_announcement")
    title: str = Field(..., description="证据源页面或文档的标题")
    url: str = Field(..., description="证据提取的原始 URL 来源")
    snippet: str = Field(..., description="抽取的源文本片段/证据")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="证据采集时间戳")
    competitor: str = Field(..., description="对应的竞品厂商名称")
    reliability_score: float = Field(..., description="数据源可靠性得分 (0.0 - 1.0)")

class AnalysisClaim(BaseModel):
    """
    分析主张/结论 Schema。通过 evidence_ids 关联多个原始证据条目。
    示例:
        claim = AnalysisClaim(
            claim_id="cl_001",
            claim_text="OpenAI gpt-4o 比前代降价 50%",
            dimension="pricing",
            competitor="OpenAI",
            confidence=0.9,
            evidence_ids=["ev_001"],
            risk_note="仅针对标准 API，不排除未来微调定价调整风险"
        )
    """
    claim_id: str = Field(..., description="分析结论/主张的唯一标识符")
    claim_text: str = Field(..., description="具体提取的核心分析结论与论断")
    dimension: str = Field(..., description="该论断所属维度，如 pricing, rate_limits, features, coding_plan, user_feedback")
    competitor: str = Field(..., description="对应的竞品厂商名称")
    confidence: float = Field(..., description="该论断的置信度评分 (0.0 - 1.0)")
    evidence_ids: List[str] = Field(default_factory=list, description="关联的 EvidenceItem.evidence_id 列表，实现多对多证据回溯")
    risk_note: Optional[str] = Field(None, description="潜在的不确定性或幻觉风险提示说明")

class UserFeedbackSchema(BaseModel):
    developer_satisfaction: float = Field(..., description="开发者生态整体满意度评分 (0.0 - 5.0)")
    strengths: List[str] = Field(..., description="用户核心满意度优势/亮点列表")
    pain_points: List[str] = Field(..., description="用户主要吐槽、痛点或局限性列表")

class CodingPlanSchema(BaseModel):
    is_supported_in_editor: bool = Field(..., description="是否支持 IDE/编辑器插件无缝嵌入（如 Cursor, VS Code）")
    language_optimizations: List[str] = Field(..., description="特别优化支持的编程语言列表")
    has_sandbox_env: bool = Field(..., description="是否提供免费的测试沙盒或免费额度开发者环境")
    plan_description: str = Field(..., description="编码专项服务或开发者优惠计划细节描述")

class ModelIntelligenceSchema(BaseModel):
    model_name: str = Field(..., description="模型名称，例如 gemini-3.5-flash 或 gemini-3.5-pro")
    pricing: PricingSchema
    rate_limits: RateLimitsSchema
    features: FeaturesSchema
    user_feedback: UserFeedbackSchema
    coding_plan: CodingPlanSchema

class CompetitorIntelligence(BaseModel):
    provider_name: str = Field(..., description="厂商名称，例如 OpenAI, 火山引擎, 深度求索, Anthropic")
    region: str = Field(..., description="厂商区域，国内(domestic) 或 国外(international)")
    model_family: str = Field(..., description="代表性核心模型系列")
    pricing: PricingSchema
    rate_limits: RateLimitsSchema
    features: FeaturesSchema
    user_feedback: UserFeedbackSchema  # 用户舆情与生态反馈契约
    coding_plan: CodingPlanSchema      # 新增：开发者编程开发专项支持契约
    models: List[ModelIntelligenceSchema] = Field(default_factory=list, description="该厂商下的核心多个模型系列")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # 可追溯性与信息溯源 Trace
    sources: Dict[str, SourceTrace] = Field(
        default_factory=dict, 
        description="字段到信息源头 Trace 的映射表，键对应属性名（如 'pricing.prompt_price_per_million'）"
    )
    
    # 合规脱敏状态标记
    is_sanitized: bool = Field(default=False, description="数据是否已通过脱敏过滤")
    sanitized_fields: List[str] = Field(default_factory=list, description="被脱敏过滤的敏感字段或文本片段")

    # 新增 Evidence-first 与 Claim 可追溯性支持（向后兼容：默认置为空列表）
    claims: List[AnalysisClaim] = Field(default_factory=list, description="针对该竞品的分析结论与证据映射列表")
    evidences: List[EvidenceItem] = Field(default_factory=list, description="针对该竞品的原始证据列表")

# ==========================================
# Phase 1-4 新增数据模型契约
# ==========================================

class SmartAnalyzeRequest(BaseModel):
    query: str = Field(..., description="智能分析自然语言输入")

class BudgetRange(BaseModel):
    max_price_per_million: Optional[float] = Field(None, description="最大可接受的每百万 Token 价格")
    currency: str = Field("CNY", description="计价货币，如 USD, CNY")

class PerformanceRequirements(BaseModel):
    context_window_min: int = Field(0, description="最小需要的上下文 Token 长度")
    latency_priority: str = Field("medium", description="延迟优化优先级 (high|medium|low)")
    throughput_priority: str = Field("medium", description="吞吐量优先级 (high|medium|low)")

class FeatureRequirements(BaseModel):
    function_calling: Optional[bool] = Field(None, description="是否必须支持函数调用")
    vision_support: Optional[bool] = Field(None, description="是否必须支持视觉多模态")

class ParsedRequirement(BaseModel):
    scenario: str = Field("general", description="提取的目标业务场景")
    budget_range: BudgetRange
    language_preference: List[str] = Field(default_factory=lambda: ["中文"], description="语言偏好列表")
    performance_requirements: PerformanceRequirements
    feature_requirements: FeatureRequirements
    keywords: List[str] = Field(default_factory=list, description="关键字列表")
    raw_query: str = Field("", description="原始查询文本")

class VendorRecommendation(BaseModel):
    vendor_name: str = Field(..., description="厂商名称")
    match_score: float = Field(..., description="需求匹配得分")
    match_reasons: List[str] = Field(default_factory=list, description="匹配推荐原因亮点")
    auto_selected: bool = Field(False, description="是否被系统自动勾选推荐")

class FeedbackRequest(BaseModel):
    report_id: str = Field(..., description="评估报告的 ID")
    rating: int = Field(..., description="反馈星级评分 1-5")
    comments: str = Field("", description="具体改进评论建议")
    vendor_list: Optional[List[str]] = Field(None, description="分析的厂商列表")

