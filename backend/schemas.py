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
