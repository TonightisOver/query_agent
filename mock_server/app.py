from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Mock LLM Provider Pages - 2026 Edition")

# 1. OpenAI (gpt-5.5-instant)
OPENAI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenAI API Pricing & Rate Limits</title>
</head>
<body>
    <h1>OpenAI API Pricing Page</h1>
    <p>Welcome to the official developer portal. Below are our current pricing models for developer API access.</p>
    
    <h2>Models & Pricing</h2>
    <div id="pricing-gpt55-instant">
        <h3>Model: gpt-5.5-instant</h3>
        <ul>
            <li><strong>Input Token Price:</strong> $1.10 per 1M tokens</li>
            <li><strong>Output Token Price:</strong> $4.40 per 1M tokens</li>
            <li><strong>Context Window:</strong> 200000 tokens</li>
            <li><strong>Features:</strong> High-speed reasoning, Multi-step agentic planning, Voice & Vision support.</li>
        </ul>
    </div>
 
    <h2>Rate Limits</h2>
    <p>Default tier limits for production accounts:</p>
    <ul>
        <li><strong>RPM (Requests Per Minute):</strong> 10000 requests</li>
        <li><strong>TPM (Tokens Per Minute):</strong> 1000000 tokens</li>
    </ul>
 
    <h2>Developer Eco Feedback & Sentiment</h2>
    <p>According to our latest developer community feedback survey, the overall satisfaction rating for the OpenAI gpt-5.5-instant model has reached an impressive 4.8 out of 5.
    Developers rave about its core strengths: [unprecedented ultra-low latency reasoning] and [excellent cost-performance trade-off for complex programming and chat].
    However, the main user complaints and pain points are: [slight cognitive regression compared to full GPT-5.5 flagship on edge cases] and [heavy prompt token consumption during long-term agentic executions].</p>
 
    <h2>Coding Plan & Developer Support</h2>
    <p>We are excited to launch the OpenAI Coding-Plan for 2026. This plan features:
    - <strong>IDE Integration:</strong> [Yes, fully supported in Cursor, VS Code, and major editors].
    - <strong>Language Optimizations:</strong> [Specifically optimized for Python, C++, and TypeScript].
    - <strong>Developer Sandbox:</strong> [Yes, provided as a free sandbox environment for enterprise APIs].
    - <strong>Summary Description:</strong> [We provide professional API test sandbox for verified developers on our enterprise coding tier].</p>
 
    <h2>Developer Admin Contact & Security Note</h2>
    <!-- 故意包含敏感测试信息，用于安全和脱敏 Agent 识别过滤 -->
    <p>For urgent API issues, contact our emergency developer group lead on phone: +86-13812345678 (Zhang Wei) or at admin-dev-private@openai.com.</p>
    <p>WARNING: Internal sandbox debug API Key (DO NOT DISTRIBUTE): sk-proj-89123hskdJHSAKJ1238912389123 (Temporary Sandbox for Test-Dev Group).</p>
</body>
</html>
"""

# 2. Anthropic (claude-opus-4.7)
ANTHROPIC_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Anthropic Claude API Reference & Cost</title>
</head>
<body>
    <h1>Claude Opus 4.7 Reference</h1>
    
    <h2>Cost</h2>
    <ul>
        <li><strong>Model: claude-opus-4.7-20260416</strong></li>
        <li><strong>Input Pricing:</strong> $15.00 per million tokens</li>
        <li><strong>Output Pricing:</strong> $75.00 per million tokens</li>
        <li><strong>Context Window:</strong> 200000 tokens</li>
        <li><strong>Function Calling:</strong> Supported (Tools API)</li>
        <li><strong>Vision Support:</strong> Yes (Image & PDF high-res multimodal inputs supported)</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 4000</li>
        <li><strong>TPM:</strong> 400000</li>
    </ul>

    <h2>Developer Sentiment & Reviews</h2>
    <p>The developer community overall satisfaction rating for Claude Opus 4.7 is a stellar 4.9 out of 5.
    Key strengths praised by community include: [groundbreaking multi-step reasoning capabilities with complex code libraries] and [industry-leading coding accuracy and instruction following].
    Main complaints and pain points reported: [extremely high token cost for large-context tasks] and [strict rate limiting during peak commercial hours].</p>

    <h2>Coding Plan</h2>
    <p>The Anthropic 2026 Developer Coding Support details:
    - <strong>IDE Plugins:</strong> [Yes, native support in Cursor, VS Code, and IntelliJ].
    - <strong>Optimized Languages:</strong> [Specially optimized for TypeScript, Python, Java, and Rust].
    - <strong>Sandbox Environment:</strong> [No, we do not provide a free sandbox environment currently].
    - <strong>Detail:</strong> [We offer dedicated developer SDKs and deeply integrate with Cursor, providing premium coding capabilities without free sandboxes].</p>

    <h2>Security Mask Test</h2>
    <p>For support, call +86-13822223333 (Secret Channel). Sandbox key is sk-proj-anthropic1234567890abcdef.</p>
</body>
</html>
"""

# 3. Google (gemini-3.5-flash)
GOOGLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Google Gemini Developer Studio Pricing</title>
</head>
<body>
    <h1>Gemini 3.5 Flash Details</h1>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: gemini-3.5-flash</strong></li>
        <li><strong>Input Token Price:</strong> $0.10 per 1M tokens</li>
        <li><strong>Output Token Price:</strong> $0.40 per 1M tokens</li>
        <li><strong>Context Window:</strong> 2000000 tokens</li>
        <li><strong>Function Calling:</strong> Supported</li>
        <li><strong>Vision Support:</strong> Yes</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 2000</li>
        <li><strong>TPM:</strong> 200000</li>
    </ul>

    <h2>Developer Reviews</h2>
    <p>Google Gemini 3.5 Flash overall satisfaction rating: 4.65 out of 5.
    Strengths: [extremely high token processing speed and low latency] and [top-tier multi-step agentic execution and tool calling].
    Pain points: [occasional over-safety filtering that blocks valid responses].</p>

    <h2>Coding Plan</h2>
    <p>Google AI Studio Coding Plan:
    - <strong>IDE Support:</strong> [Yes, integrated with Cursor, IDX, and VS Code].
    - <strong>Optimized Languages:</strong> [Specially optimized for Python, Kotlin, and Go].
    - <strong>Free Sandbox:</strong> [Yes, provided for testing].
    - <strong>Description:</strong> [Google AI Studio provides a free daily request sandbox for registered developers to test API integration].</p>

    <h2>Security MASK</h2>
    <p>Admin: +86-13911112222. Secret token is sk-proj-gemini8888888888888888888.</p>
</body>
</html>
"""

# 4. Mistral (mistral-large-3)
MISTRAL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mistral AI - Models and Pricing</title>
</head>
<body>
    <h1>Mistral Large 3</h1>
    
    <h2>Specs</h2>
    <ul>
        <li><strong>Model: mistral-large-3</strong></li>
        <li><strong>Input Price:</strong> $2.00 per 1M tokens</li>
        <li><strong>Output Price:</strong> $6.00 per 1M tokens</li>
        <li><strong>Context Window:</strong> 128000 tokens</li>
        <li><strong>Function Calling:</strong> Supported</li>
        <li><strong>Vision Support:</strong> Yes</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 5000</li>
        <li><strong>TPM:</strong> 300000</li>
    </ul>

    <h2>Feedback</h2>
    <p>European developers rank Mistral Large 3 with 4.4 out of 5 satisfaction.
    Strengths: [sovereign European data hosting] and [high autonomy with customizable fine-tuning].
    Pain points: [logical reasoning is slightly behind GPT-5.5 on extremely complex tasks].</p>

    <h2>Coding Plan</h2>
    <p>Mistral La Plateforme Coding Plan:
    - <strong>IDE Integration:</strong> [Yes, supported via standard OpenAI compatible API in VS Code/Cursor].
    - <strong>Optimized Languages:</strong> [Specially optimized for C++, Python, and TypeScript].
    - <strong>Free Sandbox:</strong> [Yes, Mistral provides sandbox keys].
    - <strong>Description:</strong> [Our La Plateforme provides developers with sandbox test keys and按量优惠 discounts].</p>

    <h2>合规掩码测试</h2>
    <p>客服电话：400-900-8888。Key: sk-proj-mistral777777777777777777.</p>
</body>
</html>
"""

# 5. 火山引擎 (Doubao-Seed-2.0)
DOUBAO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>火山引擎 - 豆包大模型 API 价格与规格</title>
</head>
<body>
    <h1>豆包大模型开发者平台</h1>
    <p>助力企业高效创新，提供最具性价比和极致性能的 LLM 服务。</p>
    
    <h2>定价信息</h2>
    <div id="pricing-doubao-seed-2">
        <h3>Model: Doubao-Seed-2.0</h3>
        <ul>
            <li><strong>输入 Token 价格:</strong> 0.0008元/千 tokens (相当于 0.8元/1M tokens)</li>
            <li><strong>输出 Token 价格:</strong> 0.002元/千 tokens (相当于 2.0元/1M tokens)</li>
            <li><strong>计价货币:</strong> CNY (人民币)</li>
            <li><strong>上下文窗口:</strong> 131072 tokens</li>
            <li><strong>特性支持:</strong> 深度支持 Function Calling (函数调用), 视觉识别及多模态输入.</li>
        </ul>
    </div>

    <h2>并发限流 (Rate Limits)</h2>
    <p>标准端点并发限制：</p>
    <ul>
        <li><strong>RPM (每分钟请求数):</strong> 30000 请求</li>
        <li><strong>TPM (每分钟 Token 数):</strong> 1200000 tokens</li>
    </ul>

    <h2>开发者生态评价与反馈</h2>
    <p>根据国内开发者社区满意度调查，火山引擎豆包大模型 Seed-2.0 获得了 4.7分（满分 5.0）。
    核心优势在于：[无与伦比的极低计费性价比] 以及 [国内高并发高吞吐量极佳的稳定性]。
    用户主要吐槽的局限性和痛点是：[在极其复杂的逻辑算法推理场景中表现稍逊于 GPT-5.5] 以及 [多语言及国际化任务处理能力尚有提升空间]。</p>

    <h2>Coding Plan 开发者计划</h2>
    <p>豆包 2026 专项编码计划：
    - <strong>编辑器插件集成:</strong> [Yes, 深度整合字节跳动旗下 MarsCode 编辑器，并支持 Cursor/VS Code].
    - <strong>语言特别优化:</strong> [特别针对 Python, Go, 和 Java 语言的生成与调试进行底层框架级优化].
    - <strong>沙盒环境支持:</strong> [Yes, 注册即赠送开发者测试沙盒额度].
    - <strong>计划详情:</strong> [注册即可获得 5 亿免费 Token 调试礼包，且支持 MarsCode 云端编辑器无缝协同调试].</p>

    <h2>合规声明</h2>
    <p>本服务严格符合合规隐私安全标准。测试客服电话：400-100-1111（数据脱敏测试组专用）。 Sandbox秘钥：sk-proj-doubao9999999999999999999。</p>
</body>
</html>
"""

# 6. 深度求索 (DeepSeek-V4-Pro)
DEEPSEEK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DeepSeek API Pricing & Documentation</title>
</head>
<body>
    <h1>DeepSeek V4 Pro API</h1>
    <p>Providing top-tier open source LLM API service with extremely low price.</p>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: DeepSeek-V4-Pro</strong></li>
        <li><strong>Input Price (Cache Miss):</strong> $0.435 per 1M tokens</li>
        <li><strong>Input Price (Cache Hit):</strong> $0.04 per 1M tokens</li>
        <li><strong>Output Price:</strong> $0.87 per 1M tokens</li>
        <li><strong>Currency:</strong> USD</li>
        <li><strong>Context Window:</strong> 1000000 tokens</li>
        <li><strong>Features:</strong> Function calling, JSON Mode, Deep thinking reasoning output, 1M Context.</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 10000</li>
        <li><strong>TPM:</strong> 300000</li>
    </ul>

    <h2>Developer Ecosystem Feedback</h2>
    <p>DeepSeek V4 Pro has went viral globally, with its overall developer satisfaction rating climbing to a record-breaking 4.95 out of 5.
    Our survey highlights its unbeatable strengths: [highly transparent deep reasoning thinking chain output] and [unmatched cost-efficiency which is 1/100 of GPT-5.5].
    However, the critical user pain points and complaints consist of: [frequent API connection timeouts and capacity busy errors during peak traffic] and [extremely tight default RPM and TPM rate limit caps].</p>

    <h2>Coding Plan 专项支持</h2>
    <p>DeepSeek 2026 Coder Plan：
    - <strong>IDE 插件支持:</strong> [Yes, 全网主流 Cursor, VS Code 标配 API 选择].
    - <strong>特定优化语言:</strong> [特别优化支持 Python, C++, Java, 和 Rust，代码生成能力冠绝开源界].
    - <strong>测试沙盒:</strong> [Yes, 提供首月调试 Sandbox 及测试令牌].
    - <strong>计划详情:</strong> [新注册账号立赠千万级 Token 沙盒调试额度，API 全透明对齐 OpenAI 契约，支持零门槛接入 IDE].</p>

    <h2>合规脱敏</h2>
    <p>联系电话：+86-13788889999 (紧急支持)。开发调试 Key：sk-proj-deepseek666666666666666666。</p>
</body>
</html>
"""

# 7. 智谱AI (GLM-5.1)
ZHIPU_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>智谱AI 大模型开放平台</title>
</head>
<body>
    <h1>智谱 GLM-5.1</h1>
    
    <h2>定价与能力</h2>
    <ul>
        <li><strong>模型: GLM-5.1</strong></li>
        <li><strong>输入 Token 价格:</strong> 5.0元/1M tokens</li>
        <li><strong>输出 Token 价格:</strong> 5.0元/1M tokens</li>
        <li><strong>上下文窗口:</strong> 256000 tokens</li>
        <li><strong>特性:</strong> 强力函数调用, 高清晰多模态视频理解, 智能体 Agent API, 国产算力华为昇腾深度适配.</li>
    </ul>

    <h2>速率限制</h2>
    <ul>
        <li><strong>RPM:</strong> 8000</li>
        <li><strong>TPM:</strong> 500000</li>
    </ul>

    <h2>开发者满意度</h2>
    <p>国内大模型社区给予 GLM-5.1 整体满意度打分：4.5 分（满分 5.0）。
    核心优势：[业内顶尖的多模态视频理解能力] 以及 [全栈国产算力平台华为昇腾深度适配，自主性极高]。
    主要局限与痛点：[长文本检索在极端细节处仍偶发微小幻觉] 以及 [推理响应在大并发下有轻微队列排队延迟]。</p>

    <h2>Coding Plan</h2>
    <p>智谱 2026 Coding-Plan 专项服务：
    - <strong>IDE 嵌入:</strong> [Yes, 支持集成在主流编辑器及智谱专享插件].
    - <strong>语言优化:</strong> [针对企业级 Java, Python, 和 C# 进行了深入代码语义优化].
    - <strong>测试沙盒:</strong> [Yes, 提供首年开发者测试沙盒环境].
    - <strong>描述:</strong> [智谱专为企业提供大模型智能体 SDK 调试，配有首年开发者测试沙盒环境及按量优惠折扣].</p>

    <h2>安全校验</h2>
    <p>技术电话：+86-13611110000。 Key: sk-proj-zhipu5555555555555555555.</p>
</body>
</html>
"""

# 8. 阿里通义千问 (Qwen3.7-Max)
QWEN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>阿里云百炼大模型服务平台</title>
</head>
<body>
    <h1>通义千问 Qwen3.7-Max</h1>
    
    <h2>服务指标</h2>
    <ul>
        <li><strong>模型: Qwen3.7-Max</strong></li>
        <li><strong>输入价格:</strong> 2.5元/1M tokens</li>
        <li><strong>输出价格:</strong> 10.0元/1M tokens</li>
        <li><strong>上下文长度:</strong> 128000 tokens</li>
        <li><strong>特性支持:</strong> Function calling, 多模态输入, 原生中文代码推理, AI智能体长周期连续决策底座.</li>
    </ul>

    <h2>限流限制</h2>
    <ul>
        <li><strong>RPM:</strong> 15000</li>
        <li><strong>TPM:</strong> 800000</li>
    </ul>

    <h2>口碑与吐槽</h2>
    <p>阿里云社区的开发者满意度打分高达 4.8分（满分 5.0）。
    主要优势：[专为AI智能体深度定制的连续决策底座] 和 [极其优越的代码编写能力与复杂长周期工具遵循性能]。
    吐槽与局限性：[高并发流量瞬间超限后，重试排队带来的延迟偏高] 以及 [在英文冷门学术专业领域相比 GPT-5.5 稍有提升空间]。</p>

    <h2>Coding Plan</h2>
    <p>阿里云百炼 Qwen Coder 2026 支持计划：
    - <strong>IDE 插件支持:</strong> [Yes, 完全兼容 VS Code, Cursor 等全系列大模型编码插件].
    - <strong>语言优化:</strong> [特别优化 TypeScript, Python, C++, 和 Go 语言，语法诊断精准].
    - <strong>沙盒测试环境:</strong> [Yes, 百炼平台提供免费测试 sandbox].
    - <strong>描述:</strong> [阿里云百炼大盘专享 Qwen Coder 免费接入沙盒测试，配有专属按量特惠编码通道].</p>

    <h2>隐私遮蔽</h2>
    <p>热线: 400-800-9999. 沙盒 Key: sk-proj-qwen33333333333333333333.</p>
</body>
</html>
"""

@app.get("/mock/openai", response_class=HTMLResponse)
def get_openai_page():
    return OPENAI_HTML

@app.get("/mock/doubao", response_class=HTMLResponse)
def get_doubao_page():
    return DOUBAO_HTML

@app.get("/mock/anthropic", response_class=HTMLResponse)
@app.get("/mock/claude", response_class=HTMLResponse)
def get_anthropic_page():
    return ANTHROPIC_HTML

@app.get("/mock/deepseek", response_class=HTMLResponse)
def get_deepseek_page():
    return DEEPSEEK_HTML

@app.get("/mock/google", response_class=HTMLResponse)
@app.get("/mock/gemini", response_class=HTMLResponse)
def get_google_page():
    return GOOGLE_HTML

@app.get("/mock/mistral", response_class=HTMLResponse)
def get_mistral_page():
    return MISTRAL_HTML

@app.get("/mock/zhipu", response_class=HTMLResponse)
@app.get("/mock/glm", response_class=HTMLResponse)
def get_zhipu_page():
    return ZHIPU_HTML

@app.get("/mock/qwen", response_class=HTMLResponse)
def get_qwen_page():
    return QWEN_HTML

# ==========================================
# 新增 8 家全球厂商 (2026年5月最新)
# ==========================================

# 9. xAI (Grok-4.3)
XAI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>xAI Grok API Pricing & Documentation</title>
</head>
<body>
    <h1>xAI Grok 4.3 API</h1>
    <p>Pushing the boundaries of AI understanding with real-time knowledge and massive context.</p>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: Grok-4.3</strong></li>
        <li><strong>Input Token Price:</strong> $3.00 per 1M tokens</li>
        <li><strong>Output Token Price:</strong> $15.00 per 1M tokens</li>
        <li><strong>Context Window:</strong> 2000000 tokens</li>
        <li><strong>Features:</strong> Deep reasoning, Real-time search integration, Function calling, Vision support.</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 6000</li>
        <li><strong>TPM:</strong> 400000</li>
    </ul>

    <h2>Developer Feedback</h2>
    <p>xAI Grok 4.3 overall developer satisfaction rating: 4.6 out of 5.
    Strengths: [ultra-large 2M context window enabling full codebase-level reasoning] and [real-time search integration for up-to-date factual answers].
    Pain points: [API stability occasionally fluctuates under heavy load spikes] and [premium pricing tier is relatively expensive].</p>

    <h2>Coding Plan</h2>
    <p>xAI Developer Coding Plan 2026:
    - <strong>IDE Support:</strong> [Yes, integrated with Cursor, VS Code].
    - <strong>Optimized Languages:</strong> [Specially optimized for Python, TypeScript, and Rust].
    - <strong>Free Sandbox:</strong> [Yes, provided for registered developers].
    - <strong>Description:</strong> [xAI provides developer sandbox with generous free tier and real-time code analysis capabilities].</p>

    <h2>Security Test</h2>
    <p>Contact: +1-415-555-0123. Dev key: sk-proj-xai444444444444444444.</p>
</body>
</html>
"""

# 10. Meta (Llama-4-Maverick)
META_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Meta Llama 4 Maverick API</title>
</head>
<body>
    <h1>Meta Llama 4 Maverick</h1>
    <p>The industry-leading open-weight foundation model, available via official API and partner platforms.</p>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: Llama-4-Maverick</strong></li>
        <li><strong>Input Token Price:</strong> $0.20 per 1M tokens (via Together AI / Fireworks)</li>
        <li><strong>Output Token Price:</strong> $0.60 per 1M tokens</li>
        <li><strong>Context Window:</strong> 1000000 tokens</li>
        <li><strong>Features:</strong> Open-weight, Function calling, Self-deployment & fine-tuning, MoE architecture.</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 8000</li>
        <li><strong>TPM:</strong> 500000</li>
    </ul>

    <h2>Developer Reviews</h2>
    <p>Meta Llama 4 Maverick has a developer satisfaction rating of 4.7 out of 5.
    Strengths: [open-weight industry benchmark enabling free deployment and fine-tuning] and [million-token MoE architecture with exceptional cost-efficiency].
    Pain points: [official API ecosystem is fragmented requiring third-party platforms] and [Chinese language support slightly trails behind native domestic models].</p>

    <h2>Coding Plan</h2>
    <p>Meta AI Developer Plan:
    - <strong>IDE Support:</strong> [Yes, standard across all major platforms].
    - <strong>Optimized Languages:</strong> [Specially optimized for Python, C++, and Java].
    - <strong>Free Sandbox:</strong> [Yes, Meta AI Playground available].
    - <strong>Description:</strong> [Meta provides open model weights with free playground and enterprise deployment support through partner cloud platforms].</p>

    <h2>Security Test</h2>
    <p>Contact: +1-650-555-0456. Key: sk-proj-meta555555555555555555.</p>
</body>
</html>
"""

# 11. Cohere (Command-R-Plus-2)
COHERE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Cohere Enterprise AI - Command R+ 2</title>
</head>
<body>
    <h1>Cohere Command R+ 2</h1>
    <p>Enterprise-grade RAG and semantic search, built for business data.</p>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: Command-R-Plus-2</strong></li>
        <li><strong>Input Token Price:</strong> $2.50 per 1M tokens</li>
        <li><strong>Output Token Price:</strong> $10.00 per 1M tokens</li>
        <li><strong>Context Window:</strong> 256000 tokens</li>
        <li><strong>Features:</strong> Enterprise RAG, Function calling, Multi-language support, SOC 2 certified.</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 3000</li>
        <li><strong>TPM:</strong> 250000</li>
    </ul>

    <h2>Developer Feedback</h2>
    <p>Cohere Command R+ 2 overall satisfaction: 4.3 out of 5.
    Strengths: [industry-leading enterprise RAG retrieval-augmented generation capabilities] and [SOC 2 certified data security compliance for enterprise deployments].
    Pain points: [general conversation ability slightly weaker than GPT-5.5] and [Chinese language support still needs strengthening].</p>

    <h2>Coding Plan</h2>
    <p>Cohere Developer Plan:
    - <strong>IDE Support:</strong> [Yes, supported in VS Code].
    - <strong>Optimized Languages:</strong> [Specially optimized for Python and TypeScript].
    - <strong>Free Sandbox:</strong> [Yes, trial keys available].
    - <strong>Description:</strong> [Cohere provides enterprise SDK with trial sandbox and dedicated RAG tooling for production deployments].</p>

    <h2>Security Test</h2>
    <p>Contact: +1-416-555-0789. Key: sk-proj-cohere666666666666666666.</p>
</body>
</html>
"""

# 12. Amazon Bedrock (Nova-Premier-v2)
BEDROCK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Amazon Bedrock - Nova Premier v2</title>
</head>
<body>
    <h1>Amazon Bedrock Nova Premier v2</h1>
    <p>Enterprise-grade AI platform with 100+ model catalog on AWS infrastructure.</p>
    
    <h2>Pricing</h2>
    <ul>
        <li><strong>Model: Nova-Premier-v2</strong></li>
        <li><strong>Input Token Price:</strong> $2.50 per 1M tokens</li>
        <li><strong>Output Token Price:</strong> $12.50 per 1M tokens</li>
        <li><strong>Context Window:</strong> 300000 tokens</li>
        <li><strong>Features:</strong> AWS native integration, Function calling, Multi-modal, AgentCore tooling.</li>
    </ul>

    <h2>Rate Limits</h2>
    <ul>
        <li><strong>RPM:</strong> 5000</li>
        <li><strong>TPM:</strong> 350000</li>
    </ul>

    <h2>Developer Reviews</h2>
    <p>Amazon Bedrock Nova Premier v2 satisfaction: 4.4 out of 5.
    Strengths: [AWS cloud-native enterprise one-stop model hosting platform] and [seamless integration with 100+ model catalog including Claude, Llama, DeepSeek].
    Pain points: [complex pricing tiers lack transparency] and [vendor lock-in to AWS ecosystem dependency].</p>

    <h2>Coding Plan</h2>
    <p>AWS Bedrock Developer Plan:
    - <strong>IDE Support:</strong> [Yes, AWS Cloud9 and VS Code with AWS Toolkit].
    - <strong>Optimized Languages:</strong> [Specially optimized for Python, Java, and TypeScript].
    - <strong>Free Sandbox:</strong> [Yes, AWS free tier includes Bedrock credits].
    - <strong>Description:</strong> [Amazon provides enterprise model hosting with free tier credits and unified AgentCore API for multi-model orchestration].</p>

    <h2>Security Test</h2>
    <p>Contact: +1-206-555-0321. Key: sk-proj-bedrock777777777777777777.</p>
</body>
</html>
"""

# 13. 百度文心 (ERNIE-4.5-Turbo)
BAIDU_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>百度智能云 - 文心大模型 ERNIE 4.5 Turbo</title>
</head>
<body>
    <h1>百度文心大模型 ERNIE 4.5 Turbo</h1>
    <p>国内企业级AI领军品牌，深度融合百度搜索、地图、网盘等全生态。</p>
    
    <h2>定价信息</h2>
    <ul>
        <li><strong>模型: ERNIE-4.5-Turbo</strong></li>
        <li><strong>输入 Token 价格:</strong> 4.0元/1M tokens</li>
        <li><strong>输出 Token 价格:</strong> 8.0元/1M tokens</li>
        <li><strong>计价货币:</strong> CNY (人民币)</li>
        <li><strong>上下文窗口:</strong> 128000 tokens</li>
        <li><strong>特性支持:</strong> 多模态(文本/图像/音频/视频), Function Calling, 政务金融深度合规.</li>
    </ul>

    <h2>并发限流</h2>
    <ul>
        <li><strong>RPM:</strong> 12000</li>
        <li><strong>TPM:</strong> 600000</li>
    </ul>

    <h2>开发者评价</h2>
    <p>百度文心 ERNIE 4.5 Turbo 满意度: 4.5分（满分5.0）。
    核心优势：[国内政务金融领域合规第一品牌无可替代] 以及 [百度生态(搜索/地图/网盘)深度整合实现全栈AI应用]。
    主要痛点：[海外开发者接入流程相对繁琐] 以及 [API文档国际化仍有提升空间]。</p>

    <h2>Coding Plan</h2>
    <p>百度 Comate 2026 编码计划：
    - <strong>编辑器插件:</strong> [Yes, Comate 插件深度集成 VS Code 及百度专属 IDE].
    - <strong>语言优化:</strong> [特别优化 Python, Java, C++ 语言].
    - <strong>沙盒环境:</strong> [Yes, 注册即赠测试额度].
    - <strong>详情:</strong> [百度智能云为企业客户提供 Comate 编程助手免费额度及政务专属API通道].</p>

    <h2>合规测试</h2>
    <p>客服热线：400-920-8888。测试 Key: sk-proj-ernie888888888888888888.</p>
</body>
</html>
"""

# 14. 月之暗面 (Kimi-K2.6)
MOONSHOT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>月之暗面 - Kimi K2.6 开发者平台</title>
</head>
<body>
    <h1>月之暗面 Kimi K2.6</h1>
    <p>以超长上下文和深度Agent推理为核心竞争力的新一代大模型。</p>
    
    <h2>定价信息</h2>
    <ul>
        <li><strong>模型: Kimi-K2.6</strong></li>
        <li><strong>输入 Token 价格:</strong> 2.0元/1M tokens</li>
        <li><strong>输出 Token 价格:</strong> 6.0元/1M tokens</li>
        <li><strong>计价货币:</strong> CNY (人民币)</li>
        <li><strong>上下文窗口:</strong> 262144 tokens</li>
        <li><strong>特性支持:</strong> 超长上下文建模, 深度Agent推理, Function Calling.</li>
    </ul>

    <h2>并发限流</h2>
    <ul>
        <li><strong>RPM:</strong> 10000</li>
        <li><strong>TPM:</strong> 500000</li>
    </ul>

    <h2>开发者评价</h2>
    <p>月之暗面 Kimi K2.6 满意度: 4.6分（满分5.0）。
    核心优势：[超长上下文建模能力行业领先262K+] 以及 [Agent多步自主执行能力顶尖，代码编程任务出色]。
    主要痛点：[高并发下API响应稳定性仍在优化中] 以及 [多模态视觉能力尚在追赶头部竞品]。</p>

    <h2>Coding Plan</h2>
    <p>月之暗面 2026 编码计划：
    - <strong>编辑器插件:</strong> [Yes, Kimi 插件支持 Cursor 及 VS Code].
    - <strong>语言优化:</strong> [特别优化 Python, TypeScript, Go 语言].
    - <strong>沙盒环境:</strong> [Yes, 注册赠送测试调用额度].
    - <strong>详情:</strong> [月之暗面为开发者提供 Kimi API 免费测试沙盒，支持超长文档理解及Agent调试].</p>

    <h2>合规测试</h2>
    <p>客服热线：400-800-6666。测试 Key: sk-proj-kimi999999999999999999.</p>
</body>
</html>
"""

# 15. 零一万物 (Yi-Lightning-2)
YI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>零一万物 - Yi Lightning 2 开放平台</title>
</head>
<body>
    <h1>零一万物 Yi-Lightning-2</h1>
    <p>高性能 MoE 架构大模型，代码数学推理能力突出。</p>
    
    <h2>定价信息</h2>
    <ul>
        <li><strong>模型: Yi-Lightning-2</strong></li>
        <li><strong>输入 Token 价格:</strong> 1.0元/1M tokens</li>
        <li><strong>输出 Token 价格:</strong> 4.0元/1M tokens</li>
        <li><strong>计价货币:</strong> CNY (人民币)</li>
        <li><strong>上下文窗口:</strong> 200000 tokens</li>
        <li><strong>特性支持:</strong> MoE架构, Function Calling, 代码生成专优.</li>
    </ul>

    <h2>并发限流</h2>
    <ul>
        <li><strong>RPM:</strong> 8000</li>
        <li><strong>TPM:</strong> 400000</li>
    </ul>

    <h2>开发者评价</h2>
    <p>零一万物 Yi Lightning 2 满意度: 4.4分（满分5.0）。
    核心优势：[MoE架构极致推理效率，成本优势明显] 以及 [代码数学推理能力突出，在编程基准测试中表现优异]。
    主要痛点：[品牌知名度相比头部厂商偏低] 以及 [API生态工具链尚不完善]。</p>

    <h2>Coding Plan</h2>
    <p>零一万物 2026 编码计划：
    - <strong>编辑器插件:</strong> [Yes, 支持 VS Code 及 Cursor 插件].
    - <strong>语言优化:</strong> [特别优化 Python, C++, TypeScript 语言].
    - <strong>沙盒环境:</strong> [Yes, 注册赠送测试额度].
    - <strong>详情:</strong> [零一万物通过 WorldWise 平台为企业提供多智能体部署及编码测试沙盒].</p>

    <h2>合规测试</h2>
    <p>客服热线：400-100-2222。测试 Key: sk-proj-yi0000000000000000000.</p>
</body>
</html>
"""

# 16. 商汤科技 (SenseChat-Turbo-5)
SENSETIME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>商汤科技 - 日日新 SenseChat Turbo 5</title>
</head>
<body>
    <h1>商汤科技 SenseChat-Turbo-5</h1>
    <p>计算机视觉与大语言模型深度融合的国产AI旗舰平台。</p>
    
    <h2>定价信息</h2>
    <ul>
        <li><strong>模型: SenseChat-Turbo-5</strong></li>
        <li><strong>输入 Token 价格:</strong> 3.0元/1M tokens</li>
        <li><strong>输出 Token 价格:</strong> 9.0元/1M tokens</li>
        <li><strong>计价货币:</strong> CNY (人民币)</li>
        <li><strong>上下文窗口:</strong> 128000 tokens</li>
        <li><strong>特性支持:</strong> 视觉大模型, Function Calling, 多模态图文生成, 自动驾驶与工业视觉场景.</li>
    </ul>

    <h2>并发限流</h2>
    <ul>
        <li><strong>RPM:</strong> 6000</li>
        <li><strong>TPM:</strong> 300000</li>
    </ul>

    <h2>开发者评价</h2>
    <p>商汤科技 SenseChat Turbo 5 满意度: 4.3分（满分5.0）。
    核心优势：[计算机视觉+大模型深度融合业界领先] 以及 [自动驾驶与工业视觉场景独特优势无可替代]。
    主要痛点：[纯文本推理相比专精文本模型稍逊] 以及 [社区生态活跃度仍在培育中]。</p>

    <h2>Coding Plan</h2>
    <p>商汤 2026 编码计划：
    - <strong>编辑器插件:</strong> [Yes, 支持 VS Code].
    - <strong>语言优化:</strong> [特别优化 Python, C++ 语言].
    - <strong>沙盒环境:</strong> [Yes, 日日新平台提供测试环境].
    - <strong>详情:</strong> [商汤日日新平台为开发者提供视觉+语言多模态API测试沙盒及企业定制部署方案].</p>

    <h2>合规测试</h2>
    <p>客服热线：400-680-8888。测试 Key: sk-proj-sense1111111111111111111.</p>
</body>
</html>
"""

# 新增路由
@app.get("/mock/xai", response_class=HTMLResponse)
@app.get("/mock/grok", response_class=HTMLResponse)
def get_xai_page():
    return XAI_HTML

@app.get("/mock/meta", response_class=HTMLResponse)
@app.get("/mock/llama", response_class=HTMLResponse)
def get_meta_page():
    return META_HTML

@app.get("/mock/cohere", response_class=HTMLResponse)
def get_cohere_page():
    return COHERE_HTML

@app.get("/mock/bedrock", response_class=HTMLResponse)
@app.get("/mock/amazon", response_class=HTMLResponse)
def get_bedrock_page():
    return BEDROCK_HTML

@app.get("/mock/baidu", response_class=HTMLResponse)
@app.get("/mock/ernie", response_class=HTMLResponse)
def get_baidu_page():
    return BAIDU_HTML

@app.get("/mock/moonshot", response_class=HTMLResponse)
@app.get("/mock/kimi", response_class=HTMLResponse)
def get_moonshot_page():
    return MOONSHOT_HTML

@app.get("/mock/yi", response_class=HTMLResponse)
def get_yi_page():
    return YI_HTML

@app.get("/mock/sensetime", response_class=HTMLResponse)
@app.get("/mock/sensechat", response_class=HTMLResponse)
def get_sensetime_page():
    return SENSETIME_HTML

@app.get("/")
def read_root():
    return {"status": "Mock LLM Provider Server is healthy and running on port 8080."}
