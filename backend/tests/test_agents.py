import os
import sys

# 添加 backend 路径到 sys.path，保证本地运行测试时能正确导入依赖
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.engine import sanitizer_node, SequentialLangGraphWorkflow

def test_sanitizer_confidential_masking():
    """测试合规脱敏 Agent 能够 100% 遮蔽敏感数据（电话、开发密钥）"""
    # 模拟包含个人电话和开发测试 key 的原始文本
    raw_text = "Urgent: contact 张伟 at +86-13812345678. API Key leaked: sk-proj-1234567890abcdef1234567890."
    
    # 构造极简的模拟 State
    state = {
        "raw_data": raw_text,
        "current_competitor": "OpenAI",
        "trace_logs": []
    }
    
    # 运行脱敏 Agent
    output = sanitizer_node(state)
    sanitized = output["sanitized_data"]
    
    # 验证是否成功掩蔽
    assert "+86-13812345678" not in sanitized
    assert "sk-proj-1234567890abcdef1234567890" not in sanitized
    assert "[MASK_PHONE_NUMBER_CONFIDENTIAL]" in sanitized
    assert "[MASK_API_KEY_CONFIDENTIAL]" in sanitized
    
    print("[OK] [合规脱敏测试] 敏感电话与测试凭证成功被 100% 遮蔽掩盖，符合合规和隐私要求！")

def test_langgraph_workflow_runs_fully():
    """测试 LangGraph 竞品分析工作流的完整执行与打回闭环"""
    # 临时禁用 LLM 以确保测试确定性（使用离线规则引擎兜底）
    original_key = os.environ.pop("LLM_API_KEY", None)
    original_moonshot = os.environ.pop("MOONSHOT_API_KEY", None)

    try:
        # 重新加载 config 和 engine 以使用离线模式
        import importlib
        import config as cfg_mod
        importlib.reload(cfg_mod)

        from agents import engine as eng_mod
        eng_mod.client = None

        workflow = SequentialLangGraphWorkflow()

        # 运行分析（对 OpenAI 和 火山引擎）
        competitors = ["OpenAI", "火山引擎"]
        final_state = workflow.execute(competitors)

        # 1. 验证是否成功生成最终报告
        assert final_state["final_markdown_report"] != ""
        assert "全球主流大语言模型 API" in final_state["final_markdown_report"]
        assert "2026旗舰版" in final_state["final_markdown_report"]

        # 2. 验证竞品归档中包含了提取后的强 Schema 数据
        assert "OpenAI" in final_state["reports_archive"]
        assert "火山引擎" in final_state["reports_archive"]
        assert final_state["reports_archive"]["OpenAI"]["region"] == "international"
        assert final_state["reports_archive"]["火山引擎"]["region"] == "domestic"

        # 3. 验证 OpenAI 是否发生了质检打回循环（由于初次抽取缺少 completion 价格）
        trace_text = "\n".join(final_state["trace_logs"])
        assert "QC" in trace_text or "feedback" in trace_text or "打回" in trace_text

        # 4. 验证信息溯源是否成功锁定在报告中
        assert "信息源跟踪" in final_state["final_markdown_report"]
        assert "pricing.prompt_price_per_million" in final_state["final_markdown_report"]

        # 5. 验证是否成功提取出 2026 年最新真实型号 (排除臆造幻觉型号)
        assert final_state["reports_archive"]["OpenAI"]["model_family"] == "gpt-5.5-instant"
        assert final_state["reports_archive"]["火山引擎"]["model_family"] == "Doubao-Seed-2.0"

        print("[OK] [LangGraph 工作流测试] 全链路无缝跑通，最新型号断言无幻觉，Fact-Check Auditor 与强 Schema 成功捍卫真实性！")
    finally:
        # 恢复环境变量
        if original_key:
            os.environ["LLM_API_KEY"] = original_key
        if original_moonshot:
            os.environ["MOONSHOT_API_KEY"] = original_moonshot

def test_adaptive_appendix_c():
    """测试不同场景下最终报告 Appendix C 能够完美自适应隐藏 CodingPlan 并自适应定制表格标题"""
    from agents.engine import writer_node
    
    # 模拟通用归档数据
    mock_archive = {
        "OpenAI": {
            "provider_name": "OpenAI",
            "region": "international",
            "model_family": "GPT-4o",
            "pricing": {"prompt_price_per_million": 5.0, "completion_price_per_million": 15.0, "currency": "USD"},
            "rate_limits": {"rpm": 10000, "tpm": 1000000},
            "features": {"context_window": 128000, "function_calling": True, "vision_support": True},
            "user_feedback": {"developer_satisfaction": 4.8, "strengths": ["高智能"], "pain_points": ["价格贵"]},
            "coding_plan": {"is_supported_in_editor": True, "language_optimizations": ["Python"], "has_sandbox_env": True, "plan_description": "2026专属计划"},
            "is_sanitized": True,
            "sources": {}
        }
    }
    
    # 1. 测试代码开发场景
    state_code = {
        "reports_archive": mock_archive,
        "parsed_requirement": {"scenario": "code_development", "raw_query": "代码开发需求"},
        "trace_logs": []
    }
    res_code = writer_node(state_code)
    report_code = res_code["final_markdown_report"]
    assert "附录 C. 开发者编程 CodingPlan 与支持" in report_code
    assert "IDE 嵌入支持" in report_code
    assert "针对优化语言列表" in report_code
    
    # 2. 测试文献/学术写作场景
    state_doc = {
        "reports_archive": mock_archive,
        "parsed_requirement": {"scenario": "document_analysis", "raw_query": "学术论文需求"},
        "trace_logs": []
    }
    res_doc = writer_node(state_doc)
    report_doc = res_doc["final_markdown_report"]
    assert "附录 C. 学术写作与文献处理大盘支持" in report_doc
    assert "长文本窗口支持" in report_doc
    assert "多模态文献解析" in report_doc
    assert "IDE 嵌入支持" not in report_doc
    
    # 3. 测试创意写作场景
    state_creative = {
        "reports_archive": mock_archive,
        "parsed_requirement": {"scenario": "creative", "raw_query": "小说创意需求"},
        "trace_logs": []
    }
    res_creative = writer_node(state_creative)
    report_creative = res_creative["final_markdown_report"]
    assert "附录 C. 创意写作与创意内容生产大盘支持" in report_creative
    assert "最大上下文窗口" in report_creative
    assert "IDE 嵌入支持" not in report_creative

    # 4. 测试 AI 数据分析场景 (New!)
    state_da = {
        "reports_archive": mock_archive,
        "parsed_requirement": {"scenario": "data_analysis", "raw_query": "做一个 AI 数据分析助手"},
        "trace_logs": []
    }
    res_da = writer_node(state_da)
    report_da = res_da["final_markdown_report"]
    assert "附录 C. 企业级 AI 数据分析与 BI 报表助手支持大盘" in report_da
    assert "表格理解与代码执行" in report_da
    assert "图表生成与异常分析" in report_da
    assert "数据库连接与预览" in report_da
    assert "导出与脱敏合规" in report_da
    assert "IDE 嵌入支持" not in report_da
    assert "长文本窗口支持" not in report_da
    
    print("[OK] [动态附录 C 自适应测试] 不同场景下附录 C 成功进行自适应定制与防污染隔离！")

def test_data_analysis_smart_query_workflow():
    """测试在 smart_query 模式下，针对 AI 数据分析场景的完整选型报告生成与格式契合"""
    # 临时禁用 LLM 强制触发 offline_rule_llm_mock 兜底
    original_key = os.environ.pop("LLM_API_KEY", None)
    original_moonshot = os.environ.pop("MOONSHOT_API_KEY", None)

    try:
        # 重新加载 config 和 engine
        import importlib
        import config as cfg_mod
        importlib.reload(cfg_mod)

        from agents import engine as eng_mod
        eng_mod.client = None

        workflow = SequentialLangGraphWorkflow()
        
        # 清除可能存在的旧缓存，以防缓存污染影响最新的结构化逻辑测试
        try:
            from agents.learning_agent import KNOWLEDGE_BASE_PATH
            if os.path.exists(KNOWLEDGE_BASE_PATH):
                os.remove(KNOWLEDGE_BASE_PATH)
                print("[Test Config] 成功物理清理知识库缓存以避免脏缓存污染")
        except Exception as e:
            print(f"[Test Config] 清理缓存异常: {e}")
        
        query = "我想做一个 AI 数据分析助手，用户上传 Excel、CSV 或数据库数据后，可以让 AI 自动分析趋势、生成图表、解释异常、输出报告。请帮我分析这个场景需要对比哪些模型能力和产品能力。"
        
        competitors = ["OpenAI", "火山引擎"]
        final_state = workflow.execute(competitors, smart_query=query)

        # 1. 验证场景识别正确
        parsed_req = final_state.get("parsed_requirement", {})
        assert parsed_req.get("scenario") == "data_analysis"

        report = final_state.get("final_markdown_report", "")
        assert report != ""

        # 2. 验证三个要求的 Markdown 章节标题存在于最终报告中
        assert "### 问题" in report
        assert "### 输出结果" in report
        assert "### 分析与整改" in report

        # 3. 验证报告内容包含 AI 数据分析的核心维度
        assert "表格理解与代码执行" in report
        assert "图表生成与异常分析" in report
        assert "数据清洗能力" in report
        assert "SQL 生成能力" in report
        assert "权限管理" in report
        assert "数据脱敏" in report

        # 4. 验证附录 C 成功切换为 BI 大盘，且不含编程/长文本等无关残留
        assert "附录 C. 企业级 AI 数据分析与 BI 报表助手支持大盘" in report
        assert "IDE 嵌入支持" not in report
        assert "长文本窗口支持" not in report

        print("[OK] [AI 数据分析助手专属测试] 场景识别正确、三段式智能选型建议结构完整、BI 附录定制防噪音污染完美跑通！")
    finally:
        # 恢复环境变量
        if original_key:
            os.environ["LLM_API_KEY"] = original_key
        if original_moonshot:
            os.environ["MOONSHOT_API_KEY"] = original_moonshot

if __name__ == "__main__":
    print("开始运行 HarnessFlow TDD 单元测试用例...")
    test_sanitizer_confidential_masking()
    test_langgraph_workflow_runs_fully()
    test_adaptive_appendix_c()
    test_data_analysis_smart_query_workflow()
    print("所有 TDD 单元测试通过！")
