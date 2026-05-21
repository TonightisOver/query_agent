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

if __name__ == "__main__":
    print("开始运行 HarnessFlow TDD 单元测试用例...")
    test_sanitizer_confidential_masking()
    test_langgraph_workflow_runs_fully()
    print("所有 TDD 单元测试通过！")
