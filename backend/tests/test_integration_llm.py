"""End-to-end integration test: verify all agents call LLM"""
import os, sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

os.environ['LLM_API_KEY'] = 'sk-kcYuRnXh29u0v4J5SykodmJsTv3hGktefMpIdFju9zMmeWM2'
os.environ['LLM_BASE_URL'] = 'https://api.moonshot.cn/v1'
os.environ['LLM_MODEL'] = 'kimi-k2.5'
os.environ['LLM_TEMPERATURE'] = '1'

import importlib
import config as cfg
importlib.reload(cfg)

from agents import engine as eng
importlib.reload(eng)

from agents.engine import SequentialLangGraphWorkflow

wf = SequentialLangGraphWorkflow()
result = wf.execute([], smart_query="我需要一个适合代码开发的低成本模型，支持函数调用，预算每百万Token不超过10元")

print("\n" + "="*60)
print("INTEGRATION VERIFICATION")
print("="*60)

trace = "\n".join(result.get("trace_logs", []))
report = result.get("final_markdown_report", "")

checks = {
    "RequirementParser (LLM)": "code_development" in str(result.get("parsed_requirement", {})),
    "CompetitorDiscovery (LLM)": "竞品发现专家" in trace,
    "Planner (LLM)": "PlannerAgent" in trace,
    "ScoringAgent (LLM)": "ScoringAgent" in trace,
    "ScenarioAnalyst (LLM)": "场景分析" in trace,
    "Writer (LLM)": "核心洞察" in report,
    "FreshnessAuditor": "FreshnessAuditor" in trace,
    "EvidenceRetriever": "EvidenceRetriever" in trace,
}

for name, passed in checks.items():
    status = "OK" if passed else "MISS"
    print(f"  [{status}] {name}")

print(f"\nReport length: {len(report)} chars")
scenario = result.get("scenario_analysis", {})
print(f"Scenario: {scenario.get('scenario_name', 'N/A')}")
print(f"LLM Analysis: {len(scenario.get('llm_analysis', ''))} chars")
scoring = result.get("scoring_results", {})
print(f"Top: {scoring.get('top_recommendation', 'N/A')}")
print("="*60)
