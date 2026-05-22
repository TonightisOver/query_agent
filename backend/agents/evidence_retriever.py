"""
证据检索 Agent (Phase 2)
从 RAG 向量存储中检索相关文档片段，增强分析上下文。
"""
from typing import Dict, List, Any
from agents.rag_indexer import get_vector_store


def evidence_retriever_node(state: dict) -> dict:
    """证据检索节点 — 从向量库检索与当前竞品相关的文档片段"""
    current = state.get("current_competitor", "")
    query = state.get("sanitized_data", "") or current
    trace_logs = list(state.get("trace_logs", []))
    
    store = get_vector_store()
    stats = store.get_stats()
    
    rag_context = ""
    
    if stats["total_chunks"] == 0:
        trace_msg = f"[EvidenceRetriever Agent] 向量库为空，跳过 RAG 检索。"
        print(trace_msg)
        trace_logs.append(trace_msg)
        return {"rag_context": "", "trace_logs": trace_logs}
    
    try:
        # 获取场景以防跨场景 RAG 污染
        parsed_req = state.get("parsed_requirement", {})
        scenario = parsed_req.get("scenario", "general") if parsed_req else "general"
        
        # 构建检索查询
        search_query = f"{current} 模型 定价 能力 特点"
        results = store.search(search_query, top_k=3, target_scenario=scenario)
        
        if results:
            rag_pieces = []
            for r in results:
                if r['score'] >= 0.20:
                    rag_pieces.append(f"[相关度: {r['score']:.2f}] {r['text'][:300]}")
            if rag_pieces:
                rag_context = "\n---\n".join(rag_pieces)
                trace_msg = f"[EvidenceRetriever Agent] 为 [{current}] 检索到 {len(rag_pieces)} 个相关度 >= 0.20 的文档片段。"
            else:
                trace_msg = f"[EvidenceRetriever Agent] 为 [{current}] 检索到的文档片段相关度均低于 0.20，已过滤以防噪声污染。"
        else:
            trace_msg = f"[EvidenceRetriever Agent] 未找到与 [{current}] 相关的文档片段。"
        
        print(trace_msg)
        trace_logs.append(trace_msg)
        
    except Exception as e:
        trace_msg = f"[EvidenceRetriever Agent] 检索异常: {e}"
        print(trace_msg)
        trace_logs.append(trace_msg)
    
    return {"rag_context": rag_context, "trace_logs": trace_logs}
