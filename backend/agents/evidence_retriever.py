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
        return {"rag_context": "", "has_matched_evidence": False, "trace_logs": trace_logs}
    
    try:
        # 获取场景以防跨场景 RAG 污染
        parsed_req = state.get("parsed_requirement", {})
        scenario = parsed_req.get("scenario", "general") if parsed_req else "general"
        
        # 构建检索查询
        search_query = f"{current} 模型 定价 能力 特点"
        results = store.search(search_query, top_k=3, target_scenario=scenario)
        
        has_matched_evidence = False
        if results:
            rag_pieces = []
            for r in results:
                score = r.get('score', 0.0)
                if score < 0.20:
                    continue
                
                # RAG strict check & scenario metadata validation
                meta = r.get('metadata', {})
                doc_scenario = meta.get('scenario')
                
                # 1. 如果包含明确的场景元数据且不匹配，直接丢弃
                if doc_scenario and doc_scenario != scenario:
                    continue
                
                # 2. 如果不包含场景元数据且当前目标场景是数据分析，应用白名单与禁用过滤
                text_lower = r.get('text', '').lower()
                if not doc_scenario and scenario == "data_analysis":
                    # 白名单词汇：必须匹配至少一个
                    whites = ["table understanding", "excel", "csv", "sql", "python", "chart", "anomaly", "report", "data cleaning", "pandas", "dataframe", "数据分析", "报表", "表格", "清洗", "可视化"]
                    # 禁用噪点词汇：只要包含一个就拒绝
                    blacks = ["academic writing", "literature review", "citation management", "ide assistant", "ide", "codingplan", "mistral news", "codestral", "学术写作", "文献"]
                    
                    # 检查白名单匹配
                    has_white = any(w in text_lower for w in whites)
                    # 检查禁用词匹配
                    has_black = any(b in text_lower for b in blacks)
                    
                    if not has_white or has_black:
                        # 强不匹配，予以丢弃
                        continue
                
                # 3. 如果当前是数据分析场景，即使其他地方没有冲突，只要文本包含明显禁用噪点词，也同样丢弃以防污染
                if scenario == "data_analysis":
                    blacks = ["academic writing", "literature review", "citation management", "ide assistant", "ide", "codingplan", "mistral news", "codestral", "学术写作", "文献"]
                    if any(b in text_lower for b in blacks):
                        continue
                
                rag_pieces.append(f"[相关度: {score:.2f}] {r['text'][:300]}")
                
            if rag_pieces:
                rag_context = "\n---\n".join(rag_pieces)
                has_matched_evidence = True
                trace_msg = f"[EvidenceRetriever Agent] 为 [{current}] 检索到 {len(rag_pieces)} 个高置信度且场景高度匹配的文档片段。"
            else:
                trace_msg = f"[EvidenceRetriever Agent] 为 [{current}] 检索到的文档片段相关度过低或由于跨场景/噪点污染被全部过滤。"
        else:
            trace_msg = f"[EvidenceRetriever Agent] 未找到与 [{current}] 相关的文档片段。"
        
        print(trace_msg)
        trace_logs.append(trace_msg)
        
    except Exception as e:
        trace_msg = f"[EvidenceRetriever Agent] 检索异常: {e}"
        print(trace_msg)
        trace_logs.append(trace_msg)
    
    return {"rag_context": rag_context, "has_matched_evidence": has_matched_evidence, "trace_logs": trace_logs}
