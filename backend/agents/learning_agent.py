"""
知识沉淀与复用 Agent (Phase 4)
缓存成功的分析结果到知识库，支持相似需求的快速复用。
"""
import os
import json
import hashlib
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 持久化路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")
_knowledge_lock = threading.Lock()


def _ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_knowledge_base() -> Dict:
    """加载知识库"""
    _ensure_data_dir()
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_knowledge_base(data: Dict):
    """保存知识库"""
    _ensure_data_dir()
    with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compute_cache_key(vendor_list: List[str], scenario: str = "") -> str:
    """计算缓存键 (基于厂商列表和场景的哈希)"""
    key_str = ",".join(sorted(vendor_list)) + "|" + scenario
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def cache_analysis_result(
    vendor_list: List[str],
    scenario: str,
    final_report: str,
    reports_archive: Dict,
    scoring_results: Dict = None
) -> str:
    """缓存分析结果到知识库"""
    cache_key = _compute_cache_key(vendor_list, scenario)
    
    with _knowledge_lock:
        kb = _load_knowledge_base()
        
        # 清理 reports_archive 中不可序列化的内容 (例如将 datetime 序列化为 string)
        clean_archive = {}
        for k, v in reports_archive.items():
            try:
                serialized = json.dumps(v, default=str)
                clean_archive[k] = json.loads(serialized)
            except (TypeError, ValueError):
                clean_archive[k] = str(v)
        
        kb[cache_key] = {
            "vendor_list": vendor_list,
            "scenario": scenario,
            "final_report": final_report[:10000],  # 限制缓存大小
            "reports_archive": clean_archive,
            "scoring_results": scoring_results,
            "cached_at": datetime.utcnow().isoformat(),
            "hit_count": 0
        }
        
        _save_knowledge_base(kb)
        
        # 动态增量将最新分析报告切分并索引入 RAG 向量数据库并持久化
        try:
            from agents.rag_indexer import get_vector_store, chunk_text
            store = get_vector_store()
            chunks = chunk_text(final_report)
            metadata_list = [
                {
                    "doc_id": f"history_report_{cache_key}",
                    "chunk_id": i,
                    "source": "history_report",
                    "cache_key": cache_key,
                    "scenario": scenario
                }
                for i in range(len(chunks))
            ]
            store.add_documents(chunks, metadata_list)
            store._rebuild_index()
            store.save_index()
            print(f"[LearningAgent] RAG 动态增量索引成功: 报告 {cache_key} 已切分为 {len(chunks)} 个片段并持久化到磁盘。")
        except Exception as e:
            print(f"[LearningAgent] RAG 动态增量索引失败: {e}")
    
    print(f"[LearningAgent] 分析结果已缓存: key={cache_key}, 厂商={vendor_list}")
    return cache_key


def lookup_cached_result(
    vendor_list: List[str],
    scenario: str = "",
    max_age_hours: int = 24
) -> Optional[Dict]:
    """查找缓存的分析结果"""
    cache_key = _compute_cache_key(vendor_list, scenario)
    
    kb = _load_knowledge_base()
    entry = kb.get(cache_key)
    
    if not entry:
        return None
    
    # 检查时效性
    try:
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.utcnow() - cached_at > timedelta(hours=max_age_hours):
            print(f"[LearningAgent] 缓存已过期: key={cache_key}")
            return None
    except (ValueError, KeyError):
        return None
    
    # 更新命中计数
    with _knowledge_lock:
        kb = _load_knowledge_base()
        if cache_key in kb:
            kb[cache_key]["hit_count"] = kb[cache_key].get("hit_count", 0) + 1
            _save_knowledge_base(kb)
    
    print(f"[LearningAgent] 缓存命中: key={cache_key}, 命中次数={entry.get('hit_count', 0) + 1}")
    return entry


def get_cached_detail_by_key(cache_key: str) -> Optional[Dict]:
    """通过缓存键获取缓存的分析详情"""
    kb = _load_knowledge_base()
    return kb.get(cache_key)


def get_history_list(limit: int = 20) -> List[Dict]:
    """获取历史分析列表"""
    kb = _load_knowledge_base()
    
    entries = []
    for key, entry in kb.items():
        entries.append({
            "cache_key": key,
            "vendor_list": entry.get("vendor_list", []),
            "scenario": entry.get("scenario", ""),
            "cached_at": entry.get("cached_at", ""),
            "hit_count": entry.get("hit_count", 0)
        })
    
    # 按缓存时间降序排列
    entries.sort(key=lambda x: x.get("cached_at", ""), reverse=True)
    return entries[:limit]


def learning_agent_node(state: dict) -> dict:
    """学习 Agent 节点 — 在分析完成后缓存结果到知识库"""
    trace_logs = list(state.get("trace_logs", []))
    
    try:
        vendor_list = state.get("competitor_list", [])
        scenario = ""
        parsed_req = state.get("parsed_requirement", {})
        if parsed_req:
            scenario = parsed_req.get("scenario", "")
        
        final_report = state.get("final_markdown_report", "")
        reports_archive = state.get("reports_archive", {})
        scoring_results = state.get("scoring_results", {})
        
        if final_report and reports_archive:
            cache_key = cache_analysis_result(
                vendor_list, scenario, final_report, reports_archive, scoring_results
            )
            trace_msg = f"[LearningAgent] 分析结果已沉淀到知识库 (key: {cache_key})"
            cache_key_val = cache_key
        else:
            trace_msg = "[LearningAgent] 无有效分析结果可缓存，跳过知识沉淀。"
            cache_key_val = None
        
        print(trace_msg)
        trace_logs.append(trace_msg)
        
    except Exception as e:
        trace_msg = f"[LearningAgent] 知识沉淀异常: {e}"
        print(trace_msg)
        trace_logs.append(trace_msg)
        cache_key_val = None
    
    return {"trace_logs": trace_logs, "cache_key": cache_key_val}

