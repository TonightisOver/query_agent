"""
RAG 索引管理模块 (Phase 2)
轻量级内存 TF-IDF 向量存储，支持文档上传、切分、索引和检索。
当 scikit-learn 不可用时自动降级为关键词匹配。
"""
import os
import re
import json
import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter

# 尝试导入 sklearn，不可用时降级
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SimpleVectorStore:
    """基于 TF-IDF + 余弦相似度的轻量级内存向量存储"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self._tfidf_matrix = None
        self._vectorizer = None
        self._dirty = True  # 标记是否需要重建索引
    
    def add_documents(self, chunks: List[str], metadata_list: List[Dict] = None) -> int:
        """添加文档片段到索引"""
        if not chunks:
            return 0
        start_idx = len(self.documents)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                self.documents.append(chunk.strip())
                meta = (metadata_list[i] if metadata_list and i < len(metadata_list) 
                        else {"chunk_id": start_idx + i})
                meta["indexed_at"] = datetime.utcnow().isoformat()
                self.metadata.append(meta)
        self._dirty = True
        count = len(self.documents) - start_idx
        print(f"[RAGIndexer] 成功索引 {count} 个文档片段，总计 {len(self.documents)} 个片段。")
        return count
    
    def _rebuild_index(self):
        """重建 TF-IDF 索引"""
        if not self._dirty or not self.documents:
            return
        if HAS_SKLEARN:
            self._vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
            self._tfidf_matrix = self._vectorizer.fit_transform(self.documents)
        self._dirty = False
    
    def search(self, query: str, top_k: int = 5, target_scenario: Optional[str] = None) -> List[Dict]:
        """检索最相关的文档片段，支持按 scenario 进行强隔离过滤"""
        if not self.documents:
            return []
        
        if HAS_SKLEARN:
            self._rebuild_index()
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            top_indices = scores.argsort()[::-1]
            results = []
            for idx in top_indices:
                if scores[idx] > 0.01:  # 最低相关度阈值
                    meta = self.metadata[idx]
                    doc_scenario = meta.get("scenario")
                    # RAG 检索污染强隔离：如果分块明确指明了场景，且与当前目标场景不符，则直接隔离过滤
                    if doc_scenario and target_scenario and doc_scenario != target_scenario:
                        continue
                    results.append({
                        "text": self.documents[idx],
                        "score": float(scores[idx]),
                        "metadata": meta
                    })
                    if len(results) >= top_k:
                        break
            return results
        else:
            # 降级：关键词匹配
            return self._keyword_search(query, top_k, target_scenario)
    
    def _keyword_search(self, query: str, top_k: int, target_scenario: Optional[str] = None) -> List[Dict]:
        """关键词匹配降级搜索，支持按 scenario 进行强隔离过滤"""
        query_words = set(re.findall(r'[\w]+', query.lower()))
        if not query_words:
            return []
        scored = []
        for i, doc in enumerate(self.documents):
            meta = self.metadata[i]
            doc_scenario = meta.get("scenario")
            if doc_scenario and target_scenario and doc_scenario != target_scenario:
                continue
            doc_words = set(re.findall(r'[\w]+', doc.lower()))
            if not doc_words:
                continue
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:top_k]:
            results.append({
                "text": self.documents[idx],
                "score": score,
                "metadata": self.metadata[idx]
            })
        return results
    
    def get_stats(self) -> Dict:
        """返回索引统计信息"""
        return {
            "total_chunks": len(self.documents),
            "has_sklearn": HAS_SKLEARN,
            "index_built": not self._dirty
        }
    
    def clear(self):
        """清空索引"""
        self.documents.clear()
        self.metadata.clear()
        self._tfidf_matrix = None
        self._vectorizer = None
        self._dirty = True

    def save_index(self, file_path: str = None):
        """保存索引数据到磁盘 JSON 文件"""
        if file_path is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(backend_dir, "data", "rag_index.json")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            "documents": self.documents,
            "metadata": self.metadata
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[RAGIndexer] 成功持久化保存索引至 {file_path} (共 {len(self.documents)} 个片段)")
        except Exception as e:
            print(f"[RAGIndexer] 保存持久化索引失败: {e}")

    def load_index(self, file_path: str = None) -> bool:
        """从磁盘 JSON 文件加载索引数据"""
        if file_path is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(backend_dir, "data", "rag_index.json")
            
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.metadata = data.get("metadata", [])
            self._dirty = True
            print(f"[RAGIndexer] 成功从磁盘加载 RAG 持久化索引，共 {len(self.documents)} 个片段")
            return True
        except Exception as e:
            print(f"[RAGIndexer] 加载持久化索引失败: {e}")
            return False


# 全局单例向量存储
_global_vector_store = SimpleVectorStore()

def get_vector_store() -> SimpleVectorStore:
    """获取全局向量存储实例"""
    return _global_vector_store


def chunk_text(text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """将文本切分为重叠的片段"""
    if not text or not text.strip():
        return []
    
    # 先按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            # 对超长段落按固定大小+重叠切分
            start = 0
            while start < len(para):
                end = min(start + max_chunk_size, len(para))
                chunks.append(para[start:end])
                start = end - overlap
                if start + overlap >= len(para):
                    break
    
    return [c for c in chunks if c.strip()]


def parse_document(file_path: str) -> str:
    """解析文档文件为纯文本 (支持 .txt, .md, .pdf)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ('.txt', '.md', '.markdown'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.pdf':
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            text_parts = []
            for match in re.finditer(rb'\(([^)]+)\)', content):
                try:
                    text_parts.append(match.group(1).decode('utf-8', errors='ignore'))
                except:
                    pass
            if text_parts:
                return ' '.join(text_parts)
            return f"[PDF 文件需要专业解析器: {os.path.basename(file_path)}]"
        except Exception as e:
            return f"[PDF 解析失败: {e}]"
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return f"[不支持的文件格式: {ext}]"


def index_document(file_path: str, doc_id: str = None) -> Dict:
    """完整的文档索引流程: 解析 → 切分 → 索引化"""
    if doc_id is None:
        doc_id = os.path.basename(file_path)
    
    print(f"[RAGIndexer] 开始索引文档: {doc_id}")
    
    text = parse_document(file_path)
    chunks = chunk_text(text)
    metadata_list = [
        {"doc_id": doc_id, "chunk_id": i, "source": file_path}
        for i in range(len(chunks))
    ]
    
    store = get_vector_store()
    count = store.add_documents(chunks, metadata_list)
    
    result = {
        "status": "ok",
        "doc_id": doc_id,
        "chunks_count": count,
        "total_chars": len(text),
        "store_stats": store.get_stats()
    }
    
    print(f"[RAGIndexer] 文档 {doc_id} 索引完成: {count} 个片段")
    return result


def index_text_content(text: str, doc_id: str = "inline") -> Dict:
    """直接索引文本内容（不从文件读取）"""
    chunks = chunk_text(text)
    metadata_list = [
        {"doc_id": doc_id, "chunk_id": i, "source": "inline_text"}
        for i in range(len(chunks))
    ]
    store = get_vector_store()
    count = store.add_documents(chunks, metadata_list)
    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunks_count": count
    }


def initialize_rag_store():
    """初始化 RAG 外部资源库与用户历史记录 RAG 索引"""
    store = get_vector_store()
    
    # 1. 尝试从磁盘加载持久化索引
    if store.load_index():
        store._rebuild_index()
        return
        
    print("[RAGIndexer] 未检测到持久化索引，启动冷启动预加载与全量构建流程...")
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    external_dir = os.path.join(backend_dir, "data", "external_resources")
    kb_path = os.path.join(backend_dir, "data", "knowledge_base.json")
    
    # 2. 扫描并索引外部资源库文件
    if os.path.exists(external_dir):
        files_indexed = 0
        for fname in os.listdir(external_dir):
            if fname.endswith(('.md', '.markdown', '.txt')):
                fpath = os.path.join(external_dir, fname)
                try:
                    index_document(fpath, doc_id=fname)
                    files_indexed += 1
                except Exception as e:
                    print(f"[RAGIndexer] 索引外部资源 {fname} 失败: {e}")
        print(f"[RAGIndexer] 冷启动成功导入 {files_indexed} 个外部资源文档。")
        
    # 3. 扫描并导入用户的历史分析报告 (knowledge_base.json)
    if os.path.exists(kb_path):
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            history_count = 0
            for key, entry in kb_data.items():
                report_text = entry.get("final_report", "")
                scenario = entry.get("scenario", "general")
                if report_text:
                    chunks = chunk_text(report_text)
                    metadata_list = [
                        {
                            "doc_id": f"history_report_{key}",
                            "chunk_id": i,
                            "source": "history_report",
                            "cache_key": key,
                            "scenario": scenario
                        }
                        for i in range(len(chunks))
                    ]
                    store.add_documents(chunks, metadata_list)
                    history_count += 1
            print(f"[RAGIndexer] 冷启动成功从用户历史数据导入 {history_count} 个历史分析报告。")
        except Exception as e:
            print(f"[RAGIndexer] 导入历史数据 RAG 索引失败: {e}")
            
    # 4. 构建索引并保存到磁盘
    store._rebuild_index()
    store.save_index()

