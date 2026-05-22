import os
import sys
import queue
import threading
import json
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from schemas import SmartAnalyzeRequest, FeedbackRequest

# 确保 backend 路径在 sys.path 中，便于直接运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.engine import SequentialLangGraphWorkflow

app = FastAPI(
    title="HarnessFlow LLM API Competitor Analysis System",
    description="字节跳动集团信息系统 AI 全栈项目挑战赛 - LLM API 服务商竞品分析系统后端",
    version="1.0.0"
)

# 开启 CORS 跨域支持，确保前后端能够顺畅联调对接
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    competitors: List[str]

@app.get("/api/health")
def health_check():
    """健康检查与存活探测接口"""
    return {"status": "ok", "message": "HarnessFlow Backend is active"}

@app.post("/api/analyze")
async def analyze_competitors(request: AnalyzeRequest):
    """
    多 Agent 竞品分析核心接口。
    采用 SSE (Server-Sent Events) 流式推送 Agent DAG 协同状态及 Trace 可观测日志。
    """
    competitors = request.competitors
    if not competitors:
        raise HTTPException(status_code=400, detail="竞品监控名单不能为空")
    if len(competitors) > 16:
        raise HTTPException(status_code=400, detail="单次分析最多支持 16 家厂商")

    event_queue = queue.Queue()

    def run_workflow():
        workflow = SequentialLangGraphWorkflow()
        
        def callback(event_data):
            # 将工作流产生的节点流转状态推入共享队列
            event_queue.put(event_data)
            
        try:
            workflow.execute(competitors, event_callback=callback)
        except Exception as e:
            # 捕获突发异常并传递给前端展示，提高容错体验
            event_queue.put({
                "event": "workflow_error",
                "message": f"工作流运行异常: {str(e)}"
            })
        finally:
            # 队列放置结束符，关闭数据流
            event_queue.put(None)

    # 启动工作线程执行图计算，保障主事件循环处于非阻塞状态
    thread = threading.Thread(target=run_workflow)
    thread.start()

    async def event_generator():
        import asyncio
        loop = asyncio.get_event_loop()
        
        while True:
            # 异步非阻塞地从队列中读取事件
            event = await loop.run_in_executor(None, event_queue.get)
            if event is None:
                break
            
            # 自定义 JSON 序列化器以支持 datetime 类型
            def json_serial_default(o):
                from datetime import datetime
                if isinstance(o, datetime):
                    return o.isoformat()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            # 使用标准 SSE 协议格式进行序列化推送，支持 UTF-8 编码
            yield f"data: {json.dumps(event, ensure_ascii=False, default=json_serial_default)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==========================================
# Phase 1-4 Endpoints
# ==========================================

@app.post("/api/analyze/smart")
async def analyze_competitors_smart(request: SmartAnalyzeRequest):
    """
    智能自然语言竞品分析核心接口。
    采用 SSE (Server-Sent Events) 流式推送 Agent DAG 协同状态及 Trace 可观测日志。
    """
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="智能分析输入不能为空")

    event_queue = queue.Queue()

    def run_workflow():
        workflow = SequentialLangGraphWorkflow()
        
        def callback(event_data):
            # 将工作流产生的节点流转状态推入共享队列
            event_queue.put(event_data)
            
        try:
            workflow.execute(competitor_list=[], event_callback=callback, smart_query=query)
        except Exception as e:
            event_queue.put({
                "event": "workflow_error",
                "message": f"智能工作流运行异常: {str(e)}"
            })
        finally:
            event_queue.put(None)

    thread = threading.Thread(target=run_workflow)
    thread.start()

    async def event_generator():
        import asyncio
        loop = asyncio.get_event_loop()
        
        while True:
            event = await loop.run_in_executor(None, event_queue.get)
            if event is None:
                break
            
            def json_serial_default(o):
                from datetime import datetime
                if isinstance(o, datetime):
                    return o.isoformat()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            yield f"data: {json.dumps(event, ensure_ascii=False, default=json_serial_default)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """
    上传知识库参考文档，支持 RAG。
    """
    import shutil

    ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".csv"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    safe_filename = os.path.basename(file.filename or "upload.txt")
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, safe_filename)
    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(status_code=413, detail="文件大小超过 10MB 限制")
                buffer.write(chunk)

        from agents.rag_indexer import index_document
        res = index_document(file_path, doc_id=safe_filename)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")


@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    提交用户对分析结果的星级评分和改进建议。
    """
    try:
        from agents.feedback_agent import save_feedback
        entry = save_feedback(
            report_id=request.report_id,
            rating=request.rating,
            comments=request.comments,
            vendor_list=request.vendor_list
        )
        return {"status": "ok", "message": "反馈保存成功", "data": entry}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存反馈异常: {e}")


@app.get("/api/knowledge/history")
def get_analysis_history():
    """
    获取知识库中的历史分析大盘报告列表。
    """
    try:
        from agents.learning_agent import get_history_list
        history = get_history_list(limit=20)
        return {"status": "ok", "data": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {e}")

@app.get("/api/knowledge/history/{cache_key}")
def get_analysis_history_detail(cache_key: str):
    """
    根据缓存键获取单个历史分析报告的完整详情。
    """
    try:
        from agents.learning_agent import get_cached_detail_by_key
        detail = get_cached_detail_by_key(cache_key)
        if not detail:
            raise HTTPException(status_code=404, detail="未找到对应的历史分析报告")
        return {"status": "ok", "data": detail}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史详情失败: {e}")

if __name__ == "__main__":
    import uvicorn
    # 本地启动测试服务
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
