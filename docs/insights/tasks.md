# WBS 任务拆解与开发计划 (Tasks Plan)

本项目严格遵循测试驱动开发（TDD）及 HarnessFlow 强质量纪律。项目在编码前将整体设计与计划锁定，并由 Gate 1 人工门禁予以审核批准。

---

## 1. 任务开发看板 (WBS Tasks)

### 📌 [阶段 1] 环境与骨架搭设
- [/] **TASK-001**: **Docker 联合编排环境与多目录骨架搭设** -> 负责人: Subagent
  *   *目标*：创建 `backend/`、`frontend/`、`mock_server/` 目录；编写初始的 `docker-compose.yml` 及基础 Dockerfile。
  *   *DoD*：在本地执行 `docker-compose build` 能够成功构建无错骨架。

### 📌 [阶段 2] 数据源与 Schema 锁定
- [ ] **TASK-002**: **竞品知识 Schema 强契约声明与 Mock 采集数据服务实现** -> 负责人: Subagent
  *   *目标*：在 backend 中利用 Pydantic/JSON 锁定 Schema；在 mock_server 中编写 Python Flask/FastAPI 极简服务，返回 OpenAI, Anthropic, 豆包等模型价格的模拟 HTML 内容（包含干扰的隐私字段以测试脱敏 Agent）。
  *   *DoD*：单元测试能够成功解析 Schema；Mock 服务能够通过 HTTP API 访问，并返回带有干扰数据的文档。

### 📌 [阶段 3] 核心多 Agent 逻辑 (TDD 实现)
- [ ] **TASK-003**: **基于 LangGraph 的多 Agent（采集、脱敏、分析、质检、撰写）协同与打回闭环开发** -> 负责人: Principal Agent & Subagent
  *   *目标*：编写 TDD 单元测试验证各 Agent 的核心逻辑；使用 LangGraph 串联 5 个 Agent，特别实现**“质检不合规打回分析”**的条件路由逻辑。
  *   *DoD*：单元测试通过：
      1.  验证脱敏 Agent 能够 100% 遮蔽敏感数据。
      2.  验证质检不通过时，图节点能够自动折返打回。
      3.  验证通过质检的数据严格符合 JSON Schema。

### 📌 [阶段 4] 后端集成与接口暴露
- [ ] **TASK-004**: **FastAPI 后端路由封装与实时 Trace 推送** -> 负责人: Subagent
  *   *目标*：编写 FastAPI 服务路由，封装 LangGraph 执行调用，支持以 WebSockets 或 SSE (Server-Sent Events) 实时推送 Agent DAG 流转状态及 Trace 日志。
  *   *DoD*：使用测试脚本调用 FastAPI API，能成功流式获取 Agent DAG 执行日志，生成最终的 Markdown 对比报告。

### 📌 [阶段 5] 前端可视化 (可视化 Trace 与溯源)
- [ ] **TASK-005**: **React 可视化前端开发（DAG 状态展示、溯源引用与报告看板）** -> 负责人: Principal Agent
  *   *目标*：开发响应式 React 应用，包含三大面板：配置与启动、多 Agent 协同实时 DAG 动效状态树、竞品分析报告展示。特别实现**“点击对比数据弹窗查阅原始网页溯源”**的功能。
  *   *DoD*：前端页面无 JS 错误，界面流畅，DAG 状态与打回高亮能够与后端流式消息实时同步。

### 📌 [阶段 6] 联调、Docker 整合与交付
- [ ] **TASK-006**: **全链路一键 Docker 部署联调与交付报告归档** -> 负责人: Principal Agent
  *   *目标*：整合全部 Dockerfile，编写一键拉起脚本；在本地进行全链路的真采与 Mock 测试。
  *   *DoD*：执行一键启动命令后，系统能够完全在容器内跑通，React 前端展现完整的竞品分析和溯源，无任何死锁或幻觉暴露。

---

## 2. 门禁解锁状态

*   **GATE 1 (设计与计划审批门禁)**: [x] **已通过** (架构师用户于 2026-05-20 批准，开始进入 TDD 开发周期)
*   **GATE 2 (发布与交付审批门禁)**: 🔴 **未通过** (等待所有开发测试及评审通过后，由用户输入“批准发布”解锁)
