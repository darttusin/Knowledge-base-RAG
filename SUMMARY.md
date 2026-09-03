# Knowledge Base RAG — фактическая сводка

## Назначение

Исследовательская система вопросов и ответов по базе знаний. В текущем demo
используются документация PyTorch и Stack Overflow: пользователь задаёт вопрос в
веб-чате, backend извлекает контекст из ChromaDB, обращается к внешнему
OpenAI-compatible LLM и сохраняет диалог в PostgreSQL.

Проект демонстрирует полный RAG/ML workflow, но не является готовой production
платформой и не гарантирует, что данные остаются внутри закрытого контура.

## Runtime

```text
Next.js :3000
  └─ FastAPI :8001
       ├─ PostgreSQL :5432 — auth, dialogues, messages, folders, sources, FTS
       ├─ ChromaDB + embeddings + reranker
       ├─ внешний OpenAI-compatible generator
       ├─ off-topic classifier
       └─ code-executor :8002
```

Реальный retrieval — dense, dense + CrossEncoder rerank или query transform
(rewrite + HyDE + rerank). PostgreSQL FTS обслуживает поиск по загруженным
sources. BM25/RRF/FAISS retrieval, Prometheus/Grafana/Jaeger и общий backend
`/health` отсутствуют. Telegram bot и parser — legacy/experimental и не входят в
Compose.

## Что есть в интерфейсе и API

- JWT login существующего пользователя; публичной регистрации нет.
- CRUD диалогов, streaming через SSE, feedback и история.
- Upload/list/search/read/download/move/delete источников и дерево папок.
- Просмотр источников/citations и запуск Python blocks через отдельный executor.
- Live API schema: `/api/docs`; `backend/openapi.json` и `.yaml` синхронизированы
  с `app.openapi()` на 2026-09-03, но неверно моделируют media types SSE/download.

## Offline ML-контур

- `dataset-prep`: Stack Overflow + retrieved context в legacy SFT format.
- `dataset-synth`: teacher-generated/adversarial examples; groundedness не проверяется.
- `prompt-contract`: общий prompt format и fingerprint.
- `lora-train`: LoRA/QLoRA adapter на Linux/CUDA.
- `lora-pipeline`: ingest/synth переиспользуют artifacts; training запускается заново.
- `eval-runner`: lexical/semantic/RAGAS metrics и обязательное W&B logging.

## Основные ограничения

- Внешний LLM не входит в Compose; без него чат не работает.
- Shared Chroma retrieval не фильтруется по пользователю. Upload в уже непустую
  collection может остаться только в PostgreSQL, потому что indexing пропускается.
- Выбор папок в UI пока не ограничивает RAG retrieval end-to-end.
- DB и Chroma не образуют общую транзакцию; upload/delete/move могут расходиться.
- PDF/DOCX перечислены API, но backend фактически принимает UTF-8 text.
- Code executor имеет timeout/RestrictedPython, однако разрешённые scientific API,
  сеть и writable mount не дают strong isolation.
- JWT хранится в browser localStorage; soft delete не отзывает выданный token.
- Docker backend build context включает весь repo и потенциальные env/artifacts;
  full image build пока не считается green.
- Pipeline/eval могут сохранить API keys в manifest, W&B config или subprocess
  arguments.

## Состояние качества

На аудите 2026-09-03 lock и структура Compose валидировались, но whole-repo
baseline не green: Ruff, formatting и Pyright имеют накопленные ошибки; backend
tests содержат stale imports/mocks и смешивают PostgreSQL FTS с SQLite; executor
manifest не перечисляет все test/runtime scientific dependencies. CI, Alembic и
frontend test runner отсутствуют.

Поэтому изменения проверяются адресно, а data/index/model artifacts нельзя
перегенерировать как побочный эффект. Bundled corpus и Chroma фактически являются
обычными tracked Git blobs, несмотря на нерекурсивные LFS-правила.

## Навигация

- [README.md](README.md) — setup, запуск, env names, проверки и предупреждения.
- [AGENTS.md](AGENTS.md) — правила разработки и карта рисков.
- [backend/AGENTS.md](backend/AGENTS.md), [rag/AGENTS.md](rag/AGENTS.md),
  [frontend/AGENTS.md](frontend/AGENTS.md) — component-specific contracts.
- [backend/README_RAG_INTEGRATION.md](backend/README_RAG_INTEGRATION.md) и
  [frontend/FOLDER_SYSTEM.md](frontend/FOLDER_SYSTEM.md) — текущая интеграция и
  document/folder behavior.
- [lora-pipeline/README.md](lora-pipeline/README.md) — docs-to-adapter flow.
- [eval-runner/README.md](eval-runner/README.md) — eval presets; defaults
  перепроверяются по [config.py](eval-runner/eval_runner/config.py).
