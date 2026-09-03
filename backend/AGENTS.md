# Backend agent guide

Наследует корневой `AGENTS.md`. Перед изменениями backend прочитайте также связанные
`rag/AGENTS.md` и frontend contracts, если меняются retrieval или API.

## Архитектура и запуск

- `app.py`: FastAPI, CORS, routers, lifespan.
- `api/<domain>/router.py`: HTTP/DI; `controller.py`: orchestration;
  `models.py`/`schema.py`: Pydantic contracts.
- `db.py`: async SQLAlchemy engine/ORM/startup DDL; `auth.py`: JWT/bcrypt.
- `services/rag_service.py` и `title_service.py`: adapters.

Lifespan сначала проверяет JWT secret и инициализирует DB, затем синхронно загружает
RAG models/Chroma/classifier. RAG startup error логируется, API остаётся жив, а
message endpoints дают 503. При shutdown закрывается engine и RAG singleton.

Settings читают `.env` и относительные paths из текущего cwd. При запуске из
`backend/` default Chroma path и Docker executor hostname могут быть неверны для
host; задавайте временные overrides явно, не меняя tracked env.

## API и ownership

Routes: `/api/user` (auth/me/update/delete), `/api/dialogue` (CRUD/queries),
`/api/message` (POST/stream/feedback), `/api/source`
(upload/list/read/download/move/delete), `/api/folder` (create/list/move/delete),
`/api/code/execute`. Registration controller есть, public route нет.

Каждый lookup user-owned entity включает `user_id`; чужой id маскируется 404.
Особенно проверяйте raw SQL/`or_` precedence. Известный FTS дефект `user_id AND
name_match OR content_match` может раскрывать чужие content matches — исправление
обязано покрыть list/count/size/folder cases PostgreSQL integration test.

При contract change меняйте schema/router/controller/tests и соответствующие
frontend transport types, adapters, stores. FastAPI error shape — `detail`.
OpenAPI snapshots синхронизированы с `app.openapi()` на 2026-09-03; при изменении
routes/schemas регенерируйте оба snapshot без запуска lifespan или реальных
подключений. Live `/api/docs` остаётся runtime-проверкой.

## Message/RAG flow

Controller проверяет dialogue/parent ownership, flush'ит одну `Message` row,
строит history, проверяет topic только для первого вопроса и всегда использует
`query_transform`. Затем rewrite + HyDE, три searches, dedup/rerank/top-k,
generation, DB source mapping, citation remap, first-message title и commit.
Non-stream path после ответа последовательно исполняет Python blocks; stream path
этого не делает.

`Message` содержит вопрос и ответ; `parent_message_id` связывает variants. SSE:

```text
data: {"type":"chunk","delta":"..."}
data: {"type":"complete","sources":[...],"message_id":...,
      "parent_message_id":...,"created_at":"..."}
data: [DONE]
```

Generator идёт через thread/async queue, но rewrite/retrieval/rerank и часть retry
остаются blocking; retry использует `time.sleep`. Cancel/error должен rollback.
Сейчас deltas уходят до citation remap, `complete` — до commit и без полного
remapped answer: не закрепляйте это как желаемый contract.

Citation grouping/remap живёт в `message_citation_utils.py`. Меняя order/dedup,
проверяйте grouped sources, `[§N]` mapping и stale test expectations.

## DB, Source и Folder

Alembic нет. `init_db()` делает `create_all` и вручную добавляет PostgreSQL
`tsvector`, GIN, triggers и columns. Schema patch включает ORM, idempotent upgrade,
DDL privileges, cascade/ownership и real PostgreSQL smoke; SQLite недостаточно.

Folder хранит `parent_id` + materialized `path`; move пересчитывает descendants BFS
и не допускает цикл. Create сейчас доверяет parent/path больше, чем следует.
Source path lookup зависит от `utils/path_utils.py`, folder path и filename.
Уникального constraint на `(user_id, folder_id, filename)` нет.

DB и Chroma не транзакционны: upload commit до embed; delete продолжает после
Chroma error; folder delete, source/folder move не синхронизируют vectors/metadata.
Backend объявляет PDF/DOCX, но UTF-8-декодирует input — binary formats не считать
поддержанными.

Soft delete user не отзывает старый JWT; current-user dependency не проверяет
`is_active`. Не ослабляйте и исправляйте только с auth regression tests.

## Проверки и baseline

Из корня репозитория:

```bash
cd backend
uv run --locked --package backend --group dev python -m pytest
```

Suite сейчас не green: stale import `generate_dialogue_title` ломает collection;
`greenlet` отсутствует; далее остаются obsolete `controller.httpx` mocks, citation
expectations и PostgreSQL `@@` test на SQLite. Не лечите это отключением tests и не
приписывайте failure новому diff без сравнения.

Backend tests обходят lifespan и реальный PostgreSQL/RAG. Для route patch проверьте
auth success/foreign-id/validation; для DB — PostgreSQL; для SSE — event order,
disconnect/rollback/commit; для upload/delete — отдельную temp Chroma collection.
