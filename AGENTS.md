# AGENTS.md — руководство для coding agents

Этот файл действует для всего репозитория и предназначен для Codex, Claude Code,
Cursor и других coding agents. Явный запрос пользователя имеет приоритет.
`backend/AGENTS.md`, `rag/AGENTS.md` и `frontend/AGENTS.md` дополняют этот
файл. Перед правкой соответствующей подсистемы прочитайте дополнение вручную, даже
если агент запущен из корня.

Codex по умолчанию загружает не более 32 KiB совокупных project instructions.
Сохраняйте этот файл ниже лимита: добавляйте только проверенные, полезные в работе
правила; детали реализации ищите в коде. Не дублируйте один тезис в нескольких
разделах. После правки проверяйте, что размер root плюс любого одного nested
`AGENTS.md` остаётся меньше 32768 байт.

## 1. Рабочий протокол

Цель — минимальное законченное изменение, подтверждённое подходящей проверкой.

Перед правкой:

1. Прочитайте запрос и определите затронутые подсистемы.
2. Выполните `git status --short`: существующие изменения принадлежат пользователю.
3. Ищите через `rg`/`rg --files`; прочитайте реализацию, вызывающий код, типы,
   конфигурацию и релевантные тесты.
4. Зафиксируйте критерий готовности и риски: API, данные, безопасность, стоимость.

Во время работы:

- Не делайте попутный рефакторинг, массовое форматирование, обновление зависимостей
  или перегенерацию артефактов.
- Сохраняйте public contracts и старые форматы, если задача не требует миграции.
- Не скрывайте ошибки широким fallback, `except`, `# noqa` или отключением теста.
- Не ослабляйте auth, tenant ownership, sandbox или валидацию.
- Не запускайте платные, сетевые, GPU-heavy, destructive или долгие data jobs лишь
  ради проверки.
- После правки просмотрите полный diff, `git diff --check` и `git status --short`.

В отчёте отделяйте реально выполненные проверки от непроверенного. Не называйте
baseline зелёным, если команда не запускалась либо упала по известной причине.

## 2. Проект и архитектура

Это монорепозиторий private knowledge-base RAG. Демонстрационный корпус — PyTorch
2.x и StackOverflow. Основной flow:

```text
Browser / Next.js :3000
  └─ JSON + SSE, JWT из localStorage
       └─ FastAPI backend :8001
            ├─ PostgreSQL :5432 — users/dialogues/messages/folders/sources + FTS
            ├─ rag — chunking, Chroma, dense retrieval, rerank, LLM
            │    └─ внешний OpenAI-compatible endpoint, не входит в compose
            ├─ outlier detector — проверка первого вопроса
            └─ code-executor :8002 — RestrictedPython в отдельном процессе
```

Offline flow:

```text
docs → Chroma
  ├─ dataset-prep: StackOverflow + retrieved legacy flat context
  └─ dataset-synth: teacher LLM → structured chunks/distractors/refusals
       └─ lora-train: LoRA/QLoRA adapter + prompt_contract.json

lora-pipeline: ingest/synth reuse → training reruns unless skipped
eval-runner: base/LoRA/RAG metrics + optional RAGAS judge + W&B
```

`docker-compose.yml` поднимает только `postgres`, `code-executor`, `backend` и
`frontend`. vLLM/Ollama/gateway, Telegram bot и observability там отсутствуют.

## 3. Карта репозитория

| Путь | Назначение и источник истины |
|---|---|
| `backend/` | FastAPI/JWT/async SQLAlchemy; `app.py`, `api/*`, `services/*`, `db.py`, `tests/` |
| `rag/rag/` | `documents.py`, `vectorstore.py`, `retriever.py`, `chains.py`, `llm.py` |
| `frontend/` | Next.js App Router; `app/`, `components/`, `hooks/`, `lib/api/`, `lib/store/` |
| `code-executor/` | Изолированное исполнение Python; `app.py`, `tests/test_executor.py` |
| `prompt-contract/` | Версионируемый prompt format/fingerprint для train и явного inference |
| `dataset-prep/` | Legacy подготовка SFT из StackOverflow и retrieval context |
| `dataset-synth/` | Teacher-generated Q&A, distractors, adversarial rows; groundedness не проверяется |
| `lora-pipeline/` | Docs-to-adapter CLI; ingest/synth reuse, training reruns; manifest |
| `lora-train/` | Transformers/PEFT/TRL; реальное обучение только Linux/CUDA |
| `eval-runner/` | Base/LoRA/RAG eval, RAGAS, W&B |
| `outlier-detection/` | TF-IDF + OneClassSVM classifier, scripts/tests/models |
| `torch-parser/`, `tgbot/` | Legacy/experimental контуры, не текущий runtime |
| `scripts/` | Ручные DB/import helpers; часть destructive или для старого API |
| `data/` | Крупный corpus и mutable Chroma index; это не source code |
| `notebooks/`, `wandb/`, `lora-train/runs/` | Исследования и сохранённые результаты |

`frontend/clo` — отслеживаемый ELF-бинарник неясного происхождения: не запускайте и
не заменяйте его. Не ориентируйтесь на caches, `.venv`, `.next` или notebook output.

## 4. Что считать источником истины

При конфликте: запрос/ближайший `AGENTS.md` → код и типы → тесты/config → docs.
Guides синхронизированы на 2026-09-03, но код остаётся источником истины.

- `backend/openapi.json`/`.yaml` сгенерированы из `app.openapi()`; обновляйте оба
  после route/schema change. Схема не отражает реальные SSE/download media types.
- `notebooks/defense_deck.md` — исторический отчёт с legacy-весами метрик, не
  описание текущего runtime или production policy.
- Веса и defaults eval берите из `eval_runner/config.py`; README объясняет их, но
  config остаётся canonical.
- `.env.example` содержит только Compose inputs; прочие defaults берите из Settings.

Меняя фактическое поведение, обновите связанную документацию в том же diff.

## 5. Toolchain и зависимости

- Корневой `pyproject.toml` — uv workspace из 12 пакетов; `uv.lock` canonical.
- Из корня используйте Python 3.13. Некоторые ML-пакеты декларируют `>=3.12`, но
  общий lock требует `>=3.13`, поэтому workspace через Python 3.12 не работает.
- Все Python-пакеты делят одну `.venv`. `uv sync` делает exact sync и при смене
  `--package` может удалить зависимости другой подсистемы.
- Используйте `uv run --locked --package <name> ...`. Не делайте
  `uv sync --all-packages --all-groups`: это тянет тяжёлый ML/CUDA stack.
- Для frontend canonical — Node 20 и `pnpm`/`frontend/pnpm-lock.yaml`. Старый
  `package-lock.json` не обновлять; `npm install` не запускать.
- CI, pre-commit, Makefile, tox/nox и frontend test runner отсутствуют.

`.gitattributes` пытается назначить LFS путям `data/dataset` и `data/chromadb`, но
паттерны не рекурсивны. Фактические файлы corpus/Chroma — обычные tracked blobs;
`git lfs ls-files` показывает только PPTX. Не рассчитывайте, что `git lfs pull`
восстановит данные, и не переписывайте LFS-правила попутно.

Установка frontend:

```bash
cd frontend
pnpm install --frozen-lockfile
```

Не редактируйте lock-файлы вручную. Dependency change должен менять правильный
package manifest и соответствующий lock одним осмысленным diff.

## 6. Конфигурация и секреты

Не печатайте env values, пароли/API keys в terminal, diff, logs, tests, docs или
ответ. Для диагностики показывайте только имя и факт наличия.

- Root `.env` ignored; `backend/.env` и `frontend/.env.local` tracked. В backend
  обнаружено правдоподобное секретное значение: считайте раскрытым, не копируйте и
  не полагайтесь на него. Ротация/history cleanup — отдельная согласованная задача.
- `NEXT_PUBLIC_*` попадает в browser bundle. Новые secrets — local secret store;
  examples содержат только placeholders.
- Compose читает root env; backend/standalone RAG — `.env` относительно cwd;
  scripts могут явно читать backend env. Всегда проверяйте resolved data/model path.

Canonical backend config — `backend/settings.py`: PostgreSQL/JWT/CORS/RAG/outlier/
executor/history. Insecure JWT defaults отвергаются. Backend `RAG_LLM_MODEL` и
standalone `RAG_LLM_MODEL_GENERATION` различаются; unknown env молча ignored.
Frontend `NEXT_PUBLIC_API_URL` — backend origin без `/api`; fallback `"/api"`
сейчас образует ошибочный `/api/api/...`.

ML secret hazards: pipeline manifest сохраняет `teacher_api_key` через `asdict`;
eval отправляет generator/judge keys в W&B config тем же способом, причём отключить
W&B через CLI нельзя. До redaction не запускайте cloud-key eval и не сохраняйте
реальный teacher key в manifest. Judge key имеет только JSON overlay, попадает в
W&B config и передаётся RAGAS subprocess через argv; до исправления этот cloud path
небезопасен даже с private/ignored config.

## 7. Запуск и Docker

```bash
docker compose config --quiet
docker compose up --build postgres code-executor backend frontend
```

Не печатайте полный config с secrets и не делайте `down -v` без разрешения.
Compose — dev setup с bind mounts/reload, внешнего LLM endpoint в нём нет.

Docker baseline нельзя считать green:

- Backend context = repo root (~2.2 GiB локально), root `.dockerignore` отсутствует;
  daemon получает env/data/venvs/runs, а `COPY backend ./` кладёт tracked env в
  image layer. `backend/.dockerignore` здесь не действует.
- Backend image не копирует `uv.lock`/`prompt-contract` и не удовлетворяет всем
  workspace members.
- `CODE_EXECUTOR_URL` не задан Compose: container service URL сейчас зависит от
  tracked env, тогда как default localhost неверен.
- Executor подключён также к non-internal network; writable bind mount ослабляет
  `read_only`. Frontend pnpm и executor scientific deps ставятся вне общего lock.

Локально:

```bash
docker compose up -d postgres
(cd backend && uv run --locked --package backend --group dev \
  uvicorn app:app --host 0.0.0.0 --port 8001 --reload)

(cd code-executor && uv run --locked --package code-executor --group dev \
  uvicorn app:app --host 127.0.0.1 --port 8002 --reload)

(cd frontend && pnpm dev)
```

Docs `:8001/api/docs`, UI `:3000`. Первый RAG start может скачать models.
`RAG_ENABLED=false` помогает CRUD smoke, но `OUTLIER_DETECTION_ENABLED=false`
сейчас приводит к `TopicClassifier.load(None)` и ломает RAG init. Для host-run
явно задайте корректные Chroma/classifier/executor paths, не меняя tracked env.

## 8. Backend

Перед любым изменением `backend/` прочитайте `backend/AGENTS.md`, даже если Codex
запущен из корня. Там зафиксированы layers, routes, ownership, message/SSE flow,
PostgreSQL DDL, Source/Folder invariants и красный test baseline.

Cross-layer минимум: user-owned lookup всегда включает `user_id`; API patch меняет
Pydantic/router/controller/tests и frontend type/adapter/store; SQLite не заменяет
PostgreSQL FTS test; DB↔Chroma операции проверяются отдельно. Не запускайте
destructive scripts и не считайте PDF/DOCX поддержанными без parser.

## 9. RAG, dataset, LoRA и eval

Для `rag/` прочитайте `rag/AGENTS.md`: там index contract, retrieval strategies,
prompt fingerprint и temp-test requirements. Главные общие запреты: не мутировать
bundled Chroma, не смешивать несовместимые embedding/chunk/metadata versions, не
считать shared retrieval tenant-isolated и не считать training contract
подключённым к serving/eval.

В `dataset-synth` normal row содержит gold, adversarial — нет. JSONL сохраняет
chunk ids/order, но не seed; pipeline manifest хранит config. `dataset-prep` пишет
legacy flat context;
`lora-train` читает оба формата. Не удаляйте legacy reader без миграции. В PEFT
`"all-linear"` остаётся строкой; loss сейчас по полной sequence; output — adapter,
не merged model.

Eval config precedence: `defaults → configs по порядку → CLI`. Для сравнения
фиксируйте corpus/index/sample/seed/generator params/prompt/judge и меняйте одну
ось. Full synth/eval вызывает teacher/generator/judge endpoints и может быть
платным; RAGAS управляет отдельным env. Training требует GPU/model files;
standalone CLI по умолчанию пишет в W&B, pipeline — нет. Full jobs создают
артефакты. Без явного запроса — лишь unit tests/offline smoke без LLM/GPU в новом
output.

## 10. Code execution — security boundary

Flow: frontend prefilter → authenticated backend proxy → `code-executor`. Python
проходит AST allowlist, RestrictedPython, spawned process, timeout и input/output
limits. Frontend validation — только UX, не security boundary.
С текущими writable mount, egress и мощными numpy/pandas/torch API это research
sandbox, а не strong isolation для враждебного кода.

- Не ослабляйте import/AST/guards/timeout/limits/auth/Docker hardening.
- Новый разрешённый модуль требует threat analysis и negative tests: filesystem,
  process, network, reflection, import bypass, `eval`/`exec`/`open`, oversized
  input/output, timeout и sanitized traceback.
- Не публикуйте executor напрямую. Текущий Compose network/read-only не даёт полной
  изоляции; проверяйте фактический container.
- JavaScript идёт через sandboxed iframe, WebGPU/ONNX — отдельными browser paths.
  Не включайте raw Markdown HTML и не ослабляйте iframe. CDN integrity constants
  сейчас не означают фактическую hash verification; validators покрывают не все
  execution paths.

## 11. Frontend

Перед изменением `frontend/` прочитайте `frontend/AGENTS.md`, даже если сессия
запущена из корня. Там описаны routes, API/auth/message ID/SSE/folder/document
contracts, code-rendering boundary, style и browser checklist.

Кратко: Node 20 + pnpm; `NEXT_PUBLIC_API_URL` — origin без `/api`; API transport/
adapter/store меняются вместе с backend; auth только localStorage; folder selection
не доходит до retrieval; raw Markdown HTML и sandbox weakening запрещены. Не
трогайте `clo`, `package-lock.json`, `.next` и tracked `tsconfig.tsbuildinfo`.

## 12. Проверки

Запускайте минимальный риск-ориентированный набор. Whole-repo Ruff/format baseline
красный; проверяйте changed files и не форматируйте unrelated:

```bash
uv --no-cache lock --check
uv run --locked --group dev ruff check <changed-python-paths>
uv run --locked --group dev black --check <changed-python-paths>
uv run --locked --group dev pyright <changed-python-paths>
git diff --check
```

`uv run` может менять общую `.venv`/качать packages; отсутствие сети не лечите
lock change. Package tests:

```bash
(cd backend && uv run --locked --package backend --group dev python -m pytest)
(cd code-executor && uv run --locked --package code-executor --group dev python -m pytest)
uv run --locked --package lora-pipeline --with pytest \
  python -m pytest prompt-contract/tests dataset-synth/tests lora-pipeline/tests -q
(cd outlier-detection/outlier_detection && \
  uv run --locked --package outlier-detection --group dev python -m pytest ../tests)
```

Backend baseline заведомо красный (stale title import, missing greenlet, obsolete
httpx/citation mocks, PostgreSQL FTS на SQLite); подробности в nested guide.
Executor tests требуют неописанные manifest'ом numpy/pandas/torch. Не смешивайте
environment/legacy failures с regression; для отсутствующего suite пишите targeted
test/smoke.

Frontend из `frontend/`:

```bash
pnpm lint
pnpm format:check
pnpm exec tsc --noEmit --incremental false
pnpm build
```

Test script нет; build может требовать сеть для fonts. Visual QA обязателен для UI.
Integration: `docker compose config --quiet`, build/up только нужных services, затем
проверка flow — auth/schema, SSE order/cancel/commit, PostgreSQL DDL/FTS,
DB↔Chroma и executor negative cases.

## 13. Матрица влияния

- **API/schema:** backend model/router/controller/tests + frontend transport type,
  adapter, mutation/query, domain type, store/consumer; JSON/SSE/204/error shape.
- **DB:** `db.py`, cascade/ownership, idempotent upgrade, PostgreSQL smoke; никаких
  destructive resets.
- **Retrieval/index:** `rag/rag/documents.py`, `vectorstore.py`, `retriever.py`,
  backend RAG/source/citations, datasets и eval; определите index compatibility.
- **Prompt/answer:** prompt contract, `chains.py`/`prompts.py`, synth schema,
  training formatter, adapter, citation parser, eval config/fingerprint.
- **Folders/sources:** materialized paths, ownership, DB↔Chroma metadata, citations,
  frontend store/hook/DnD/breadcrumbs/search/upload/download.
- **Code execution:** frontend renderer/validators, backend proxy, executor AST/
  namespace/process, container isolation и negative tests.
- **Dependency:** правильный package `pyproject.toml` + `uv.lock` либо
  `package.json` + `pnpm-lock.yaml`; обоснуйте новую тяжёлую ML/browser dependency.

## 14. Данные, scripts и внешние эффекты

Не меняйте без прямой задачи:

- `data/chromadb/**` — tracked mutable SQLite/vector index;
- `data/dataset/**`, `data/stackoverflow-pytorch.csv` и `data/sft_synth/**`;
- `backend/models/*.joblib`, `outlier-detection/models`/`eval_results`;
- `lora-train/runs/**`, `wandb/**`, `eval-runner/logs/**`, `ragas_venv/**` — могут
  содержать prompts, contexts, host paths и другие приватные данные;
- `.venv`, `.next`, `*.tsbuildinfo`, caches, notebook output, `smoke_test.log`;
- `backend/openapi.*` без осознанной регенерации.

Ignore rules не защищают уже tracked files. Default synth/eval/outlier outputs могут
перезаписать tracked data/models/results; любой smoke направляйте в новый temp/
ignored output. Не открывайте Chroma SQLite инструментом, который мигрирует schema,
и не пересохраняйте joblib другой версией sklearn.

Legacy scripts опасны: loaders ходят на отсутствующий `:8000/embed/document`,
выбирают первого DB user и могут коммитить Source после failed embed;
`cleanup_database.py` оставляет stale vectors; outlier train/eval перезаписывают
tracked outputs и eval включает W&B по умолчанию. `torch-parser` и `tgbot` имеют
подтверждённые packaging/CMD проблемы — сначала воспроизведите, не «чините» import
через новый `sys.path` hack.

Требуют явного разрешения/контекста: rebuild/delete Chroma; shared/prod DB migration;
удаление Docker volumes; полный crawl; paid teacher/judge/eval; W&B upload; LoRA
training/HF upload; публикация executor; исполнение неизвестного кода. Сохраняйте
для эксперимента отдельный output, seed, config без secrets, Git SHA, endpoint model,
corpus/index provenance, prompt fingerprint и manifest.

## 15. Приоритетные известные дефекты

Это context, не разрешение на попутный fix:

1. Chroma: upload в непустой index пропускается; tenant filter/`user_id` нет;
   DB↔vector upload/delete/move/folder-delete расходятся.
2. Backend: FTS `AND ... OR ...` может обходить ownership; soft-deleted JWT жив;
   outlier-off ломает RAG init; cwd paths несовместимы.
3. SSE показывает deltas до citation remap и complete до commit; grouped citations
   могут remap'иться дважды; blocking LLM/retrieval/`time.sleep` есть в async flow.
4. Docker отправляет большой repo/secrets в context/image, dependency build
   неполон; executor isolation и service URL зависят от dev details.
5. Frontend folder scope/upload/rename/delete/download и ряд auth flows неполны;
   code-render validators/integrity/stop не образуют полную security boundary.
6. Tracked Chroma/data/default ML outputs легко мутировать; pipeline/eval могут
   записать API keys в manifest/W&B.
7. OpenAPI имеет gaps для SSE/download media types; legacy scripts/parser/bot и
   tests drift'ят. CI/Alembic/frontend tests отсутствуют.

## 16. Definition of Done

Изменение готово, когда:

- запрос выполнен без незаявленного расширения scope;
- diff не содержит secrets, чужих правок, caches или generated noise;
- затронутые public contracts синхронизированы между слоями;
- ownership/security/index/prompt/data invariants не ослаблены;
- добавлен regression test либо объяснён подходящий smoke;
- реально выполнен релевантный lint/test/type/build/integration набор;
- `git diff --check` проходит, документация отражает новое поведение;
- отчёт перечисляет проверки, непроверенное и остаточные риски.

Если проверка требует сети, secrets, PostgreSQL, browser, GPU, платного API или
пересоздания данных, не симулируйте успех: выполните доступное локально и точно
опишите оставшийся шаг.
