# Knowledge Base RAG

Монорепозиторий демонстрационной системы вопросов и ответов по базе знаний.
Текущий демонстрационный корпус — документация PyTorch 2.x и выгрузка
Stack Overflow. Система объединяет веб-интерфейс, FastAPI backend, PostgreSQL,
ChromaDB, dense retrieval с reranking, внешний OpenAI-compatible LLM и отдельный
сервис исполнения Python-кода.

Проект находится в исследовательском/dev-состоянии. Это не готовый production
deployment: часть контуров экспериментальна, baseline проверок красный, а в
Docker-, data- и security-path есть известные ограничения, перечисленные ниже.

## Что реализовано сейчас

- Next.js 16 / React 19 интерфейс: вход, диалоги, streaming-ответы, источники,
  папки, просмотр документов и запуск code blocks.
- FastAPI API с JWT-аутентификацией существующих пользователей, диалогами,
  сообщениями, feedback, sources/folders и SSE endpoint.
- PostgreSQL для пользователей, диалогов, сообщений, папок и источников;
  полнотекстовый поиск по сохранённым источникам реализован через PostgreSQL
  `tsvector`/GIN.
- ChromaDB с cosine distance, sentence-transformer embeddings, dense retrieval,
  CrossEncoder reranking и query transform (rewrite + HyDE).
- TF-IDF + OneClassSVM классификатор off-topic вопросов.
- Отдельный FastAPI code executor на RestrictedPython со spawned process,
  timeout и ограничениями размера.
- Offline-пакеты для подготовки и синтетической генерации датасетов,
  LoRA/QLoRA-обучения, prompt contract и оценки через lexical/semantic/RAGAS
  metrics с W&B.

Не реализованы заявлявшиеся ранее BM25/RRF/FAISS retrieval, общий backend
`/health`, Prometheus/Grafana/Jaeger и production-ready Telegram deployment.
`tgbot` и `torch-parser` остаются legacy/experimental пакетами и не входят в
Docker Compose.

## Архитектура

```text
Browser / Next.js :3000
  └─ JSON + SSE; JWT хранится в localStorage
       └─ FastAPI backend :8001
            ├─ PostgreSQL :5432
            │    users / dialogues / messages / folders / sources / FTS
            ├─ rag package
            │    ├─ ChromaDB persistent collection
            │    ├─ sentence-transformer embedder
            │    ├─ CrossEncoder reranker
            │    └─ внешний OpenAI-compatible LLM endpoint
            ├─ outlier detector
            └─ code-executor :8002
                 RestrictedPython + отдельный процесс
```

Внешний LLM/vLLM/Ollama/gateway в Compose не входит. Без доступного
OpenAI-compatible generation endpoint backend может обслуживать CRUD, но чат
RAG недоступен.

Offline-контур:

```text
документы ──► ChromaDB
     ├──────► dataset-prep ──► legacy SFT JSONL
     └──────► dataset-synth ──► teacher-generated/adversarial SFT JSONL
                                  └─► lora-train ──► LoRA adapter

lora-pipeline: ingest/synth reuse artifacts; training reruns unless skipped
eval-runner: retrieval + base/LoRA generation ──► metrics/RAGAS ──► W&B
```

## Компоненты

| Путь | Назначение | Документация / источник истины |
|---|---|---|
| `frontend/` | Next.js UI, API adapters, Zustand state, SSE, code rendering | [frontend guide](frontend/CLAUDE.md), [folders/documents](frontend/FOLDER_SYSTEM.md), [agent rules](frontend/AGENTS.md) |
| `backend/` | FastAPI, JWT, async SQLAlchemy, API и orchestration | [backend/RAG integration](backend/README_RAG_INTEGRATION.md), [agent rules](backend/AGENTS.md), live `/api/docs` |
| `rag/` | документы, индекс, retrieval, prompts, LLM и metrics | [rag/AGENTS.md](rag/AGENTS.md) |
| `code-executor/` | выполнение разрешённого Python-кода | [app.py](code-executor/app.py), [tests](code-executor/tests/test_executor.py) |
| `outlier-detection/` | тематический OneClassSVM classifier | [classifier](outlier-detection/outlier_detection/topic_classifier.py), [tests](outlier-detection/tests/test_topic_classifier.py) |
| `prompt-contract/` | версионируемый формат/fingerprint training/inference prompt | [contract](prompt-contract/prompt_contract/__init__.py), [tests](prompt-contract/tests/test_contract.py) |
| `dataset-prep/` | подготовка SFT из Stack Overflow + retrieved context | [CLI](dataset-prep/dataset_prep/__main__.py) |
| `dataset-synth/` | teacher-generated Q&A, distractors и refusals | [CLI](dataset-synth/dataset_synth/__main__.py) |
| `lora-pipeline/` | единый docs-to-adapter pipeline | [lora-pipeline/README.md](lora-pipeline/README.md) |
| `lora-train/` | Transformers/PEFT/TRL training; Linux/CUDA | [CLI](lora-train/lora_train/__main__.py), [scripts](lora-train/scripts) |
| `eval-runner/` | experiment presets, evaluation и W&B logging | [eval-runner/README.md](eval-runner/README.md), [canonical defaults](eval-runner/eval_runner/config.py) |
| `data/` | tracked corpus, Stack Overflow CSV и mutable Chroma index | не редактировать как source code |
| `scripts/` | ручные DB/import helpers | перед запуском обязательно читать исходник |
| `torch-parser/`, `tgbot/` | legacy/experimental контуры | [parser README](torch-parser/README.md), [bot source](tgbot/__main__.py) |

Общие правила разработки — в [AGENTS.md](AGENTS.md). Исполняемый код и типы имеют
приоритет над документацией и сохранёнными OpenAPI snapshots.

## Требования

- Python 3.13 для корневого `uv` workspace.
- `uv` и корневой `uv.lock`.
- Node.js 20 и pnpm для frontend. Canonical lock — `frontend/pnpm-lock.yaml`;
  `package-lock.json` является старым параллельным lock-файлом.
- Docker и Docker Compose для PostgreSQL и dev-контейнеров.
- Доступ к model registry при первом скачивании embedding/reranking моделей.
- OpenAI-compatible endpoint для генерации; для полного eval также endpoint
  judge-модели.
- Linux/CUDA для реального `lora-train`; training extra не предназначен для
  обычной macOS/CPU-разработки.

В workspace 12 Python-пакетов с общей `.venv`. `uv sync` выполняет exact sync, и
переключение `--package` может удалить зависимости другой подсистемы. Не запускайте
`uv sync --all-packages --all-groups` без необходимости: он разрешает тяжёлый
ML/CUDA stack.

## Конфигурация без секретов

Не копируйте значения из tracked env-файлов и не публикуйте их в issue, log или
документации. Корневой `.env` игнорируется Git и используется Compose;
`backend/.env` и `frontend/.env.local` уже отслеживаются. В tracked backend env
обнаружено правдоподобное секретное значение — его следует считать раскрытым и
ротировать отдельно. Component-local `torch-parser/.env` сейчас тоже не покрыт
ignore-правилом, поэтому задавайте legacy-переменные через окружение процесса.

Основные имена переменных backend/Compose:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_HOST
POSTGRES_PORT
POSTGRES_DATABASE
JWT_SECRET_KEY
JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES
CORS_ORIGINS

RAG_ENABLED
RAG_EMBEDDING_MODEL
RAG_RERANK_MODEL
RAG_LLM_MODEL
RAG_LLM_API_URL
RAG_LLM_API_KEY
RAG_LLM_TIMEOUT
RAG_TOP_K
RAG_CHUNK_SIZE
RAG_CHUNK_OVERLAP
RAG_CHROMA_PATH
RAG_CHROMA_COLLECTION
RAG_SOURCE_PATH_PREFIXES

OUTLIER_DETECTION_ENABLED
OUTLIER_CLASSIFIER_PATH
OUTLIER_REJECT_OFF_TOPIC

CODE_EXECUTOR_URL
CODE_EXECUTOR_TIMEOUT
CODE_EXECUTOR_MAX_CODE_LENGTH
CONVERSATION_HISTORY_ENABLED
CONVERSATION_MAX_HISTORY_MESSAGES
HF_TOKEN
```

Standalone `rag.Settings` использует отдельные имена с тем же префиксом; особенно
важно не путать их с backend-настройками:

```text
RAG_LLM_MODEL_GENERATION
RAG_LLM_MODEL_JUDGE
RAG_JUDGE_API_URL
RAG_JUDGE_API_KEY
RAG_JUDGE_TIMEOUT
RAG_EVAL_EMBEDDING_MODEL
RAG_DATASET_PATH
RAG_QA_DATASET_PATH
RAG_WANDB_PROJECT
RAG_WANDB_API_KEY
RAG_DEVICE
```

Frontend:

```text
NEXT_PUBLIC_API_URL
```

Legacy parser/bot используют `TORCH_URL`, `PATH_TO_SAVE`, `TGBOT_TOKEN`,
`ALLOWED_USERS`, `API_URL`, `API_HISTORY_URL` и `API_STATS_URL`; эти контуры не
нужны для основного runtime.

`NEXT_PUBLIC_API_URL` — origin backend без суффикса `/api`, потому что endpoint
paths уже содержат `/api/...`. Любая `NEXT_PUBLIC_*` переменная попадает в browser
bundle и не должна содержать секрет.

Для экспериментов также используются `WANDB_API_KEY` и параметры teacher/judge
endpoint, передаваемые CLI/config. Корневой `.env.example` намеренно содержит
только inputs, которые текущий Compose реально интерполирует. Canonical полный
список и defaults находятся в `backend/settings.py`, `rag/rag/config.py` и
соответствующих CLI config dataclasses. Pydantic читает `.env` относительно
текущего рабочего каталога, поэтому проверяйте разрешённые пути к Chroma и model.

## Запуск dev-контура

### Вариант 1: Docker Compose

Создайте приватный корневой `.env`, задайте безопасный JWT secret, PostgreSQL и
RAG-параметры, затем сначала проверьте структуру Compose:

```bash
docker compose config --quiet
docker compose up --build postgres code-executor backend frontend
```

После запуска:

- UI: `http://localhost:3000`;
- live OpenAPI/Swagger: `http://localhost:8001/api/docs`;
- backend API prefix: `http://localhost:8001/api`.

Compose — dev-конфигурация с bind mounts и `--reload`, не production baseline.
На текущем состоянии проверена только команда `docker compose config --quiet`;
полный build нельзя считать green. Известные причины:

- backend build context — весь корень репозитория, корневого `.dockerignore` нет;
- backend image не копирует `uv.lock` и `prompt-contract` и видит неполный
  workspace;
- `CODE_EXECUTOR_URL` не задан Compose явно и зависит от tracked env;
- внешний generation endpoint не поднимается;
- executor подключён к обычной сети и имеет writable bind mount.

Не выводите полный `docker compose config`: он может раскрыть подставленные
секреты. `docker compose down -v` удаляет PostgreSQL volume и допустим только для
осознанного сброса локальных данных.

### Вариант 2: запуск компонентов отдельно

PostgreSQL:

```bash
docker compose up -d postgres
```

Backend запускается из `backend/`, иначе меняется разрешение `.env` и относительных
путей:

```bash
cd backend
uv run --locked --package backend --group dev \
  uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Для CRUD-only smoke можно отключить RAG приватной runtime-конфигурацией. Для чата
нужны корректные LLM URL/model/key, Chroma path, classifier path и executor URL.
Публичного endpoint регистрации сейчас нет: пользователя нужно provision через БД.
`scripts/create_test_user.py` создаёт фиксированную demo-учётную запись и допустим
только на одноразовой локальной БД после чтения исходника.

```bash
cd backend
uv run --locked --package backend python ../scripts/create_test_user.py
```

Code executor:

```bash
cd code-executor
uv run --locked --package code-executor --group dev \
  uvicorn app:app --host 127.0.0.1 --port 8002 --reload
```

Его package manifest не включает `numpy`, `pandas` и `torch`, хотя runtime/tests
их используют; Dockerfile ставит их отдельно. Не публикуйте этот сервис напрямую.

Frontend:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

Из-за общей Python `.venv` параллельные `uv run --package ...` могут
пересинхронизировать окружение друг друга. Для длительной одновременной локальной
работы используйте изолированные environments либо контейнеры.

## API и фактическое поведение RAG

Основные routes (login и pre-generated queries не требуют JWT):

- `/api/user`: login, current user, update и soft delete;
- `/api/dialogue`: CRUD диалогов и список pre-generated queries;
- `/api/message`: обычный ответ, `/stream` через SSE и `/feedback`;
- `/api/source`: upload/list/search/read/download/move/delete;
- `/api/folder`: создание, список, перемещение и удаление папок;
- `/api/code/execute`: proxy к code executor.

Backend startup создаёт отсутствующие ORM-таблицы и выполняет PostgreSQL-specific
DDL для FTS. Alembic нет. Ошибка инициализации RAG логируется, backend продолжает
работать, а message endpoints возвращают unavailable. При включённом RAG значение
`OUTLIER_DETECTION_ENABLED=false` сейчас также ломает RAG initialization из-за
попытки загрузить classifier по `None`.

Текущий chat path использует query transform: исходный запрос, LLM rewrite и HyDE
дают три dense search, после чего candidates дедуплицируются и rerank'ятся.
PostgreSQL FTS используется для списка загруженных sources, а не как BM25-ветка
RAG retrieval.

Citation/source mapping тоже имеет известные edge cases: неизвестная folder path
может fallback'нуться к одноимённому root Source, а grouped `[§1, §2]` иногда
повторно remap'ится и получает неверные/дублирующиеся номера.

## Offline ML workflows

Сначала смотрите CLI help; эти команды сами по себе не запускают paid/full jobs:

```bash
uv run --locked --package dataset-prep python -m dataset_prep --help
uv run --locked --package dataset-synth python -m dataset_synth --help
uv run --locked --package lora-pipeline python -m lora_pipeline --help
uv run --locked --package eval-runner python -m eval_runner --help
```

Минимальный контролируемый smoke через `lora-pipeline` должен использовать новый
output directory, малый `--max-chunks` и `--skip-train`. Он всё равно вызывает
teacher endpoint. Full generation и eval делают внешние запросы; LoRA training
требует GPU. Synthetic dataset хранит
structured chunks, training рендерит их через `prompt-contract`, а serving/eval
должны явно загрузить тот же contract. Backend/eval пока не делают этого и не
валидируют adapter fingerprint автоматически.

## Проверки и известный красный baseline

Аудитированное состояние на 2026-09-03:

- `uv --no-cache lock --check` проходит; lock разрешает 211 packages;
- `docker compose config --quiet` проходит с предупреждением об устаревшем поле
  `version`; build/runtime не проверены без Docker daemon;
- статически найдено 123 Python test functions;
- Ruff на 125 tracked Python files сообщает 231 ошибку;
- Ruff formatter потребовал бы изменить 49 из 125 Python files;
- Black потребовал бы изменить 48 из 125;
- Pyright 1.1.408 не имеет стабильного repo-wide count в общей `.venv`: после
  root dev-only exact sync зафиксированы 291 errors / 9 warnings, а при наличии
  зависимостей других workspace packages — 224 / 3; оба baseline красные;
- frontend checks не запускались без установленного dependency tree;
- CI, pre-commit, Makefile, tox/nox и frontend test runner отсутствуют.

Это baseline, а не допустимость новых ошибок. Проверяйте только затронутые файлы и
не форматируйте весь репозиторий попутно:

```bash
uv --no-cache lock --check
uv run --locked --group dev ruff check <changed-python-paths>
uv run --locked --group dev black --check <changed-python-paths>
uv run --locked --group dev pyright <changed-python-paths>
git diff --check
```

Python tests:

```bash
(cd backend && uv run --locked --package backend --group dev python -m pytest)
(cd code-executor && uv run --locked --package code-executor --group dev python -m pytest)

uv run --locked --package lora-pipeline --with pytest \
  python -m pytest prompt-contract/tests dataset-synth/tests lora-pipeline/tests -q

(cd outlier-detection/outlier_detection && \
  uv run --locked --package outlier-detection --group dev python -m pytest ../tests)
```

Backend suite сейчас не green: collection ломают stale title import и missing
runtime dependency, затем остаются obsolete HTTP mocks/citation expectations и
PostgreSQL FTS на SQLite. Executor tests дополнительно зависят от scientific
packages, отсутствующих в manifest.

Frontend checks из `frontend/`:

```bash
pnpm lint
pnpm format:check
pnpm exec tsc --noEmit --incremental false
pnpm build
```

Build может требовать сеть для `next/font/google`. В `package.json` нет отдельного
test или typecheck script.

## Данные и generated artifacts

Следующие пути крупные, mutable, generated либо содержат результаты экспериментов:

- `data/dataset/`, `data/stackoverflow-pytorch.csv`, `data/sft_synth/`;
- `data/chromadb/` — tracked persistent SQLite/vector index;
- `backend/models/`, `outlier-detection/models/`, `outlier-detection/eval_results/`;
- `lora-train/runs/`, `wandb/`, `eval-runner/logs/`, `ragas_venv/`;
- `.venv`, `.next`, `*.tsbuildinfo`, notebook outputs и `smoke_test.log`.

Tracked `eval-runner/logs/` и `wandb/` могут содержать prompts, ответы, contexts,
локальные пути и сведения о host. Добавление ignore-правила не удалит уже
отслеживаемые артефакты и не очистит Git history.

Правила `.gitattributes` для `data/dataset` и `data/chromadb` не рекурсивны:
фактические файлы там являются обычными Git blobs, а Git LFS отслеживает только
презентацию PPTX. Не рассчитывайте восстановить corpus/index через `git lfs pull`.

Не тестируйте на bundled `data/chromadb`: upload/delete/rebuild могут изменить
tracked index. `index_chunks()` полностью пропускает добавление, если collection
уже непуста, поэтому runtime upload сейчас может сохраниться только в PostgreSQL.
Retrieval не фильтрует Chroma по `user_id`, а старые bundled chunks не имеют
tenant metadata. Для тестов создавайте временный Chroma path и отдельную
collection. Выбор папок в frontend пока не передаётся в message API и не задаёт
RAG scope.

Default paths dataset-synth, outlier train/eval и других scripts могут
перезаписывать tracked datasets/models/results. Full crawl, index rebuild, DB
migration/reset, teacher/judge calls, W&B/Hugging Face upload и LoRA training —
явные операции с внешними эффектами, не smoke checks.

## Security limitations

- JWT хранится frontend в `localStorage`; soft-deleted пользовательский token
  остаётся валиден до expiry.
- Shared Chroma retrieval сейчас не tenant-isolated. Есть также известный риск
  precedence в PostgreSQL FTS `name OR content` filter.
- Code executor — research sandbox, не strong isolation для враждебного кода:
  разрешённые NumPy/Pandas/Torch API имеют косвенные filesystem/network
  поверхности, а текущий Compose оставляет egress и writable bind mount.
- Backend Docker build отправляет большой корневой context Docker daemon и может
  включить локальные env/artifacts; tracked backend env попадает в image layer.
- `lora-pipeline` сохраняет `teacher_api_key` в manifest через `asdict`.
- `eval-runner` отправляет generator/judge keys в W&B config; отключить W&B через
  CLI нельзя, а RAGAS передаёт judge key subprocess через argv.
- API объявляет PDF/DOCX source types, но upload декодирует все файлы как UTF-8;
  сквозно надёжны только текстовые форматы.

Не используйте реальные cloud keys в pipeline/eval до redaction этих путей и не
считайте текущую конфигурацию production-safe.

## Дополнительная документация

- [AGENTS.md](AGENTS.md) — общие инструкции для coding agents.
- [backend/AGENTS.md](backend/AGENTS.md) — API, ownership, DB и SSE invariants.
- [backend/README_RAG_INTEGRATION.md](backend/README_RAG_INTEGRATION.md) — текущая
  интеграция backend, RAG, PostgreSQL и executor.
- [rag/AGENTS.md](rag/AGENTS.md) — index/retrieval/prompt contract.
- [frontend/AGENTS.md](frontend/AGENTS.md) — frontend contracts и visual checks.
- [frontend/CLAUDE.md](frontend/CLAUDE.md) и
  [frontend/FOLDER_SYSTEM.md](frontend/FOLDER_SYSTEM.md) — runtime и фактическая
  реализация folders/documents.
- [lora-pipeline/README.md](lora-pipeline/README.md) — docs-to-adapter workflow.
- [eval-runner/README.md](eval-runner/README.md) — experiment configuration;
  defaults и список presets перепроверяйте по коду.
- `backend/openapi.json` и `backend/openapi.yaml` — snapshots из текущего
  `app.openapi()` (2026-09-03; 15 paths, 24 operations, 31 schema). После
  route/schema changes регенерируйте оба файла. Они ошибочно показывают JSON для
  SSE/download responses; runtime сверяйте отдельно через `/api/docs` и smoke.
