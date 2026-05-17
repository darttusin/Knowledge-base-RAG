# eval-runner — параметры конфигурации

Все настройки одного прогона объявлены в [`eval_runner/config.py`](eval_runner/config.py)
в dataclass `RunConfig`. Любое поле задаётся одним из трёх способов:

1. **CLI-флаг**: `--retriever-type rerank --top-k 5`
2. **JSON-пресет(ы)**: `--config configs/_base.json --config configs/lora-rerank-k5.json`
   (поля файла совпадают с именами полей `RunConfig`; ключи, начинающиеся
   с `_`, считаются комментариями и игнорируются)
3. **Комбинация**: JSON задаёт базу, CLI перекрывает отдельные поля

**Порядок применения** (last wins): defaults → каждый `--config` файл по очереди → CLI-флаги.

`asdict(RunConfig)` целиком уходит в `wandb.config`, поэтому любое сравнение
runs в wandb знает, чем они отличаются.

## Готовые пресеты — папка [`configs/`](configs/)

12 файлов под рекомендованную матрицу экспериментов. Используются
послойно: общая база `_base.json` + конкретный delta-файл эксперимента.

| Файл | Phase | Что делает |
|---|---|---|
| `_base.json` | — | Общие настройки (chroma path, embed model, eval seed/sample). Loaded first. |
| `base-vanilla-k5.json` | 1.1 | Baseline: base LLM + vanilla retrieval, k=5 |
| `base-rerank-k5.json` | 1.2 | Base LLM + cross-encoder rerank |
| `base-qt-k5.json` | 1.3 | Base LLM + query rewriting + HyDE + rerank |
| `lora-rerank-k5.json` | 1.4 | **Main**: LoRA-finetuned + rerank (best retriever assumed) |
| `lora-rerank-k3.json` | 2 | top_k sweep, k=3 |
| `lora-rerank-k10.json` | 2 | top_k sweep, k=10 |
| `lora-rerank-k5-fetch10.json` | 2 | fetch_k sweep, 10 candidates |
| `lora-rerank-k5-fetch50.json` | 2 | fetch_k sweep, 50 candidates |
| `lora-rerank-k5-temp0.json` | 2 | temperature=0 (deterministic) |
| `lora-rerank-k5-seed43.json` | 3 | Stability: eval_seed=43 |
| `lora-rerank-k5-seed44.json` | 3 | Stability: eval_seed=44 |

URLs (`llm_api_url`, `judge_api_url`) намеренно отсутствуют в пресетах —
задайте их через CLI под текущий vast.ai инстанс:

```bash
python -m eval_runner \
    --config configs/_base.json \
    --config configs/lora-rerank-k5.json \
    --llm-api-url http://193.222.57.16:44090/v1 \
    --judge-api-url http://193.222.57.16:44090/v1
```

Чтобы прогнать всю матрицу из 12 runs последовательно:

```bash
for c in configs/base-vanilla-k5 \
         configs/base-rerank-k5 \
         configs/base-qt-k5 \
         configs/lora-rerank-k5 \
         configs/lora-rerank-k3 \
         configs/lora-rerank-k10 \
         configs/lora-rerank-k5-fetch10 \
         configs/lora-rerank-k5-fetch50 \
         configs/lora-rerank-k5-temp0 \
         configs/lora-rerank-k5-seed43 \
         configs/lora-rerank-k5-seed44; do
    python -m eval_runner \
        --config configs/_base.json \
        --config "$c.json" \
        --llm-api-url http://YOUR_IP:PORT/v1 \
        --judge-api-url http://YOUR_IP:PORT/v1
done
```

RAGAS venv создаётся один раз (первый run ~+3 мин) и переиспользуется
во всех последующих — общее время матрицы ~3 часа.

---

## Retriever

Параметры извлечения чанков из ChromaDB.

| Параметр | Дефолт | Тип | Что значит |
|---|---|---|---|
| `retriever_type` | `vanilla` | `vanilla` / `rerank` / `query_transform` | Стратегия извлечения |
| `top_k` | `5` | int | Сколько чанков попадает в финальный context |
| `fetch_k` | `20` | int | Кандидатов из ChromaDB до reranker (только для `rerank` / `query_transform`) |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | str | Модель для эмбеддинга запроса и чанков |
| `rerank_model` | `BAAI/bge-reranker-base` | str | Cross-encoder для переранжирования |
| `chroma_path` | `data/chromadb` | str | Путь к локальному ChromaDB |
| `chroma_collection` | `docs_fast` | str | Имя коллекции внутри ChromaDB |
| `device` | `auto` | `auto` / `cpu` / `cuda` / `mps` | На чём гонять embedder и reranker |

⚠️ Если меняете `embedding_model` — нужно **перестроить ChromaDB тем же эмбеддером**,
иначе query-вектор окажется в другом пространстве и retrieval сломается.

---

## Generator LLM

Главная точка для сравнения **base vs LoRA** — меняется `llm_model` (alias `--lora-modules`
для LoRA-варианта vs имя базовой модели).

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | **Ключевой**: имя модели как её знает vLLM |
| `llm_api_url` | пусто (обязательно!) | URL вашего vLLM endpoint, напр. `http://193.222.57.16:44090/v1` |
| `llm_api_key` | `EMPTY` | Для vLLM не валидируется; для cloud — реальный ключ |
| `llm_temperature` | `0.1` | Сэмплинг. `0.0` = детерминированно |
| `llm_max_tokens` | `1024` | Максимальная длина ответа |
| `llm_timeout` | `60.0` | Сколько секунд ждать ответа |

---

## Judge LLM (для RAGAS)

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `judge_model` | `gpt-4o-mini` | Модель-судья |
| `judge_api_url` | пусто | URL судьи. Свой vLLM, OpenAI, OpenRouter и т.д. |
| `judge_api_key` | `EMPTY` | Через CLI не задаётся — только через JSON-пресет (для cloud-судей) |

⚠️ Если judge == generator (одна и та же модель), RAGAS-метрики будут
**biased**: модель оценивает свои же ответы. Годится для относительного
сравнения (LoRA-вариант vs base), но не как абсолютная оценка.

---

## Eval dataset

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `eval_csv_path` | `data/stackoverflow-pytorch.csv` | Откуда брать вопросы и эталонные ответы |
| `eval_sample_size` | `100` | Сколько примеров оценивать |
| `eval_min_answer_score` | `1` | Минимальный `answer_score` StackOverflow для фильтра |
| `eval_seed` | `42` | Сид сэмплинга — фиксируйте, иначе разные runs берут разные вопросы |

---

## Семантическая метрика

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `eval_embedding_model` | `Snowflake/snowflake-arctic-embed-m` | Отдельный эмбеддер для `cosine(answer_rag, answer_pure)`. Намеренно ОТЛИЧНЫЙ от retriever — чтобы метрика не оценивала «сама себя» |

---

## Что считать

| Параметр | Дефолт | CLI-флаг для выключения |
|---|---|---|
| `compute_lexical` | `True` | `--no-lexical` |
| `compute_semantic` | `True` | `--no-semantic` |
| `compute_ragas` | `True` | `--no-ragas` |

Выключение `compute_ragas` экономит ~3 мин на setup изолированного venv + время на запросы судье.

---

## Веса композитного RAG-score

Композитный балл, который агрегирует RAGAS-метрики в одно число:

```
rag_score = w_faithfulness × faithfulness
          + w_answer_relevancy × answer_relevancy
          + w_context_recall × context_recall
```

| Параметр | Дефолт | Источник дефолта |
|---|---|---|
| `rag_score_w_faithfulness` | `0.4` | `notebooks/BaseLine.ipynb` |
| `rag_score_w_answer_relevancy` | `0.4` | то же |
| `rag_score_w_context_recall` | `0.2` | то же |

---

## Tracking (wandb)

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `wandb_project` | `pytorch-rag-eval` | Имя проекта в wandb |
| `wandb_run_name` | `None` (auto) | Имя конкретного run'а. Если `None` — генерируется по шаблону `{model}_{retr}_k{k}_n{n}` |
| `wandb_tags` | `[]` | Теги для фильтрации в UI: `--wandb-tag lora --wandb-tag rerank` (флаг повторяемый) |
| `wandb_notes` | пусто | Длинное текстовое описание |
| `wandb_api_key` | — | Только из CLI/env, в config не сохраняется |

---

## Метаданные эксперимента

| Параметр | Дефолт | Что меняется |
|---|---|---|
| `description` | пусто | Короткое описание для wandb.config |
| `metadata` | `{}` | Произвольный JSON: путь к LoRA-адаптеру, git SHA, hyperparams тренировки и т.д. — `--metadata-json '{"lora":"runs/v1/final"}'` |

---

## Оси сравнения для типовых экспериментов

| Что сравниваете | Меняйте | Не меняйте |
|---|---|---|
| **Base vs LoRA** | `llm_model` | retriever, top_k, judge, eval_seed |
| **Vanilla vs Rerank vs Query Transform** | `retriever_type` | llm_model, top_k, judge, eval_seed |
| **Чувствительность к top_k** | `top_k` (3, 5, 10) | retriever_type, llm_model |
| **Чувствительность к fetch_k** | `fetch_k` (10, 20, 50) | retriever_type=rerank, top_k |
| **Другой embedder** | `embedding_model` + перестроить ChromaDB | judge, llm_model |
| **Другой reranker** | `rerank_model` | retriever_type=rerank, top_k |
| **Температура** | `llm_temperature` | всё остальное |
| **Размер ответа** | `llm_max_tokens` | всё остальное |
| **Объективность судьи** | `judge_model`, `judge_api_url` | llm_model, retriever |
| **Стабильность по выборке** | `eval_seed` (несколько прогонов с разными сидами) | всё остальное |

---

## Чего **нельзя** менять через `RunConfig` (захардкожено в коде)

Эти точки фиксированы в исходниках `rag/`. Чтобы их менять — нужна правка кода.

| Параметр | Где сейчас | Когда понадобится |
|---|---|---|
| **System prompt** | `rag/rag/prompts.py:5` — `SYSTEM_INSTRUCTIONS` | A/B-тестировать разные стили инструкций |
| **Answer format prompt** | `rag/rag/prompts.py:13` — `ANSWER_INSTRUCTIONS` | Менять формат с цитированием/без, длину |
| **Context truncation** | `rag/rag/prompts.py:24` — `render_context(max_chars=14000)` | Упираетесь в context window |
| **HyDE prompt** | `rag/rag/retriever.py:86` — `HYDE_PROMPT` | Менять стратегию query transform |
| **Query rewrite prompt** | `rag/rag/retriever.py:79` — `REWRITE_PROMPT` | То же |
| **RAGAS metrics list** | `rag/rag/evaluation.py:95` — `_RAGAS_SCRIPT` | Добавить `context_precision`, `answer_correctness` |
| **RAGAS judge timeout** | `rag/rag/evaluation.py` — `RunConfig(timeout=180)` | Если ваш судья медленный |
| **RAGAS concurrency** | `rag/rag/evaluation.py` — `max_workers=4` | Параллелить больше/меньше |

Если на каком-то из этих понадобится конфигурируемость — выносим в `RunConfig`
отдельным полем.
