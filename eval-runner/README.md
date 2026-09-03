# eval-runner — воспроизводимая оценка RAG

`eval-runner` сравнивает RAG-ответ и ответ той же generator-модели без контекста
на выборке StackOverflow, считает локальные метрики, при необходимости запускает
RAGAS и отправляет результаты в Weights & Biases (W&B).

Это исследовательский инструмент, а не часть HTTP runtime. Он использует текущие
retriever, prompts и LLM client из пакета `rag`, открывает существующую ChromaDB
только для поиска и обращается к внешнему OpenAI-compatible endpoint.

> Важно: текущий CLI **всегда вызывает W&B** и передаёт туда полный
> `RunConfig`, per-row вопросы, эталонные и сгенерированные ответы, а также
> retrieved context. Кроме того, `llm_api_key` и `judge_api_key` входят в
> сериализуемый config. Не запускайте этот код с реальными cloud API keys или
> приватным корпусом, пока в коде не реализованы redaction и явный режим без W&B.
> Для локального vLLM используйте значение `EMPTY`.

## Минимальный локальный smoke

Команды ниже выполняются из корня репозитория. Общий `uv.lock` требует Python
3.13, хотя package manifest допускает более старую версию.

```bash
WANDB_MODE=offline uv run --locked --package eval-runner python -m eval_runner \
  --config eval-runner/configs/_base.json \
  --config eval-runner/configs/base-vanilla-k5.json \
  --llm-api-url http://127.0.0.1:8000/v1 \
  --eval-sample-size 1 \
  --no-semantic \
  --no-ragas
```

Даже этот smoke открывает Chroma, загружает query embedder и делает два LLM calls:
RAG и pure baseline. `WANDB_MODE=offline` не отменяет запись чувствительных данных
в локальные W&B-артефакты. Отдельного `--no-wandb` сейчас нет.

Первый запуск может скачать embedding model; reranker скачивается только для
`rerank`/`query_transform`, а semantic model — если не указан `--no-semantic`.
Не меняйте bundled `data/chromadb`: eval открывает коллекцию `docs_fast`, и query
embedder обязан совпадать с тем, которым индекс был построен.

Полный список параметров:

```bash
uv run --locked --package eval-runner python -m eval_runner --help
```

## Как собирается конфигурация

Источник истины — `eval_runner/config.py`, dataclass `RunConfig`.

Приоритет значений, от меньшего к большему:

```text
defaults → JSON overlays в порядке CLI → явные CLI-флаги
```

`--config` повторяемый. Ключи JSON совпадают с полями `RunConfig`; ключи,
начинающиеся с `_`, игнорируются и могут служить комментариями. Неизвестное поле
завершает запуск с ошибкой. CLI покрывает не все поля: например,
`judge_api_key` и веса composite score можно изменить только через JSON overlay.

### Готовые presets

В `configs/` сейчас 16 JSON-файлов: два общих overlay и 14 run-specific presets.

| Группа | Файлы | Назначение |
|---|---|---|
| Общая база | `_base.json` | Corpus/index, sample, seed, judge model и W&B project |
| Judge overlay | `_judge_qwen32b.json` | Отдельная Qwen 32B judge model; URL всё равно задаётся при запуске |
| Base | `base-{vanilla,rerank,qt}-k5.json` | Базовая generator-модель с тремя retrieval strategies |
| LoRA v1 | `lora-rerank-k5.json` и варианты `k3`, `k10`, `fetch10`, `fetch50`, `temp0`, `seed43`, `seed44` | Основной v1 run и sensitivity/stability варианты |
| LoRA v2 | `v2-{vanilla,rerank,qt}-k5.json` | Synth-v2 adapter alias с тремя retrieval strategies |

Имена `pytorch-rag` и `synth-v2` в presets — aliases, которые должен реально
обслуживать запущенный vLLM через `--lora-modules`. Поле `metadata.lora_adapter`
само по себе адаптер не загружает.

Пример с отдельным локальным judge endpoint:

```bash
WANDB_MODE=offline uv run --locked --package eval-runner python -m eval_runner \
  --config eval-runner/configs/_base.json \
  --config eval-runner/configs/_judge_qwen32b.json \
  --config eval-runner/configs/v2-rerank-k5.json \
  --llm-api-url http://127.0.0.1:8000/v1 \
  --judge-api-url http://127.0.0.1:8001/v1
```

Это всё ещё полный 100-row run, а не smoke: offline mode лишь запрещает сетевую
синхронизацию W&B и оставляет локальные артефакты с context/results и config.

Не запускайте всю матрицу вслепую: это сетевой, compute-heavy и потенциально
платный процесс. Сначала проверьте один preset на малой выборке отдельным JSON
overlay, фиксируя endpoint model, corpus/index provenance и Git SHA.

## Параметры `RunConfig`

### Retrieval

| Поле | Default | Смысл |
|---|---:|---|
| `retriever_type` | `vanilla` | `vanilla`, `rerank` или `query_transform` |
| `top_k` | `5` | Чанков в итоговом context |
| `fetch_k` | `20` | Dense candidates до reranker; не используется в `vanilla` |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Query embedder; также передаётся RAGAS embeddings и должен быть совместим с индексом |
| `rerank_model` | `BAAI/bge-reranker-base` | Cross-encoder для `rerank` и `query_transform` |
| `chroma_path` | `data/chromadb` | Persistent Chroma directory |
| `chroma_collection` | `docs_fast` | Имя существующей коллекции |
| `device` | `auto` | `auto`, `cpu`, `cuda` или `mps` |

`query_transform` делает LLM rewrite и HyDE, затем rerank. Поэтому он добавляет
LLM calls, latency и стоимость относительно обычного dense retrieval.

### Generator

| Поле | Default | Смысл |
|---|---:|---|
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Model id или vLLM LoRA alias |
| `llm_api_url` | пусто | Обязательный OpenAI-compatible `/v1` base URL |
| `llm_api_key` | `EMPTY` | API key; сейчас небезопасно хранить реальное значение в config |
| `llm_temperature` | `0.1` | Sampling temperature |
| `llm_max_tokens` | `1024` | Максимум output tokens |
| `llm_timeout` | `60.0` | Timeout одного generator request, секунд |

Один и тот же client используется для RAG и pure baseline. При сравнении Base и
LoRA меняйте только `llm_model`/server state и фиксируйте остальные параметры.

### Judge и RAGAS

| Поле | Default | Смысл |
|---|---:|---|
| `judge_model` | `gpt-4o-mini` | Model id judge |
| `judge_api_url` | пусто | Если пусто, RAGAS фактически пропускается |
| `judge_api_key` | `EMPTY` | Доступен только через JSON; см. предупреждение о secrets |
| `compute_ragas` | `true` | Отключается флагом `--no-ragas` |

RAGAS считает `faithfulness`, `answer_relevancy` и `context_recall`. Judge,
совпадающий с generator или близкий к нему, создаёт self-evaluation bias; для
финальных выводов нужен отдельно зафиксированный judge и human calibration.

RAGAS выполняется в `./ragas_venv` относительно текущего рабочего каталога.
Код переиспользует исправное окружение, а неисправное удаляет и создаёт заново,
устанавливая сетевые зависимости без lock. Judge key передаётся subprocess через
argv и может быть виден в process list. Поэтому cloud-key path в текущем виде
небезопасен; используйте локальный endpoint с `EMPTY` либо сначала исправьте
secret handling.

До завершения run в `ragas_venv/temp_input.json` и `temp_output.json` остаются
plaintext question, gold, context и answers. Эти файлы не удаляются автоматически;
не направляйте RAGAS на приватные данные без отдельной политики хранения/очистки.

### Dataset и локальные метрики

| Поле | Default | Смысл |
|---|---:|---|
| `eval_csv_path` | `data/stackoverflow-pytorch.csv` | CSV с обязательными `question_body`, `answer_body`, `answer_score` |
| `eval_sample_size` | `100` | Размер deterministic sample |
| `eval_min_answer_score` | `1` | Нижняя граница StackOverflow answer score |
| `eval_seed` | `42` | Seed выборки |
| `eval_embedding_model` | `Snowflake/snowflake-arctic-embed-m` | Отдельная модель semantic metric |
| `compute_lexical` | `true` | SQuAD precision/recall/F1 против gold; `--no-lexical` |
| `compute_semantic` | `true` | Cosine между RAG и pure answers; `--no-semantic` |

`answer_similarity` не измеряет correctness относительно gold: это сходство
RAG-ответа и ответа без context. При ошибке отдельного RAG/pure call runner пишет
warning и подставляет пустой ответ, поэтому итоговые метрики нужно сопоставлять с
логами ошибок, а не интерпретировать изолированно.

Изменение `embedding_model` одновременно меняет retrieval и RAGAS embeddings —
это уже две оси эксперимента. RAGAS averages считают каждый metric по доступным
non-NaN значениям независимо; effective N не попадает в summary. При частичных
failures composite может смешать компоненты с разным coverage, поэтому сверяйте
NaN counts/per-row table и не сравнивайте только headline score.

### Composite score

```text
rag_score = 0.6 × faithfulness
          + 0.2 × answer_relevancy
          + 0.2 × context_recall
```

Текущие defaults дают приоритет groundedness. Исторический notebook/deck считал
`0.4/0.4/0.2`; сравнивать числа из разных схем напрямую нельзя. Код не проверяет,
что пользовательские веса нормированы. Для отчёта показывайте и исходные RAGAS
компоненты, и sensitivity analysis.

Пересчитать composite score по сохранённым логам без повторного eval:

```bash
uv run --locked --package eval-runner python eval-runner/scripts/recompute_score.py \
  --logs-dir eval-runner/logs
```

Скрипт ищет последнюю строку `eval done. summary: {...}` в каждом `*.log` и
сравнивает заранее заданные base/v2 пары по нескольким схемам весов.

## W&B и артефакты

CLI после eval безусловно вызывает `log_to_wandb` и записывает:

- все поля `RunConfig` через `asdict`;
- scalar summary;
- полную per-row table;
- строки с `faithfulness < 0.5`;
- случаи, где pure F1 превышает RAG F1 более чем на `0.1`.

`--wandb-api-key` относится только к авторизации W&B и не входит в `RunConfig`,
но значение видно в argv/shell history и `wandb.login` может сохранить локальную
авторизацию. Generator/judge keys входят в config. Не сохраняйте secrets в presets.

В репозитории уже отслеживаются `eval-runner/logs/*.log` и многочисленные файлы
`wandb/`; среди них могут быть payloads, machine paths и per-row context. Добавление
пути в `.gitignore` не удаляет уже tracked artifacts. Не копируйте их в новые
отчёты и не выполняйте history cleanup без отдельной согласованной задачи.

Runner не пишет отдельный canonical result JSON/CSV. Для programmatic использования
вызывайте `run_evaluation(cfg)`: он возвращает `EvalResult(config, per_row,
summary)` без W&B side effect; tracking вызывается отдельно только CLI-слоем.

## Правила корректного сравнения

Для каждой серии фиксируйте corpus и Chroma collection, embedding/chunk versions,
sample и seed, generator parameters, prompt, judge, Git SHA и endpoint model. В
одной серии меняйте одну ось. Не выдавайте один seed или self-judge score за
статистически устойчивый результат.

Текущий `eval-runner` вызывает legacy prompt path из `rag.chains.answer` и не
загружает `prompt_contract.json` LoRA-адаптера. Наличие контракта рядом с весами не
гарантирует его применение в eval; совместимость prompt нужно проверять вручную
или сначала интегрировать contract в runner.

## Проверка изменений

Отдельного test suite у `eval-runner` сейчас нет. Для безопасной локальной
проверки документации/config-кода используйте targeted lint/import checks; полный
eval требует Chroma data, моделей, внешнего LLM endpoint и W&B behavior.

Не считайте run успешным только по exit code: проверьте warnings, число реально
оценённых строк, наличие всех ожидаемых metric columns и provenance конфигурации.
