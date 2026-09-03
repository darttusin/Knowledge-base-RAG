# lora-pipeline — документы → synthetic dataset → LoRA adapter

`lora-pipeline` объединяет три независимо отключаемые стадии:

```text
docs/ → ingest в отдельную ChromaDB → teacher-generated JSONL → LoRA/QLoRA adapter
```

Pipeline предназначен для воспроизводимого исследовательского прогона в новом
`--output-dir`. Он не подключает получившийся adapter к backend автоматически и
не должен писать поверх tracked `data/`, `lora-train/runs/` или bundled ChromaDB.
Корневой `runs/` сейчас не игнорируется Git; используйте каталог вне репозитория
или сначала добавьте осознанное ignore-правило без переписывания tracked history.

## Требования и запуск

Команды выполняются из корня репозитория. Общий `uv.lock` требует Python 3.13.
Ingest/synth используют embedding и ML-зависимости пакета `rag`; первый запуск
может скачать модель. Полное обучение рассчитано на Linux/CUDA. На macOS
`bitsandbytes` training stack не поддерживается текущим lock.

Проверить CLI без запуска pipeline:

```bash
uv run --locked --package lora-pipeline python -m lora_pipeline --help
```

Для полного обучения добавьте optional training extra:

```bash
uv run --locked --package lora-pipeline --extra train \
  python -m lora_pipeline --help
```

`uv run` может дополнить общую `.venv` и скачать пакеты. Не используйте
`uv sync --all-packages --all-groups`: это устанавливает тяжёлый workspace stack.

## Критичное предупреждение о secrets

Сейчас `manifest.json` сериализует `PipelineConfig` через `asdict`, включая
`teacher_api_key` в открытом виде. Ключ, переданный через CLI, также может быть
виден в process list и shell history.

Поэтому текущий cloud-key workflow небезопасен. Не передавайте настоящий API key,
не коммитьте manifest и не запускайте платную генерацию, пока key не вынесен в
secret store/env и не исключён из сериализации. Безопасный документированный
вариант — локальный OpenAI-compatible endpoint с фиктивным значением `EMPTY`.

Teacher получает полный plaintext каждого отобранного документа/chunk. Даже при
безопасной передаче key внешний endpoint раскрывает ему corpus; приватные данные
отправляйте только в одобренный контур с подходящей retention policy.

## Входные документы

`--docs-dir` обязателен и проверяется даже при `--skip-ingest`. Поиск рекурсивный.
По умолчанию читаются только `.md`; список расширений задаётся через запятую:

```bash
--ext md,txt,rst
```

Путь исходного файла сохраняется в metadata чанка и затем в synthetic rows. Он
используется при группировке близких distractors и как source для sourced prompt,
поэтому осмысленная структура каталогов полезна.

Пример:

```text
my-docs/
├── api/
│   ├── client.md
│   └── server.md
└── guides/
    └── quickstart.md
```

Pipeline поддерживает текстовые файлы, которые может прочитать текущий loader. Он
не является PDF/DOCX parser.

## Рекомендуемый smoke с локальным teacher

Сначала запустите малую генерацию без обучения в новом output directory:

```bash
uv run --locked --package lora-pipeline python -m lora_pipeline \
  --docs-dir ./my-docs \
  --output-dir /tmp/rag-lora-smoke \
  --teacher-api-url http://127.0.0.1:8000/v1 \
  --teacher-api-key EMPTY \
  --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --max-chunks 20 \
  --skip-train
```

Если synth включён, preflight до ingest вызывает `models.list()` teacher endpoint.
Некоторые gateways не перечисляют все модели: отсутствие выбранной модели в
списке даёт warning, недоступный endpoint — ошибку. `--no-preflight` отключает
эту раннюю проверку.

`--max-chunks 20` ограничивает число исходных чанков, но итоговое число строк
зависит от teacher output, dedup и adversarial fraction. Это всё ещё сетевой и
потенциально платный вызов, если endpoint не локальный.

## Стадии

### 1. Ingest

Текущая последовательность:

1. рекурсивно загрузить файлы с выбранными extensions;
2. разбить их на чанки (`chunk_size=1000`, `chunk_overlap=200`);
3. создать embedding через `BAAI/bge-base-en-v1.5` по умолчанию;
4. записать чанки в `<output-dir>/chromadb`, collection `docs`.

Если collection уже непуста, обычный повторный запуск переиспользует её целиком и
не проверяет, соответствует ли она текущим docs/chunk/embed settings.
`--force-ingest` пересоздаёт collection и уничтожает прежний индекс в этом
output directory. Не направляйте его на общий или production Chroma path.

### 2. Synthetic dataset

Teacher по инструкции пытается генерировать grounded Q&A, затем pipeline:

- дедуплицирует нормализованные вопросы;
- формирует окно из gold chunk и близких/случайных distractors;
- перемешивает позицию gold chunk;
- добавляет adversarial rows без gold context и с отказом;
- делает seeded shuffle и train/validation split;
- сохраняет structured `chunks`, а не заранее отрендеренный prompt.

Предпочтительная строка имеет поля `question`, `answer`, `chunks`,
`is_adversarial`, `source`. `lora-train` также продолжает принимать legacy rows из
`dataset-prep` с плоским полем `context`.

Groundedness автоматически не проверяется: pipeline валидирует только parseable
JSON и непустые question/answer. Кроме того, concurrent `as_completed` меняет
порядок teacher results, temperature по умолчанию `0.7`, а endpoint может быть
недетерминированным. `seed` контролирует локальные distractor/shuffle/split
операции, но не гарантирует bitwise-identical dataset.

Если `dataset/train.jsonl` существует, synth переиспользует его без проверки
параметров и считает только число train rows. `--force-synth` генерирует dataset
заново и может перезаписать `train.jsonl`/`val.jsonl`.

### 3. Training

`lora-train` загружает base model, применяет PEFT LoRA/QLoRA, форматирует rows
через prompt contract и сохраняет adapter. Текущие основные defaults:

| Параметр | Default |
|---|---:|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Targets | строка `all-linear` |
| Epochs | `2` |
| Batch / accumulation | `1 / 16` |
| Learning rate | `2e-4` |
| Max sequence length | `4096` |
| Optimizer | `paged_adamw_8bit` |
| Gradient checkpointing | включён |
| Pipeline tracking | `report_to=none` |

Значение `none` — override именно pipeline. Standalone CLI
`python -m lora_train` по умолчанию использует `report_to=wandb`; для локального
запуска без внешнего tracking явно передайте `--report-to none`.

Loss сейчас считается по всей chat sequence, включая system/user tokens, а не
только по assistant answer. Результат — PEFT adapter и tokenizer files, не merged
base model.

Training не имеет resume/reuse guard: каждый запуск без `--skip-train` снова
вызывает trainer и пишет в тот же adapter output. `--force-ingest` и
`--force-synth` на это не влияют.

## Артефакты

После полного успешного запуска структура в общих чертах такая:

```text
<output-dir>/
├── chromadb/                 # отдельный индекс pipeline
├── dataset/
│   ├── train.jsonl
│   └── val.jsonl
├── adapter/
│   ├── run_config.json
│   └── final/                # PEFT adapter + tokenizer + contract
│       └── prompt_contract.json
├── prompt_contract.json
└── manifest.json
```

`manifest.json` пишется только в конце успешного orchestration и содержит stage
summaries, timestamps, contract и текущий config. Из-за описанной утечки key его
нельзя считать безопасным provenance artifact с реальными credentials. После
частичного сбоя manifest может отсутствовать или остаться от предыдущего запуска.

Защищайте и не коммитьте весь `<output-dir>`, а не только manifest: `chromadb/` и
JSONL содержат plaintext корпуса, chunks, ответы и source paths; adapter metadata
тоже может раскрывать сведения об эксперименте. `.gitignore` не удаляет файлы,
которые уже отслеживаются Git.

Всегда храните рядом с экспериментом seed, Git SHA, source corpus revision,
embedding/chunk settings, endpoint model id и prompt fingerprint — без secrets.

## Prompt contract

Доступны два встроенных контракта:

- `grounded` — context без обязательных numbered citations;
- `sourced` — numbered snippets и ссылки вида `[§N]`.

Можно передать путь к custom JSON через `--contract`. `--context-chunks` по
умолчанию равен `5`; он влияет и на synthetic context window, и на fingerprint
контракта, используемого training stage.

```python
from prompt_contract import PromptContract

contract = PromptContract.load("../rag-runs/my-lora/adapter/final")
print(contract.fingerprint())
```

Важное ограничение текущей интеграции: `rag.chains.answer(..., contract=contract)`
умеет применить contract только при явной передаче, но backend и `eval-runner`
сейчас его автоматически не загружают и не сверяют с adapter. Файл рядом с весами
сам по себе не гарантирует prompt compatibility в serving/eval.

## Разделение ноутбука и GPU-машины

На машине для подготовки данных:

```bash
uv run --locked --package lora-pipeline python -m lora_pipeline \
  --docs-dir ./my-docs \
  --output-dir ../rag-runs/my-lora \
  --teacher-api-url http://127.0.0.1:8000/v1 \
  --teacher-api-key EMPTY \
  --skip-train
```

После копирования **всего** output directory и доступной docs directory на
Linux/CUDA машину:

```bash
uv run --locked --package lora-pipeline --extra train \
  python -m lora_pipeline \
  --docs-dir ./my-docs \
  --output-dir ../rag-runs/my-lora \
  --skip-ingest \
  --skip-synth
```

При `--skip-synth` teacher URL не требуется. При `--skip-ingest --skip-synth`
pipeline всё равно требует существующий `--docs-dir` и проверяет наличие
`dataset/train.jsonl` перед training.

## Параметры управления

| Флаг | Default | Назначение |
|---|---:|---|
| `--context-chunks` | `5` | Чанков в synthetic/training context; согласуйте с serving `top_k` |
| `--qa-per-chunk` | `3` | Запрошенных Q&A на исходный chunk |
| `--max-chunks` | `0` | `0` означает все подходящие chunks |
| `--adversarial-fraction` | `0.20` | Доля добавляемых adversarial rows относительно normal rows |
| `--teacher-workers` | `8` | Параллельные teacher calls |
| `--seed` | `42` | Локальные distractor/shuffle/split; не делает teacher output детерминированным |
| `--qlora` | выключен | 4-bit base weights для уменьшения VRAM |
| `--trust-remote-code` | выключен | Разрешает исполнение model-repository Python code; только для доверенного pinned revision |
| `--report-to` | `none` | `none`, `wandb` или `tensorboard` для training |
| `--skip-*` | выключены | Явно пропустить отдельную стадию |
| `--force-ingest` | выключен | Пересоздать output collection |
| `--force-synth` | выключен | Перегенерировать output dataset |

Полный и актуальный список — в `python -m lora_pipeline --help` и
`lora_pipeline/__main__.py`.

## Deployment

Adapter можно объявить в vLLM под alias:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --enable-lora \
  --lora-modules my-lora=../rag-runs/my-lora/adapter/final
```

После этого backend/eval должны обращаться к alias `my-lora` и использовать тот
же prompt contract. Текущий compose vLLM не поднимает; endpoint настраивается
отдельно.

## Проверки

Unit tests для prompt contract, synth и orchestration:

```bash
uv run --locked --package lora-pipeline --with pytest \
  python -m pytest prompt-contract/tests dataset-synth/tests lora-pipeline/tests -q
```

Эти tests не подтверждают реальный teacher endpoint, скачивание embedder,
качество synthetic rows, CUDA training или serving compatibility. Для них нужен
отдельный согласованный smoke в новом output directory; paid API/GPU/W&B запуск
без явной задачи не выполняйте.
