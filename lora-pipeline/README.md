# lora-pipeline — своя документация → обученный LoRA-адаптер

Одна команда проходит весь путь: берёт папку с документами, строит по ним
поисковый индекс, генерирует по нему grounded-датасет teacher-моделью и
обучает LoRA поверх выбранной базовой модели.

```
docs/ ──ingest──▶ ChromaDB ──synth──▶ train.jsonl ──train──▶ adapter/final/
```

## 1. Установка

```bash
git clone <репозиторий>
cd RAG
uv sync --package lora-pipeline                  # индекс + генерация датасета
uv sync --package lora-pipeline --extra train    # то же + стек обучения (CUDA)
```

Стадии `ingest` и `synth` работают на ноутбуке и не требуют torch. Стадия
`train` ставится отдельным extra и требует CUDA-машины: `lora-train` тянет
`bitsandbytes`, которого нет под macOS. Схема «датасет на ноутбуке,
обучение на GPU-боксе» описана в разделе 6.

## 2. Куда класть документы

Любая папка с текстовыми файлами, вложенность произвольная:

```
my-docs/
├── api/
│   ├── client.md
│   └── server.md
└── guides/
    └── quickstart.md
```

По умолчанию берутся `.md`. Другие расширения — через `--ext`:

```bash
--ext md,txt,rst
```

Требований к структуре нет: путь к файлу попадает в метаданные чанка и
используется как источник в цитатах и как группа при подборе дистракторов,
поэтому осмысленные имена папок улучшают качество датасета.

## 3. Запуск

Teacher через API (OpenAI или любой совместимый шлюз):

```bash
uv run python -m lora_pipeline \
    --docs-dir ./my-docs \
    --output-dir runs/my-lora \
    --teacher-api-url https://api.openai.com/v1 \
    --teacher-api-key sk-... \
    --teacher-model gpt-4o-mini
```

Teacher локально через vLLM — тот же интерфейс, меняется только URL:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8000

uv run python -m lora_pipeline \
    --docs-dir ./my-docs \
    --output-dir runs/my-lora \
    --teacher-api-url http://localhost:8000/v1 \
    --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ
```

**Сначала прогоните смоук на 20 чанках** — увидите качество пар и оцените
стоимость полного прогона:

```bash
uv run python -m lora_pipeline ... --max-chunks 20 --skip-train
```

## 4. Что получается

Всё складывается в `--output-dir`:

```
runs/my-lora/
├── chromadb/               # поисковый индекс по вашим документам
├── dataset/
│   ├── train.jsonl         # обучающие примеры
│   └── val.jsonl
├── adapter/
│   └── final/              # ← адаптер: сюда указывать vLLM
│       ├── adapter_model.safetensors
│       └── prompt_contract.json
├── prompt_contract.json    # формат промпта этого прогона
└── manifest.json           # чем, из чего и с какими параметрами получено
```

Деплой адаптера:

```bash
vllm serve <базовая-модель> --enable-lora --lora-modules my-lora=runs/my-lora/adapter/final
```

## 5. Контракт промпта — почему это важно

Адаптер валиден **только** в том формате промпта, в котором обучался.
Поменяли системный промпт, порядок «контекст → вопрос» или способ подачи
чанков — модель решает задачу, которой не видела, и деградирует молча:
ничего не падает, просто метрики хуже.

Поэтому формат вынесен в отдельный объект (`prompt-contract`), общий для
генерации данных, обучения, сервинга и оценки. Он сохраняется рядом с
весами и имеет отпечаток:

```python
from prompt_contract import PromptContract
contract = PromptContract.load("runs/my-lora/adapter/final")
contract.fingerprint()   # '3f2a...' — сверяйте при сервинге
```

На инференсе передавайте его явно:

```python
from rag.chains import answer
answer(llm, question, chunks, contract=contract)
```

Встроенных контракта два: `grounded` (по умолчанию — чистый контекст) и
`sourced` (нумерованные чанки с источниками и обязательными цитатами
`[§N]`). Свой — JSON-файл, путь передаётся в `--contract`.

`--context-chunks` должен совпадать с `top_k` вашего ретривера: если
модель училась на 5 чанках, а на инференсе получает 20, распределение
входа другое.

## 6. Разнести генерацию и обучение

Стадии переиспользуют артефакты, поэтому прогон возобновляется:

```bash
# на ноутбуке — только датасет
uv run python -m lora_pipeline --docs-dir ./my-docs --output-dir runs/my-lora \
    --teacher-api-url ... --skip-train

# скопировать runs/my-lora на GPU-бокс, там — только обучение
uv run python -m lora_pipeline --docs-dir ./my-docs --output-dir runs/my-lora \
    --skip-ingest --skip-synth
```

Повторный запуск ничего не пересчитывает: индекс и датасет
переиспользуются. Пересобрать принудительно — `--force-ingest`,
`--force-synth`.

## 7. Основные параметры

| Флаг | Дефолт | Зачем |
|---|---|---|
| `--context-chunks` | 5 | чанков в одном примере; ставьте равным `top_k` ретривера |
| `--qa-per-chunk` | 3 | пар с одного чанка — прямо влияет на размер датасета и стоимость |
| `--max-chunks` | 0 (все) | ограничение для смоук-прогона |
| `--adversarial-fraction` | 0.20 | доля примеров с нерелевантным контекстом и ответом-отказом |
| `--base-model` | Qwen2.5-Coder-7B-Instruct | что дообучаем |
| `--lora-targets` | `all-linear` | работает на любой архитектуре; можно список модулей через запятую |
| `--qlora` | выкл. | базовые веса в 4 бита, если не хватает VRAM |
| `--embedding-model` | BAAI/bge-base-en-v1.5 | эмбеддер индекса |
| `--report-to` | none | `wandb` для логирования обучения |

Полный список — `--help`.

## 8. Если что-то не так

**`no documents found under ...`** — не тот `--ext` или не та папка;
поиск рекурсивный, проверьте расширения файлов.

**`teacher endpoint unreachable`** — preflight проверяет teacher до
индексации, чтобы не потратить час на эмбеддинги впустую. Проверьте URL
(он должен заканчиваться на `/v1`) и ключ.

**Индексация «ничего не делает»** — коллекция уже непуста и
переиспользуется намеренно. Другой набор документов → `--force-ingest`,
иначе два корпуса смешаются в одном индексе.

**Мало примеров на выходе** — teacher возвращает меньше пар на коротких
чанках. Поднимите `--qa-per-chunk`, снизьте `--chunk-size` или ослабьте
фильтр `min_chunk_chars`.

## 9. Тесты

```bash
uv run --with pytest python -m pytest prompt-contract/tests dataset-synth/tests lora-pipeline/tests -q
```
