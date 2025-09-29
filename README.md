# Knowledge Base RAG (PyTorch Docs)

> Вопросно‑ответная система по документации PyTorch на базе Retrieval‑Augmented Generation (RAG) с воспроизводимыми бенчмарками, понятными метриками и готовым интерфейсом (Telegram‑бот).

## Цели и KPI

* Ответы на вопросы по PyTorch с опорой на первоисточник (ссылки на документацию обязательны в каждом ответе)
* Базовый QoS (p95 < 2.5s при k=20, n_ctx≈2–3к токенов)
* Качество: Recall@20 ≥ 0.85 для retrieval, Faithfulness ≥ 0.80 по RAGAS для e2e
* Простая интеграция через REST + Telegram‑бот

## Архитектура (выбор по умолчанию)

* **Корпус**: `docs.pytorch.org` (стабильные версии и latest)
* **Парсинг**: sitemap → HTML → Markdown; сохранение исходного URL, заголовков H1–H3, код‑блоков, версий
* **Чанкинг**: по Markdown‑структуре, таргет длины 600±150 токенов, overlap 15%; объединяем код‑блок + текст
* **Лексический baseline**: BM25 (Pyserini/Lucene)
* **Денс‑эмбеддинги**: `BAAI/bge-small-en-v1.5` (384‑d) как дефолт; опционально `BGE-M3` для многоязычия/мульти‑режимов
* **Векторный индекс**: FAISS: `FlatIP` ≤100k чанков; при росте `IVF,HNSW`/`IVF,PQ` с автотюнингом
* **Hybrid retrieval**: RRF(BM25, Dense k=100 → topk=40)
* **Rerank**: `BAAI/bge-reranker-v2-m3` (top40→top8)
* **Генерация**: small instruct LLM (локальный 7–8B) с ответами в стиле “cite‑and‑answer”, строгие цитаты на чанки
* **Промпт‑политики**: краткий System‑prompt, формат ответа, ограничения; offline APO на dev‑наборах
* **Логирование/эксперименты**: W&B/MLflow; Hydra‑конфиги

Диаграмма потока:

```
Query → Preprocess → Retrieve: [BM25 ∥ Dense] → RRF → Re‑rank → Compose Context → LLM → Post‑process (citations, safety)
```

## Датасет и EDA

* Источник: документация PyTorch (версии 2.x), вкл. subsections (torch, torchvision, torchtext при необходимости)
* EDA: частоты терминов, n‑граммы, coverage по разделам, облако слов по модулям; выявление дубликатов/редиректов
* Аннотация QA:

  * Автоматически: сэмплинг параграфов → генерация вопросов (query2doc/HyDE) → верификация
  * Ручные проверки на 200–300 примерах для dev/test

## Метрики

**Retrieval:** Recall@k, nDCG@k, MRR@k (dev/test).
**E2E (RAGAS):** faithfulness, answer‑relevancy, context‑precision/recall.
**Вспомогательно:** LLM‑as‑a‑Judge на сложных кейсах; latency/QPS.

## Майлстоуны (осень 2025 → весна 2026)

* **M0 (до 10 окт):** каркас репо, парсер, чанкинг, индексация, базовые CLI/Make
* **M1 (окт):** BM25 + BGE‑small baselines, RRF; первичный eval
* **M2 (ноя):** Reranker, промпт‑политики, строгие цитаты; повышение Recall@20 ≥0.85
* **M3 (дек):** APO на промптах, HyDE/doc2query; отчёт по влиянию k, overlap, top‑p
* **M4 (янв):** Telegram‑бот + REST; load‑тест p95 < 2.5s
* **M5 (фев–мар):** Контрастивный fine‑tune эмбеддера/реранкера на своём QA; A/B против M2
* **M6 (апр):** Русская локаль (перевод ответов), hard‑negatives, guardrails
* **M7 (май–июн):** шлифовка, документация, постер/статья

## Задачи (детально)

### 1) Инжест

* Краулер по sitemap/TOC; нормализация ссылок; сжатие; детект языков/версий
* Очистка HTML: оставляем тексты, списки, таблицы, код; выносим “Note/Warning”
* Чанкер: markdown‑осознанный, soft‑wrap по заголовкам и токенам; метаданные: `url, hpath, version, hash`

### 2) Индексация и хранение

* FAISS + parquet/duckdb для метаданных
* Снапшоты индекса (dvc/git‑lfs)

### 3) Retrieval

* BM25 (k1=0.9, b=0.4) как baseline
* Dense: BGE‑small, нормализация (`L2`), Inner Product
* Fusion: RRF(weighted), k_dense=100, k_bm25=100 → top40
* Аналитика влияния k/overlap/фильтров

### 4) Rerank

* `bge-reranker-v2-m3` pairwise; порог отсечения; эвристики по дубликатам/версии

### 5) Генерация

* Формат ответа: кратко, затем пункты с цитатами `[§]` → URL#anchor
* Стратегии: evidence‑first, max marginal relevance для контекста
* Ограничения: запрет галлюцинаций вне контекста; если нет контекста — отказ + “где смотреть”

### 6) Оценка

* **Retrieval**: Recall@5/10/20, nDCG@10; диаграммы чувствительности
* **E2E**: RAGAS(faithfulness, answer‑relevancy, context‑precision/recall); сэмпл ручной валидации; опционально LLM‑as‑a‑Judge
* Трекинг в W&B/MLflow; фиксация сидов, версий корпусов

### 7) Улучшения

* Query‑enhancement: HyDE; docTTTTTquery для лексики; SPLADE/ColBERT как эксперимент
* Prompt‑tuning: offline APO на dev‑наборе
* Hybrid dens+sparse (BGE‑M3 lexical matching) как альтернатива BM25

### 8) Сервис и интерфейс

* REST: `/ask` (query, top_k, lang), `/feedback`, `/health`
* Telegram‑бот: синхронный поток, троттлинг, i18n, короткие ссылки на оригинал
* Observability: structured logs, latency, hit‑rate (контекст использован?)

## Репозиторий

```
├── configs/                # Hydra
├── data/{raw,processed}/
├── scripts/{crawl,ingest,index,eval}.py
├── src/
│   ├── ingest/
│   ├── retrievers/{bm25,dense,hybrid}.py
│   ├── rerank/
│   ├── generate/
│   ├── eval/{retrieval,ragas,judge}.py
│   └── service/{api,telegram}/
├── notebooks/
├── Makefile
└── README.md
```

## Быстрый старт

```
make venv && make deps
python scripts/crawl.py --site https://docs.pytorch.org
python scripts/ingest.py --chunk 600 --overlap 90
python scripts/index.py --faiss flatip
python scripts/eval_retrieval.py --k 20
python scripts/serve.py  # REST
python scripts/telegram_bot.py  # бот
```

## Definition of Done (итерация)

* [ ] Парсер + чанкинг воспроизводим
* [ ] Индексы FAISS собраны, снапшоты сохранены
* [ ] Baselines: BM25, Dense, Hybrid измерены на dev/test
* [ ] Reranker подключён; прирост nDCG@10 и faithfulness задокументирован
* [ ] Telegram‑бот работает; ответы с цитатами
* [ ] Отчёт с графиками (retrieval, e2e, latency)

## Дорожная карта (+идеи)

* Контрастивный дообучение эмбеддера/реранкера на собственном QA
* Версионирование ответов (разные релизы PyTorch)
* Мультиязычие (BGE‑M3) и автоперевод русских ответов
* Hard‑negative mining; ColBERT/SPLADE эксперимент
