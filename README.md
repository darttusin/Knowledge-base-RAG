# Knowledge Base RAG (PyTorch Docs)

> Вопросно‑ответная система по документации PyTorch на базе Retrieval‑Augmented Generation (RAG) с воспроизводимыми бенчмарками, понятными метриками и готовым интерфейсом (Telegram‑бот).

## Цели и KPI

* Ответы на вопросы по PyTorch с опорой на первоисточник (ссылки на документацию обязательны в каждом ответе)
* Простая интеграция через REST + Telegram‑бот

Диаграмма потока:

```
Query → Preprocess → Retrieve: [BM25 ∥ Dense] → RRF → Re‑rank → Compose Context → LLM → Post‑process (citations, safety)
```

## Датасет и EDA

### Stackoverflow QA's dataset

* Link - https://console.cloud.google.com/marketplace/product/stack-exchange/stack-overflow
* SQL query:
```sql
DECLARE keywords ARRAY<STRING> DEFAULT [
  'pytorch','pytorch-lightning','torchvision','torchaudio','torchtext',
  'libtorch','torchscript','torchserve','pytorch-ignite','pytorch-geometric',
  'pytorch3d','torchdata','pytorch-dataloader','pytorch-datapipe',
  'gpytorch','skorch','pytorch-forecasting','torchmetrics','torchrl','torch'
];

WITH q AS (
  SELECT id, title, body, tags, accepted_answer_id
  FROM `bigquery-public-data.stackoverflow.posts_questions`
),
a AS (
  SELECT parent_id, id, body, score, creation_date
  FROM `bigquery-public-data.stackoverflow.posts_answers`
),
q_tags AS (
  SELECT id, LOWER(tag) AS tag
  FROM q, UNNEST(REGEXP_EXTRACT_ALL(q.tags, '<([^>]+)>')) AS tag
)
SELECT
  q.body AS question_body,
  a.body AS answer_body,
  a.score AS answer_score
FROM q
LEFT JOIN a
  ON a.parent_id = q.id
LEFT JOIN q_tags t
  ON t.id = q.id
WHERE (
  t.tag IN UNNEST(keywords)
  OR EXISTS (
    SELECT 1
    FROM UNNEST(keywords) AS kw
    WHERE REGEXP_CONTAINS(LOWER(q.title), '(^|[^a-z0-9_])' || kw || '([^a-z0-9_]|$)')
       OR REGEXP_CONTAINS(LOWER(q.body),  '(^|[^a-z0-9_])' || kw || '([^a-z0-9_]|$)')
  )
  OR REGEXP_CONTAINS(LOWER(q.title), r'\bpytorch\b')
  OR REGEXP_CONTAINS(LOWER(q.body),  r'\b(import\s+torch|from\s+torch\s+import|torch\.(nn|cuda|optim|jit|compile|tensor|utils\.data|no_grad|manual_seed))\b')
  OR REGEXP_CONTAINS(q.body,         r'\b(torch::|at::)\b')
  OR REGEXP_CONTAINS(LOWER(a.body),  r'\b(import\s+torch|from\s+torch\s+import|torch\.(nn|cuda|optim|jit|compile|tensor|utils\.data|no_grad|manual_seed)|torch::|at::)\b')
)
AND NOT REGEXP_CONTAINS(LOWER(q.body), r'\brequire\s+[\'"]torch[\'"]')
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY q.id
    ORDER BY IF(a.id = q.accepted_answer_id, 1, 0) DESC, a.score DESC, a.creation_date ASC
  ) = 1;
```

* Источник: документация PyTorch (версии 2.x), вкл. subsections (torch, torchvision, torchtext при необходимости)
* EDA: частоты терминов, n‑граммы, coverage по разделам, облако слов по модулям; выявление дубликатов/редиректов
* Аннотация QA:
  * Автоматически: сэмплинг параграфов → генерация вопросов (query2doc/HyDE) → верификация
  * Ручные проверки на 200–300 примерах для dev/test
* Разбиение текста на части и токенизация

## Майлстоуны

* **M0 (до 20 окт):** каркас репо, парсер, чанкинг, индексация, базовые CLI/Make
* **M1 (ноябрь):** Baseline
* **M2 (дек):** Telegram‑бот + REST

## Оценка качества RAG

* LLM as a judge
* Response latency
* Precision@K
* Recall@K
* MRR
* BLEU

## Задачи (детально)

### 1) Инжест

* Краулер по sitemap/TOC; нормализация ссылок; сжатие; детект языков/версий
* Очистка HTML: оставляем тексты, списки, таблицы, код; выносим “Note/Warning”
* Чанкер: markdown‑осознанный, soft‑wrap по заголовкам и токенам; метаданные: `url, hpath, version, hash`

### 2) Индексация и хранение

* FAISS + бд для метаданных
* Снапшоты индекса

### 3) Retrieval

* BM25
* Dense
* Fusion

### 4) Rerank

* `bge-reranker-v2-m3` pairwise; порог отсечения; эвристики по дубликатам/версии

### 5) Генерация

* Формат ответа: кратко, затем пункты с цитатами `[§]` → URL#anchor
* Ограничения: запрет галлюцинаций вне контекста; если нет контекста — отказ + “где смотреть”

### 6) Улучшения

* Query‑enhancement: HyDE; docTTTTTquery для лексики; SPLADE/ColBERT как эксперимент
* Prompt‑tuning: offline APO на dev‑наборе. LLM as judge
* Hybrid dens+sparse (BGE‑M3 lexical matching) как альтернатива BM25

### 7) Сервис и интерфейс

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
