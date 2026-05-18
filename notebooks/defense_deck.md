---
marp: true
theme: default
paginate: true
---

# RAG для приватной базы знаний

## Опыт fine-tune Qwen2.5-Coder-7B на PyTorch Q&A

**Источник знаний:** документация PyTorch 2.x (7722 чанков в ChromaDB) + 24k Q&A со StackOverflow

**Стек:** vLLM на vast.ai, LoRA r=16, RAGAS + lexical/semantic метрики, wandb

**Главный вывод:** *LoRA не улучшила RAG-систему* — `base-vanilla` остался лучшим (rag_score 0.55 vs 0.29 у LoRA). Discussion и lessons learned ниже.

---

# 1 — Архитектура проекта

```
┌────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ dataset-prep/  │ → │ lora-train/  │ → │   vLLM на    │ ← │ eval-runner/│
│ HTML→md, фильтр│   │ LoRA r=16 на │   │   vast.ai    │   │ метрики +   │
│ RAG-aware Q&A  │   │ Qwen2.5-7B   │   │ base + LoRA  │   │ wandb log   │
└────────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
        ↑                                                          │
        │                                                          ↓
┌────────────────┐                                       ┌──────────────────┐
│ ChromaDB index │ ────── retrieval ──────────────────→ │ notebooks/       │
│ docs_fast 7722 │                                       │ eval_comparison  │
└────────────────┘                                       └──────────────────┘
```

**5 workspace-модулей:** `dataset-prep`, `lora-train`, `eval-runner` (новые) + `rag`, `outlier-detection` (исходные)

**Все настройки выносятся в `RunConfig` → `wandb.config`** — каждый прогон в wandb знает свой контекст полностью

---

# 2 — Подготовка датасета (RAG-aware, не plain SFT)

**Источник:** 24 287 Q&A пар со StackOverflow по PyTorch

**Pipeline:** `dataset-prep/dataset_prep/pipeline.py`
1. HTML → markdown (сохранение fenced code blocks)
2. Фильтрация: score ≥ 5, длина 50–4000 / 100–6000 chars
3. Дедупликация по нормализованному вопросу
4. **Retrieval per-question:** top-5 чанков из ChromaDB → `context` field
5. **+ 15% adversarial:** неподходящий контекст + refusal-ответ
6. Stratified train/val split по score

**Результат:** 1796 train + 95 val примеров с полями `{question, answer, context, is_adversarial}`

⚠️ **Малый объём (1.8k)** — критично для далнейших выводов

---

# 3 — LoRA training

**Модель:** `Qwen/Qwen2.5-Coder-7B-Instruct` (без QLoRA, bf16)

**LoRA конфиг:**
- `r=16, alpha=32, dropout=0.05`
- target = `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 40 370 176 trainable params (**0.53% от 7.66B**)

**Training:**
- 2 эпохи × batch=1 × grad_accum=16 → 226 шагов
- `paged_adamw_8bit`, lr=2e-4 cosine, warmup 3%
- bf16 + gradient checkpointing
- A6000 48GB, ~95 минут

**Артефакт:** `lora-train/runs/qwen25-coder-7b-lora-r16-v1/final/` — 154 MB adapter

---

# 4 — Матрица экспериментов (12 runs)

| Phase | Runs | Что измеряет |
|---|---|---|
| **1 — Story-line** | base-vanilla, base-rerank, base-qt, **lora-rerank** | Эффект каждого компонента |
| **2 — Sensitivity** | top_k ∈ {3,5,10}, fetch_k ∈ {10,20,50}, temp ∈ {0, 0.1} | Чувствительность к гиперпараметрам |
| **3 — Stability** | lora-rerank-k5 × seeds {42, 43, 44} | Дисперсия по сэмплированию |

**Метрики:**
- **Lexical:** SQuAD F1 / precision / recall — RAG-ответ vs gold
- **Semantic:** cosine(rag_answer, pure_answer) через Snowflake-arctic-embed
- **RAGAS:** faithfulness, answer_relevancy, context_recall — judge = Qwen 7B Coder
- **Composite:** `rag_score = 0.4·F + 0.4·AR + 0.2·CR`
- **Latency:** p50/p95 в секундах

---

# 5 — Headline result: LoRA сделала систему хуже

| Метрика | base-rerank-k5 | lora-rerank-k5 | Δ |
|---|---|---|---|
| **rag_score** | **0.530** | 0.293 | **−45%** |
| lexical/rag_f1 | 0.257 | 0.105 | −59% |
| ragas/answer_relevancy | 0.767 | 0.156 | **−80%** |
| answer_similarity | 0.886 | 0.562 | −37% |
| ragas/faithfulness | 0.248 | 0.276 | +11% |
| ragas/context_recall | 0.621 | 0.601 | −3% |
| rag_better vs pure (out of 100) | 69 | 14 | −80% |

**Только faithfulness вырос (в пределах seed-шума, см. слайд 7).** Все остальное обвалилось.

**Лучший конфиг overall = `base-vanilla-k5`** (самый простой!) с rag_score = 0.548

---

# 6 — Phase 1 в один график

```
                          rag_score    faithfulness  answer_rel
base-vanilla-k5    │██████████████ 0.548  ███████ 0.303  ████████████████████ 0.757
base-rerank-k5     │█████████████  0.530  ██████  0.248  ████████████████████ 0.767
base-qt-k5         │████████████   0.507  █████   0.205  ███████████████████  0.745
lora-rerank-k5     │███████        0.293  ███████ 0.276  ████                 0.156
```

**Наблюдения:**

- **Reranker и Query Transform** на base **не помогли**: rag_score 0.55 → 0.53 → 0.51 (в пределах шума)
- **QT** добавил +66% к латентности (8.3s → 13.8s p95) — невыгодно
- **LoRA** обвалила answer_relevancy в 5× — judge не понимает её лаконичные SO-style ответы

---

# 7 — Phase 3: дисперсия по seed огромная

`lora-rerank-k5` × seed ∈ {42, 43, 44} (одинаковый конфиг):

| Метрика | mean | std | CV% |
|---|---|---|---|
| rag_score | 0.282 | 0.016 | 5.6% |
| **faithfulness** | 0.230 | **0.041** | **18%** |
| **answer_relevancy** | 0.166 | **0.030** | **18%** |
| context_recall | 0.615 | 0.022 | 3.6% |

**Что это значит:**

- ✅ **base vs LoRA** (Δrag_score=0.24) — z-score ≈ 15 → **точно реальный эффект**
- ❌ **top_k sweep** (Δ=0.04) — внутри 1σ → нельзя сказать «k=5 лучше k=3»
- ❌ **fetch_k sweep** (Δ=0.02) — чистый шум
- ❌ **temperature** (Δ=0.025) — чистый шум

**Lesson:** 100 примеров недостаточно для micro-сравнений. Нужно ≥3 seeds на каждый конфиг или 500+ samples.

---

# 8 — Почему LoRA провалилась — root cause

### A. Стилистический сдвиг (наиболее вероятная причина)

LoRA выучила SO-стиль: краткий ответ + минимум кода. Из smoke-test:

> LoRA: «I found this solution: `if torch.cuda.is_available(): # do something`»
> base: длинный ответ с пояснениями, fallback на CPU, комментарии, ссылки

Эталонные SO-ответы **многословны** → lexical F1 механически падает, даже когда LoRA-ответ корректный.

### B. Judge bias (self-evaluation)

Судья = Qwen 7B Coder = та же база, что под LoRA-адаптером. Видит лаконичные ответы «не своим стилем» и занижает answer_relevancy. Нужен **независимый judge** (GPT-4o-mini) для честной оценки.

### C. Малый объём (1.8k) + неправильно дозированный adversarial (15% = 270 примеров)

Smoke-test тест 2 (irrelevant context): LoRA **не отказалась**, а ответила из памяти. Adversarial-сигнал утоп в нормальных примерах. Нужно ≥30% adversarial и/или 5k+ датасет.

### D. LoRA capacity r=16 могла недохватить

40M trainable из 7.66B = 0.5%. Для смены распределения ответов могло быть мало. Стоит попробовать r=32 или r=64.

---

# 9 — Lessons learned + продакшен-вывод

### Продакшен

✅ **Deploy: `base-vanilla-k5`** (rag_score=0.548, p95=8.3s)
❌ **Откатить LoRA-адаптер** — пока что регрессия
❌ **Убрать query-transform** — +66% латентность без выигрыша
🟡 **Reranker оставить опционально** — нейтрально по качеству, +0.1s

### Методологические уроки

1. **Eval-pipeline сам по себе ценный артефакт** — без него LoRA-регрессию мы не поймали бы в проде
2. **Seed-stability перед выводами** — single-shot eval мог соврать в обе стороны
3. **Self-judge bias реальный** — для финальных метрик нужен независимый judge
4. **Negative result — это тоже результат** — методология работает, даже если гипотеза не подтвердилась

### Future work

- Расширить датасет до 5–10k (снизить `--min-score` до 3)
- Adversarial-fraction 30%, разнообразить refusal-фразы
- LoRA r=32 или r=64
- Eval с GPT-4o-mini как judge
- Human eval на N=20 для калибровки RAGAS

---

# 10 — Бюджет и воспроизводимость

**vast.ai:**
- LoRA training: ~$0.50 (PRO 6000 96GB × 95 мин по $0.30/час)
- Vlllm для eval: ~$1.50 (PRO 6000 × 3 часа на матрицу)
- **Итого: ~$2** за всю серию экспериментов

**Wandb:** все 12 runs в проекте `pytorch-rag-eval` — конфиг + метрики + per-row tables (со списком hallucinations и rag-lost-to-baseline)

**Git:** одна ветка `main`, все коммиты атомарные:
- `feat(dataset-prep): ...`
- `feat(lora-train): ...`
- `feat(eval-runner, rag): layered configs, 12 presets, RAGAS venv reuse`
- `fix(eval-runner): hydrate run config via api.run()`

**Воспроизводимость:** `eval-runner/configs/*.json` + один bash-цикл из README → полная матрица повторяется за 3 часа

---

# Итог

| Что измерили | Что узнали |
|---|---|
| **Гипотеза:** LoRA на RAG-aware SO Q&A улучшит faithfulness | Не подтвердилась |
| **Headline:** rag_score baseline=0.55, LoRA=0.29 | LoRA = регрессия в 1.9× |
| **Главный вопрос:** какой компонент даёт выигрыш? | Никакой из протестированных — простой vanilla retriever выигрывает |
| **Что работает в pipeline:** | dataset-prep (1796 чистых пар), eval-runner (12 reproducible runs), wandb tracking |
| **Что НЕ работает:** | LoRA-конфиг как обучен. Query-transform. Self-judge для RAGAS |
| **Самый ценный артефакт:** | Eval-инфраструктура — позволяет быстро итерировать **и не deploy'ить регрессию вслепую** |

**Спасибо.**
