---
marp: true
theme: default
paginate: true
---

# RAG для приватной базы знаний

## Опыт fine-tune Qwen2.5-Coder-7B на PyTorch Q&A

**Исторический отчёт · 19 мая 2026 года**

**Источник знаний:** документация PyTorch 2.x (7722 чанков в ChromaDB) + 24k Q&A со StackOverflow

**Главное наблюдение:** в сохранённых runs LoRA-конфигурация получила более низкие
метрики; `base-vanilla` имел legacy rag_score 0.55 против 0.29 у LoRA. Статистическая
значимость этого сравнения не оценивалась.

---

# Как читать этот отчёт

Это снимок конкретной исследовательской серии, а не руководство по текущему
runtime или production policy.

**Стек серии:** vLLM на vast.ai, LoRA r=16, RAGAS + lexical/semantic
метрики, W&B.

Все `rag_score` ниже сохранены в исходной схеме весов `0.4/0.4/0.2`.
Текущий `eval-runner` по умолчанию использует `0.6/0.2/0.2`, поэтому новые и
старые headline scores нельзя сравнивать без пересчёта компонентов.

Актуальные setup, ограничения и security notes находятся в `../README.md` и
`../eval-runner/README.md`. Текущий W&B path сериализует keys и per-row context —
реальные secrets и приватные данные через него передавать нельзя.

---

# Provenance исторических чисел

**Evidence snapshot:** Git `691e3f2` — 11 matrix logs и 13 W&B run directories
(11 matrix + 2 sanity runs).

Три текущих `eval-runner/logs/base-*.log` позже заменены в `e408c6d`; их scores
уже не совпадают с таблицами ниже. Для проверки исходных чисел нужен snapshot
`691e3f2`, а не текущий HEAD.

Точные remote model revisions и полный mapping W&B run IDs в deck не сохранены.
LoRA adapter также отсутствует в Git snapshot, поэтому это архив результатов, а
не самодостаточный reproduction bundle.

---

# 1 — Архитектура экспериментального контура

```
┌────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ dataset-prep/  │ → │ lora-train/  │ → │   vLLM на    │ ← │ eval-runner/│
│ SO Q&A, фильтр │   │ LoRA r=16 на │   │   vast.ai    │   │ метрики +   │
│ + RAG context  │   │ Qwen2.5-7B   │   │ base + LoRA  │   │ W&B log     │
└────────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
        ↑                                                          │
        │                                                          ↓
┌────────────────┐                                       ┌──────────────────┐
│ ChromaDB index │ ────── retrieval ──────────────────→ │ notebooks/       │
│ docs_fast 7722 │                                       │ eval_comparison  │
└────────────────┘                                       └──────────────────┘
```

**Контур серии:** `dataset-prep`, `lora-train`, `eval-runner` и `rag`. Основные
eval knobs сохранялись через `RunConfig`, но prompt и часть RAGAS behavior
оставались в коде. Текущий root workspace содержит 12 packages.

---

# 2 — Подготовка датасета (RAG-aware, не plain SFT)

**Источник:** 24 287 Q&A пар со StackOverflow по PyTorch

**Pipeline:** `dataset-prep/dataset_prep/pipeline.py`
1. HTML → markdown (сохранение fenced code blocks)
2. Фильтрация: score ≥ 5, длина 50–4000 / 100–6000 chars
3. Дедупликация по нормализованному вопросу
4. **Retrieval per-question:** top-5 чанков из ChromaDB → `context` field
5. **+ 15% к normal rows adversarial:** неподходящий context + refusal-ответ
6. Stratified train/val split по score

**Результат:** 1891 row = 1645 normal + 246 adversarial (13.0% итогового набора);
split — 1796 train + 95 val.

⚠️ **Малый объём (1.8k)** — критично для дальнейших выводов

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
- ~95 минут; исходные заметки расходятся по GPU (A6000 48GB vs PRO 6000 96GB)

**Артефакт:** в отчёте был указан 154 MB adapter по пути
`lora-train/runs/qwen25-coder-7b-lora-r16-v1/final/`, но в Git snapshot его нет.

---

# 4 — Историческая матрица экспериментов (11 unique runs)

| Phase | Runs | Что измеряет |
|---|---|---|
| **1 — Story-line** | base-vanilla, base-rerank, base-qt, **lora-rerank** | Эффект каждого компонента |
| **2 — Sensitivity** | top_k ∈ {3,5,10}, fetch_k ∈ {10,20,50}, temp ∈ {0, 0.1} | Чувствительность к гиперпараметрам |
| **3 — Stability** | lora-rerank-k5 × seeds {42, 43, 44} | Дисперсия по сэмплированию |

Значения по умолчанию входят сразу в несколько sweeps; поэтому сумма вариантов
по строкам таблицы больше числа уникальных run names.

**Метрики:**
- **Lexical:** SQuAD F1 / precision / recall — RAG-ответ vs gold
- **Semantic:** cosine(rag_answer, pure_answer) через Snowflake-arctic-embed
- **RAGAS:** faithfulness, answer_relevancy, context_recall — judge = Qwen 7B Coder
- **Composite этой серии:** `rag_score = 0.4·F + 0.4·AR + 0.2·CR`
- **Latency:** p50/p95 в секундах

---

# 5 — Наблюдаемая разница Base и LoRA

| Метрика | base-rerank-k5 | lora-rerank-k5 | Δ |
|---|---|---|---|
| **rag_score** | **0.530** | 0.293 | **−45%** |
| lexical/rag_f1 | 0.257 | 0.105 | −59% |
| ragas/answer_relevancy | 0.767 | 0.156 | **−80%** |
| answer_similarity | 0.886 | 0.562 | −37% |
| ragas/faithfulness | 0.248 | 0.276 | +11% |
| ragas/context_recall | 0.621 | 0.601 | −3% |
| rag_better vs pure (out of 100) | 69 | 14 | −80% |

В этой паре runs только faithfulness был выше у LoRA; остальные показанные метрики
были ниже. Доверительные интервалы и paired significance test не считались.

**Максимальный сохранённый score:** `base-vanilla-k5` с rag_score = 0.548.

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

- Одиночные base runs с **Reranker и Query Transform** дали меньшие scores:
  0.55 → 0.53 → 0.51; выигрыш не продемонстрирован, но статистического сравнения нет.
- **QT** в этих runs добавил +66% к латентности (8.3s → 13.8s p95) без
  наблюдаемого выигрыша по score.
- **LoRA-run** получил answer_relevancy почти в 5× ниже; причина требует
  независимого judge и human eval

---

# 7 — Phase 3: разброс трёх LoRA samples

`lora-rerank-k5` × seed ∈ {42, 43, 44} (одинаковый конфиг):

| Метрика | mean | std | CV% |
|---|---|---|---|
| rag_score | 0.282 | 0.016 | 5.6% |
| **faithfulness** | 0.230 | **0.041** | **18%** |
| **answer_relevancy** | 0.166 | **0.030** | **18%** |
| context_recall | 0.615 | 0.022 | 3.6% |

**Что это значит:**

- **Base vs LoRA:** observed gap 0.24 намного больше разброса трёх LoRA samples,
  но это не z-test: base запускался один раз, samples не paired.
- **top_k:** наблюдаемый размах rag_score между сохранёнными runs (`≈0.043`) больше
  std трёх seed runs (`≈0.016`); для `fetch_k` и `temperature` величины сопоставимы
  с seed spread. Значимость ни для одного knob не оценивалась.

**Lesson:** нужны repeated base/LoRA/judge runs и paired bootstrap либо larger
sample; правило «3 seeds или 500+» здесь не было проверено power analysis.

---

# 8A — Гипотезы: стиль и judge

### A. Возможный стилистический сдвиг

LoRA выучила SO-стиль: краткий ответ + минимум кода. Из smoke-test:

> LoRA: «I found this solution: `if torch.cuda.is_available(): # do something`»
> base: длинный ответ с пояснениями, fallback на CPU, комментарии, ссылки

Более короткий ответ может получать меньший lexical F1 даже при приемлемом смысле.
Это наблюдение, а не доказанный root cause.

### B. Judge bias (self-evaluation)

Судья = Qwen 7B Coder, то есть та же base family, что у адаптера. Это создаёт риск
self-evaluation bias, но направление и размер bias не измерялись. Нужны
независимый judge и human calibration.

---

# 8B — Гипотезы: данные и capacity

### C. Малый объём (1.9k) и 246 adversarial rows (13% total)

В одном smoke-case с irrelevant context LoRA не отказалась. Этого недостаточно,
чтобы доказать причину; нужны отдельный refusal set и controlled fraction sweep.

### D. LoRA capacity r=16 могла недохватить

40M trainable из 7.66B = 0.5%. Capacity могла влиять, но r=32/r=64 в этой серии
не сравнивались.

---

# 9A — Рекомендация на дату эксперимента

### Историческая рекомендация по наблюдаемым runs

- Наблюдаемый кандидат: `base-vanilla-k5` (legacy rag_score=0.548, p95=8.3s).
- Не использовать проверенный LoRA-адаптер этой серии без новой оценки.
- Query transform в одиночном run дал +66% latency без наблюдаемого выигрыша.
- Reranker дал score на 0.018 ниже vanilla и добавил около 0.1s; эквивалентность
  или причинный эффект по одному run установить нельзя.

Это не текущая production policy: перед deployment нужны независимый judge,
актуальная схема весов, несколько seeds, human eval и проверка prompt contract.

---

# 9B — Методология и следующий эксперимент

### Методологические уроки

1. **Eval-pipeline сам по себе ценный артефакт** — без него LoRA-регрессию мы не поймали бы в проде
2. **Seed-stability перед выводами** — single-shot eval мог соврать в обе стороны
3. **Self-judge bias — некалиброванный риск**; нужен независимый judge
4. **Negative result — это тоже результат** — методология работает, даже если гипотеза не подтвердилась

### Backlog на дату эксперимента

- Расширить датасет до 5–10k (снизить `--min-score` до 3)
- Adversarial-fraction 30%, разнообразить refusal-фразы
- LoRA r=32 или r=64
- Eval с GPT-4o-mini как judge
- Human eval на N=20 для калибровки RAGAS

---

# 10A — Исторический бюджет

**vast.ai (непроверенная оценка на дату запуска, не текущая цена):**
- LoRA training: ~$0.50 за сообщавшиеся ~95 минут; точный GPU не зафиксирован
- vLLM для eval: ~$1.50 за сообщавшиеся ~3 часа
- **Итого: ~$2** за всю серию экспериментов

---

# 10B — Артефакты и воспроизводимость

**W&B:** snapshot содержит 13 run directories: 11 matrix и 2 sanity. Текущая
папка `eval-runner/configs/` содержит 16 JSON: 14 run-specific presets и 2 общих
overlays; это уже не точная копия исторической матрицы.

**Git на момент отчёта:** использовалась ветка `main`; в истории серии отмечены:
- `feat(dataset-prep): ...`
- `feat(lora-train): ...`
- `feat(eval-runner, rag): layered configs, 12 presets, RAGAS venv reuse`
- `fix(eval-runner): hydrate run config via api.run()`

**Воспроизводимость:** presets сохраняют основные knobs, но старое обещание
«повторяется за 3 часа» больше не является гарантией. Повтор требует исходного
corpus/index, точных model endpoints, prompt, judge, seeds и совместимых
dependencies; полный run остаётся сетевым, compute-heavy и потенциально платным.

---

# Итог исторической серии

| Что измерили | Что узнали |
|---|---|
| **Гипотеза:** LoRA на RAG-aware SO Q&A улучшит faithfulness | Не получила подтверждения в этой серии |
| **Headline:** rag_score baseline=0.55, LoRA=0.29 | Наблюдаемая разница 1.9×; significance не оценена |
| **Главный вопрос:** какой компонент даёт выигрыш? | Среди одиночных base runs максимум у vanilla; причинный выигрыш не установлен |
| **Что было проверено:** | dataset-prep (1891 row), eval-runner (11 matrix runs), W&B tracking |
| **Что имело худшие наблюдаемые метрики:** | Tested LoRA config; query-transform в этой выборке |
| **Самый ценный артефакт:** | Eval-инфраструктура — позволяет быстро итерировать **и не deploy'ить регрессию вслепую** |
