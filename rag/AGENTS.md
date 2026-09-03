# RAG agent guide

Наследует корневой `AGENTS.md`. При интеграции прочитайте `backend/AGENTS.md`,
`prompt-contract/` и вызывающий dataset/eval code.

## Package map

- `rag/documents.py`: recursive load, extension filter, MD5 file dedup, Markdown-aware
  splitting, chunk dedup.
- `rag/vectorstore.py`: sentence-transformer embeddings, cosine Chroma, stable ids.
- `rag/retriever.py`: dense/basic, dense + CrossEncoder rerank, query transform.
- `rag/llm.py`: OpenAI-compatible client/retries.
- `rag/chains.py`/`rag/prompts.py`: answer generation и legacy prompt path.
- `rag/tracking.py`: experiment metadata.

Standalone settings отличаются от backend: в частности
`RAG_LLM_MODEL_GENERATION` против backend `RAG_LLM_MODEL`. Paths зависят от cwd.

## Index contract

Версию индекса образуют corpus и extension set, cleaning, chunk text/order,
`chunk_size`/`chunk_overlap`, embedding model/normalization, distance metric,
metadata schema и collection name. При несовместимом изменении создайте новую
collection; bundled `data/chromadb` не rebuild/delete без явного разрешения.

`load_documents()` сортирует paths, удаляет full-file duplicates по MD5 и chunks
по детерминированному ключу. IDs используют position + SHA1; Python `hash()`
запрещён из-за process salt. Index и query используют одну model/normalization.

`index_chunks()` сейчас полностью выходит, когда collection count ненулевой. Это
не нормальная incremental indexing semantics: upload в непустой index обычно не
добавляет документ. Regression должен покрывать empty/non-empty/idempotent/update.

Bundled collection не имеет `user_id`/`document_id` metadata. Shared retrieval не
фильтрует tenant, а delete по `document_id` не удаляет её chunks. Для тестов
используйте temp directory, отдельное имя и tiny fake embedder; не мутируйте
tracked SQLite.

## Retrieval и generation

Strategies:

- `basic`/`vanilla` — dense top-k;
- `rerank` — larger dense candidate set + CrossEncoder;
- `query_transform` — original, LLM rewrite, HyDE, три searches, dedup и rerank.

BM25/RRF не реализованы. Query transform добавляет LLM latency/cost
до answer generation; тестируйте частичные failures и детерминированный candidate
merge. CrossEncoder logits не гарантированы в [0,1], поэтому не называйте текущий
relevance калиброванной вероятностью.

Source metadata должно оставаться согласованным с backend path lookup/citations.
Любая смена source path, folder, document ids или dedup order требует сквозной
проверки upload→retrieve→DB mapping→`[§N]` citations→delete.

## Prompt contract

`prompt-contract` fingerprint покрывает system/user/chunk templates, joiner и
`context_chunks`. Synth хранит structured chunks, training рендерит выбранный
contract, а `rag.chains` умеет применить его явно; число chunks должно
соответствовать serving top-k. Token-level изменение — новая версия контракта.

Текущий backend вызывает `answer(..., contract=None)`, а serving/eval не загружают
и не сверяют adapter fingerprint. Наличие `prompt_contract.json` после training не
означает end-to-end compatibility. Сохраняйте legacy path до явной миграции.

## Проверки

У package нет полноценного unit suite. Добавляйте targeted tests с fake
embedder/LLM/Chroma temp path; не скачивайте модели для обычного unit test.
Проверяйте:

- stable load/chunk/id/dedup;
- empty/non-empty incremental indexing и metadata;
- candidate merge/rerank/top-k;
- LLM retry без blocking event loop;
- contract fingerprint и legacy behavior;
- backend tenant/source/citation integration.

Ruff/Black/Pyright запускайте на изменённых files, repo-wide baseline не green.
