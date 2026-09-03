# torch-parser — legacy crawler документации PyTorch

Этот каталог содержит экспериментальный HTML → Markdown crawler для Python API
документации PyTorch. Он не участвует в текущем backend/RAG runtime и не является
canonical способом обновления corpus.

## Текущее состояние

Package пока нельзя считать корректно упакованным:

- `pyproject.toml` объявляет wheel package `src/torch_parser`, которого нет;
- фактические файлы лежат непосредственно в `src/`;
- console script не объявлен;
- test suite отсутствует;
- старый `uv run -m torch-parser.src` полагается на namespace resolution исходного
  дерева, а не на объявленный wheel package.

Source entrypoint — `src/__main__.py`. И старый `uv run -m torch-parser.src` из
корня, и прямой `python -m src` из `torch-parser/` могут разрешиться в текущем
checkout, но обходят сломанный package contract и сразу запускают сетевой crawl.
Это хрупкие source-tree invocations, не поддержанный smoke path. Сначала нужно
исправить package layout/entrypoint и добавить tests.

## Что делает код

1. Читает base URL и output directory из settings.
2. Загружает `<TORCH_URL>/pytorch-api.html`.
3. Ищет Sphinx/PyTorch markup `article.bd-article` и
   `div.toctree-wrapper.compound`.
4. Переходит по найденным ссылкам, затем по вложенным относительным `*.html`.
5. Извлекает содержимое article, преобразует HTML в Markdown и пишет отдельные
   `.md` files в `PATH_TO_SAVE`.

Parser жёстко зависит от текущей HTML-структуры сайта. Изменение Sphinx theme,
classes или URL layout может привести к пустому результату без явного failure.

## Конфигурация

`src/settings.py` использует `pydantic-settings` и способен загрузить `.env`
относительно текущего рабочего каталога. Но root `.gitignore` игнорирует только
корневой `/.env`, а `torch-parser/.env` — нет. Не создавайте component-local файл
с secrets; передавайте два обязательных значения через process environment:

```bash
export TORCH_URL=https://docs.example.invalid/stable/
export PATH_TO_SAVE=/absolute/path/to/new-output-directory
```

Это только placeholders. `TORCH_URL` должен быть доверенным base URL; trailing
slash важен для `urljoin`. `PATH_TO_SAVE` и необходимые parent directories должны
существовать: реализация использует `os.mkdir`, а не рекурсивное создание дерева.

Не добавляйте credentials в URL или `.env`: при HTTP-ошибке crawler логирует URL
и response body.

## Ограничения и риски

- Crawl требует сети и не имеет dry-run, allowlist домена или явного rate limit.
- HTTP client не задаёт проектные retries/timeout policy.
- Files открываются с режимом `w`, поэтому совпавшие имена перезаписываются.
- Resume/manifest/checksum/dedup отсутствуют.
- Имена output paths строятся непосредственно из `href`; они не проходят
  полноценную path-safety валидацию.
- Top-level `href` с `../` может вывести запись за `PATH_TO_SAVE`, а nested filter
  блокирует только literal `https://`: ссылки `http://` и `//other-host` могут
  вызвать cross-origin fetch. Запускайте только на полностью доверенном snapshot.
- Conversion удаляет все обратные слеши и post-processes anchors regex-ом, что
  может повредить code examples и Markdown.
- Parser сохраняет только найденные API articles; это не доказательство полноты
  или соответствия конкретной версии PyTorch.

Не запускайте crawler поверх `data/dataset/**` или другого tracked corpus. Для
эксперимента используйте новый temporary/ignored directory, сначала ограничьте
scope на уровне кода и вручную проверьте небольшой output. Полный crawl и замена
corpus требуют отдельного осознанного решения с provenance версии документации.

## Что нужно для возвращения в поддерживаемое состояние

Минимальный набор работ:

1. привести layout к importable package `torch_parser` и объявить console script;
2. валидировать scheme/host и безопасно нормализовать output paths;
3. добавить timeout, retries, rate limiting и ограниченный smoke mode;
4. заменить destructive overwrite на manifest/checkpointed writes;
5. добавить fixture-based tests для navigation, conversion и malicious href;
6. сохранять source URL, PyTorch version, crawl timestamp и content checksum.

До этого `torch-parser` следует считать неподдерживаемым legacy-инструментом, а
не частью воспроизводимого data pipeline.
