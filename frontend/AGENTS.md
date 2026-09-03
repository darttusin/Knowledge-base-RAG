# Frontend agent guide

Наследует корневой `AGENTS.md`. Canonical frontend — фактический код в `app/`,
`components/`, `hooks/` и `lib/` плюс реальные backend routes. `CLAUDE.md` и
`FOLDER_SYSTEM.md` синхронизированы на 2026-09-03, но код остаётся источником
истины.

## Stack, structure, style

Next.js 16 App Router, React 19, strict TypeScript, Tailwind v4, Radix/shadcn,
Zustand. Node target — 20; package manager — pnpm. Не запускайте npm и не меняйте
старый `package-lock.json`.

- `app/`: routes/layout; `components/`: feature UI; `components/ui/`: primitives.
- `hooks/`: reusable state; `lib/api/`: transport/adapters; `lib/store/`: Zustand.
- `@/*` указывает на root frontend.
- Рабочий CSS entrypoint — `app/globals.css`; `styles/globals.css` устарел.
- Routes: `/`, `/login`, `/documents`, `/settings`,
  `/dev/ui-regression/chat-sidebar-trigger`. `/terms`/`/privacy` отсутствуют.

Следуйте `.prettierrc`: no semicolons, double quotes, 2 spaces, width 100, ES5
trailing commas, Tailwind sorting. Feature files обычно kebab-case, components
PascalCase, hooks `useSomething.ts`. Не делайте массовых переименований и не
правьте shadcn primitives без проверки всех consumers.

Большая часть feature code client-side. Добавляйте `"use client"` только при
browser APIs/hooks. API client использует `window`/`localStorage` и не готов для
Server Components.

Не трогайте `clo`, `.next` или tracked `tsconfig.tsbuildinfo`. Последний сохраняйте
через `pnpm exec tsc --noEmit --incremental false`.

## API и auth

`lib/api/client.ts` централизует `NEXT_PUBLIC_API_URL`, Bearer JWT, timeout, 401
redirect, XHR upload и SSE parser. Base URL — backend origin без `/api`; текущий
fallback `"/api"` ошибочно даёт `/api/api/...`.

Queries/mutations описывают backend transport; adapters преобразуют snake_case в
camelCase. Реальные domain types главным образом в `lib/types.ts`;
`types/api.ts`/`types/messages.ts` частично дублируют intent. Не ограничивайтесь
правкой одного type file и не вызывайте новый endpoint прямым component fetch.
FastAPI `detail` пока плохо отображается клиентом, который ищет `error/message`.

Auth хранит `auth_token`/`auth_user` в localStorage; cookies/middleware/server auth
нет. Любой 401 чистит auth и hard-redirects. Known issues: delete-account удаляет
не тот key, settings ждёт другой error code, Remember Me ничего не меняет, logout
UI отсутствует. Не меняйте storage semantics без сквозных auth tests.

## Chat и streaming

`lib/store/chat-store.ts` хранит conversations, active conversation, messages,
panels, `selectedFolderIds` и async state. Сами folder entities находятся в
`lib/store/folder-store.ts`. Message invariants:

- backend ids numbers, UI ids strings;
- user view id `<message_id>-user`, assistant id — numeric string;
- `parent_message_id` связывает edit/regenerate variants;
- `undefined` messages = not loaded, empty array = loaded empty;
- active object и элемент list должны обновляться согласованно.

Send flow оптимистически добавляет user и temp assistant, читает SSE chunks,
заменяет id/sources по complete и перечитывает первый dialogue для title.
`_activeStreamController` один на module: concurrent send/route switch может
прервать чужой stream; блокируйте late events старого dialogue.

Основной backend path — `POST /api/message/stream`. Wrapper regenerate существует
без backend route и не используется. Проверяйте first message/title, chunks,
citations/source modal, edit/regenerate/variants, switch during stream и 401.
Markdown использует GFM + `skipHtml`; raw HTML не включать.

## Folders и documents

Backend top-level folder = `parent_id:null`; frontend `currentFolderId:null` =
«все документы». Остаточный synthetic `"root"` несовместим. Create/move folder
синхронизированы с API, rename/delete в store local-only.

Folder selector не подключён; `selectedFolderIds` не отправляется в message API,
backend schema/retrieval filter отсутствуют. Не выдавайте UI state за готовый RAG
scope.

`useDocuments`: pagination, debounced name/content/both search, lazy preview,
upload/delete/move и DnD. Known gaps:

- upload принимает `folderId`, но request его теряет; новый Source остаётся root;
- download Blob берёт lazy `doc.content` и может быть пустым вместо backend route;
- frontend md/txt ≤5 MiB против backend text decode ≤10 MiB;
- rename local-only; bulk move использует `Promise.allSettled`.

После mutation reload должен подтверждать server state. Проверяйте pagination,
search modes, preview, upload/download/delete, single/bulk DnD, move/cycle и
breadcrumbs.

## Code rendering

Python идёт через authenticated backend; JavaScript — sandboxed iframe; ONNX/WebGPU
имеют отдельные dynamic paths. Не включайте raw HTML, не ослабляйте iframe/
validators/auth. Stop сейчас не всегда отменяет underlying execution; TypeScript
не транспилируется; CDN hash constants не обеспечивают реальную integrity check.

## Проверки

Из корня репозитория:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm lint
pnpm format:check
pnpm exec tsc --noEmit --incremental false
pnpm build
```

Test/typecheck scripts в package.json нет. Build может требовать сеть для
`next/font/google`. Визуально проверьте changed routes, light/dark, desktop/mobile,
loading/empty/error/populated, keyboard/focus/labels/contrast и browser console.
`/documents` вызывает `useSearchParams` вне защищающего Suspense pattern — build
особенно важен для этой страницы.
