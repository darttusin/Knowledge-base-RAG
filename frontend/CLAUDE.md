# Frontend Assistant Guide

This document describes the frontend as it exists in the repository. It is a working guide for
coding agents, not a roadmap. Verify behavior in the referenced source files before changing a
contract.

## Scope and Sources of Truth

The frontend lives in **frontend/**. The canonical backend lives in the repository-level
**backend/** directory; **frontend/backend/models/.gitkeep** is only a placeholder.

When documentation and code disagree, use this order of precedence:

1. Current source in **frontend/app/**, **frontend/components/**, **frontend/hooks/**, and
   **frontend/lib/**.
2. Current backend routes and schemas in **backend/**.
3. **frontend/package.json** and the frontend configuration files.
4. This guide and **frontend/FOLDER_SYSTEM.md**.

Keep changes scoped to the task. Do not opportunistically reformat or refactor unrelated files.
Do not edit generated output such as **frontend/.next/**. The tracked files
**frontend/tsconfig.tsbuildinfo**, **frontend/package-lock.json**, and the opaque
**frontend/clo** binary should not be changed incidentally.

## Runtime and Stack

- Node.js 20 is the reference runtime; **frontend/Dockerfile** uses **node:20-alpine**.
- pnpm is the project package manager.
- Next.js 16.1.3 with the App Router.
- React 19.2.3 and TypeScript in strict mode.
- Tailwind CSS 4, Radix UI, and shadcn/ui.
- Zustand 5 with devtools for application state.
- react-markdown with remark-gfm for assistant responses.
- next-themes and Sonner for theming and notifications.
- onnxruntime-web and @huggingface/transformers are used by the browser code-execution path.

React Hook Form and Zod are installed, but they are not an application-wide convention in the
current implementation. Login and settings forms use React state, and API responses are not
runtime-validated with Zod.

## Package Manager and Commands

Run frontend commands from **frontend/** and use pnpm only:

```bash
cd frontend

pnpm install --frozen-lockfile
pnpm dev
pnpm build
pnpm start
pnpm lint
pnpm lint:fix
pnpm format
pnpm format:check
```

The scripts are defined in **frontend/package.json**. There is no test or typecheck script.
Run the compiler explicitly:

```bash
pnpm exec tsc --noEmit --incremental false
```

Use **--incremental false** to avoid modifying the tracked
**frontend/tsconfig.tsbuildinfo** file. The repository also contains
**frontend/package-lock.json**, but pnpm and **frontend/pnpm-lock.yaml** are authoritative for
normal frontend work. Do not run npm install or update the npm lockfile unless the task explicitly
standardizes package management.

The Dockerfile installs pnpm globally without pinning a version, and **frontend/package.json**
does not declare a packageManager field. Do not silently change either policy in an unrelated
change.

## Configuration

### TypeScript

**frontend/tsconfig.json** enables strict mode, noEmit, isolated modules, bundler module
resolution, and the **@/** alias rooted at **frontend/**.

Prefer aliases for shared modules:

```typescript
import { Button } from "@/components/ui/button"
import { useChatStore } from "@/lib/store/chat-store"
import type { Message } from "@/lib/types"
```

Relative imports are acceptable for tightly coupled siblings within one feature. Preserve the
pattern already used by the surrounding files.

### ESLint

**frontend/eslint.config.mjs** extends Next core-web-vitals, Next TypeScript, and Prettier.
Important local rules are:

- rules-of-hooks and no-var are errors;
- unused variables, explicit any, exhaustive hook dependencies, prefer-const, and disallowed
  console calls are warnings;
- console.warn, console.error, console.debug, and console.info are allowed.

Do not treat warnings as permission to add new warnings.

### Prettier

**frontend/.prettierrc** defines:

- no semicolons;
- double quotes;
- two-space indentation;
- 100-character print width;
- ES5 trailing commas;
- LF line endings;
- prettier-plugin-tailwindcss class sorting.

Use **pnpm format:check** for verification. **pnpm format** and **pnpm lint:fix** mutate files.

### Styling

Tailwind v4 is configured through **frontend/postcss.config.mjs**; there is no
tailwind.config file. Theme tokens and the imported global styles live in
**frontend/app/globals.css**. **frontend/styles/globals.css** is an unused legacy duplicate.

Use Tailwind utilities and **cn()** from **frontend/lib/utils.ts** for ordinary styling. Small
dynamic inline styles already exist where a value is computed at runtime, such as folder nesting
indentation; do not rewrite them solely to satisfy an absolute “Tailwind only” rule.

Shared primitives in **frontend/components/ui/** follow shadcn/Radix patterns. Preserve their
public props and accessibility behavior. **frontend/components.json** is the shadcn configuration.

## App Router and Routes

The implemented pages are:

| URL                                     | Source                                                           | Notes                                                                                     |
| --------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| /                                       | **frontend/app/page.tsx**                                        | Main chat; selected dialogue is synchronized with the chat query parameter                |
| /login                                  | **frontend/app/login/page.tsx**                                  | Email/password login                                                                      |
| /documents                              | **frontend/app/documents/page.tsx**                              | Folder and document management; checks an id query parameter only against the loaded page |
| /settings                               | **frontend/app/settings/page.tsx**                               | Profile, password, and account deletion                                                   |
| /dev/ui-regression/chat-sidebar-trigger | **frontend/app/dev/ui-regression/chat-sidebar-trigger/page.tsx** | Manual visual regression fixture                                                          |

The root layout is **frontend/app/layout.tsx**. It imports
**frontend/app/globals.css**, configures Inter, Geist Mono, and Caveat through next/font, wraps the
application in the theme and application providers, and enables Vercel Analytics.

There are currently:

- no frontend App Router API routes;
- no route.ts files;
- no middleware;
- no server actions;
- no /terms or /privacy pages, even though the login page links to them.

Most feature components are client components. Keep browser-dependent API and auth code out of
Server Components: the current API client accesses window for relative URLs and reads auth from
localStorage.

The documents id deep-link is not an arbitrary source fetch. The page checks the ID only against
the currently loaded document page for the active folder/search (limit 100). If that page is
non-empty and the ID is absent, it clears the query parameter without calling getDocument. Use an
ID present in the loaded page when testing the current behavior.

The home page correctly puts the useSearchParams consumer inside a child rendered by Suspense.
The documents page calls useSearchParams in the page component itself and only renders a Suspense
boundary below that call. Treat this as a potential prerender/build issue and verify it with a
production build before changing the boundary.

## Directory Map

```text
frontend/
├── app/                         Next.js routes, root layout, and global CSS
├── components/
│   ├── chat/                    Chat header, input, messages, citations, folder selector
│   ├── documents/               Folder tree, rows, upload, preview, statistics
│   ├── ui/                      shadcn/Radix primitives
│   └── *.tsx                    Page-level feature components and providers
├── hooks/                       Feature hooks and UI helpers
├── lib/
│   ├── api/
│   │   ├── client.ts            Fetch/XHR/SSE client and ApiResult handling
│   │   ├── adapters.ts          Backend snake_case to UI camelCase mapping
│   │   ├── queries/             GET functions
│   │   └── mutations/           POST, PUT, PATCH, and DELETE functions
│   ├── auth/                    localStorage token and user helpers
│   ├── constants/               UI, feature, and security constants
│   ├── security/                Browser code-execution validation
│   ├── store/                   Zustand chat and folder stores
│   ├── code-executor.ts         Python API and sandboxed JavaScript execution
│   ├── webgpu-executor.ts       ONNX/Transformers browser execution
│   └── types.ts                 Active chat/domain types
└── types/                       Shared API/component types; some message types are legacy
```

Backup mock files ending in **.backup** are not active runtime modules.

## File and Export Conventions

Follow the existing tree rather than older aspirational naming rules:

- component filenames are normally kebab-case, for example
  **components/chat/message-list.tsx**;
- React component symbols use PascalCase;
- hooks use the useSomething.ts form, for example **hooks/useDocuments.ts**;
- utilities and stores use kebab-case;
- Next page and layout modules use default exports;
- reusable feature components normally use named exports.

Function components and hooks are the default. **frontend/components/error-boundary.tsx** is a
deliberate class-component exception because React error boundaries require that lifecycle.

Use type-only imports where appropriate. Avoid adding any when a transport or domain type can be
expressed precisely. Do not move types based only on folder names: the active chat model is in
**frontend/lib/types.ts**, while **frontend/types/messages.ts** is largely duplicate legacy
surface.

## API Client Contract

The API entry point is **frontend/lib/api/client.ts**.

**NEXT_PUBLIC_API_URL** must be a backend origin without the /api suffix:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8001
```

Every endpoint passed by the feature modules already starts with /api. The current fallback base
is /api, which would produce /api/api/... when the environment variable is absent. Do not assume
the fallback is a working same-origin proxy.

The client:

- adds Content-Type: application/json and a Bearer token when available;
- defaults non-streaming requests to a 600000 ms timeout;
- wraps ordinary errors in ApiRequestError;
- exposes safeRequest returning ApiResult;
- clears auth and hard-redirects to /login on HTTP 401;
- uses XHR for upload progress;
- parses line-oriented data fields for SSE streams.

FastAPI error bodies commonly use a detail field, but handleResponse currently reads only error,
message, and code. Preserve this known limitation or fix it explicitly with tests; do not document
generic HTTP errors as backend behavior.

### Active endpoint surface

| Area        | Method and path                                                | Frontend module                                                              |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Auth        | POST /api/user/auth                                            | **lib/api/mutations/auth.ts**                                                |
| User        | GET /api/user/me; PUT and DELETE /api/user                     | **lib/api/mutations/auth.ts**, **lib/api/mutations/user.ts**                 |
| Dialogues   | GET and POST /api/dialogue; GET, PUT, DELETE /api/dialogue/:id | **lib/api/queries/conversations.ts**, **lib/api/mutations/conversations.ts** |
| Suggestions | GET /api/dialogue/queries/pre-generated                        | **lib/api/queries/conversations.ts**                                         |
| Chat        | POST /api/message/stream                                       | **lib/api/mutations/conversations.ts**                                       |
| Feedback    | POST /api/message/feedback                                     | **lib/api/mutations/messages.ts**                                            |
| Sources     | GET and POST /api/source; GET, PATCH, DELETE /api/source/:id   | **lib/api/queries/documents.ts**, **lib/api/mutations/documents.ts**         |
| Folders     | GET and POST /api/folder; PATCH /api/folder/:id                | **lib/api/queries/folders.ts**, **lib/api/mutations/folders.ts**             |
| Code        | POST /api/code/execute                                         | **lib/code-executor.ts**                                                     |

A non-streaming POST /api/message wrapper exists but the chat store does not use it.
**lib/api/mutations/messages.ts** also contains an unused /api/message/regenerate wrapper; the
actual regeneration flow reuses /api/message/stream with parent_message_id. A folder DELETE
wrapper exists, but the current folder store does not call it.

Backend transport types are handwritten in the API modules and
**frontend/lib/api/adapters.ts**, then mapped to UI types. There is no generated client and no
runtime schema validation. A contract change usually requires coordinated edits to:

1. the backend schema and route;
2. a query or mutation module;
3. the relevant Backend-prefixed transport type;
4. **frontend/lib/api/adapters.ts**;
5. **frontend/lib/types.ts** or the actually imported shared type;
6. the consuming store, hook, and UI.

Search actual imports before editing **frontend/types/api.ts**; several declarations there are
aspirational or only partially used.

## Authentication

Auth helpers are in **frontend/lib/auth/token.ts**. The exact localStorage keys are:

- auth_token
- auth_user

Login saves the JWT and user object, and API requests send Authorization: Bearer. There are no auth
cookies, server sessions, middleware guards, or server-side token access.

Route behavior is inconsistent by design of the current code:

- the home page checks isAuthenticated and redirects before loading chat data;
- documents relies on an API 401 to trigger the global redirect;
- settings loads /api/user/me and also relies on the API client for HTTP 401 handling.

Known auth limitations:

- the Remember me checkbox changes UI state only; storage is always localStorage;
- no logout control is rendered;
- successful account deletion removes the wrong authToken key in the settings page instead of
  auth_token and auth_user;
- settings checks for UNAUTHORIZED although ordinary client codes use HTTP_401;
- /terms and /privacy links lead to missing routes.

Changing auth to cookies or server enforcement is an architecture change, not a local component
patch.

## Chat State and Streaming

The chat store is **frontend/lib/store/chat-store.ts**. It uses Zustand with devtools and no
persistence middleware.

Important state:

- conversations and activeConversation;
- sidebar and sources-panel visibility;
- selectedSources;
- loading, error, and waiting flags;
- selectedFolderIds;
- totalDocuments and pre-generated queries.

The chat query parameter is the browser-visible selected-dialogue state. Conversation list entries
may have messages undefined until selected; an empty array means the conversation was loaded and
has no messages.

The main send flow:

1. optimistically appends a UI user message;
2. appends a temporary assistant message;
3. calls POST /api/message/stream;
4. applies chunk events containing a delta string;
5. replaces temporary metadata on the complete event;
6. reloads a first-message dialogue so a backend-generated title becomes visible.

The request body is:

```typescript
{
  dialogue_id: number
  message: string
  parent_message_id?: number
}
```

Completion events provide message_id, sources, created_at, and an optional parent_message_id.
Backend IDs are numbers at the transport boundary and generally strings in UI state. User-side UI
message IDs are derived from the backend message ID with a -user suffix. Preserve these conversions
and parent links when editing, regenerating, or grouping variants.

Regeneration and edit flows use the same streaming endpoint and parent_message_id. The store owns
one module-level active AbortController, so only one stream is intended to be active globally.
Switching dialogues during generation is an edge case that must be tested.

selectedFolderIds and its actions exist, but selected IDs are not present in the stream request
body. Folder-scoped RAG is therefore not implemented. See **frontend/FOLDER_SYSTEM.md**.

## Folder and Document State

Folder state is in **frontend/lib/store/folder-store.ts**. Document orchestration is in
**frontend/hooks/useDocuments.ts**. The implementation status and exact persistence boundaries are
documented in **frontend/FOLDER_SYSTEM.md**.

Do not assume UI success means persistence:

- folder load, create, and move call the backend;
- folder rename and delete are local-only;
- document load, delete, and move call the backend;
- document rename is local-only;
- upload does not submit the selected folder;
- download uses the in-memory content and can be empty before preview.

Top-level backend folders have parent_id null. currentFolderId null means “All Documents”; there is
no canonical backend folder with the string ID root. Some UI paths still contain synthetic-root
assumptions and must be handled carefully.

## Message Rendering and Sources

**frontend/components/chat/assistant-message-bubble.tsx** renders Markdown with remark-gfm,
skipHtml, and restricted link protocols. It transforms source markers such as [§N] into interactive
citations. Do not enable raw HTML or relax URL handling casually.

Sources for the active conversation are aggregated from assistant messages by the chat store.
**frontend/components/sources-panel.tsx** filters that aggregated list locally.
**frontend/components/source-detail-modal.tsx** does not fetch a canonical full source; outside its
mock extended data it displays the retrieved chunk as full content.

## Code Execution and Security

Treat these files as security-sensitive:

- **frontend/components/code-block.tsx**
- **frontend/lib/code-executor.ts**
- **frontend/lib/webgpu-executor.ts**
- **frontend/lib/security/**
- **frontend/lib/constants/security.ts**

Python is sent to authenticated POST /api/code/execute. Ordinary JavaScript runs in a sandboxed
iframe with allow-scripts. The ONNX/Transformers path can execute dynamically constructed code in
the page context after pattern validation.

Known limitations:

- TypeScript is offered as executable but is not transpiled;
- the UI AbortController is not propagated through every executor, so Stop does not reliably
  cancel underlying work;
- integrity values exist in constants, but downloaded CDN code is not hash-verified;
- validation differs between the iframe and WebGPU paths.

Do not weaken the iframe sandbox, Markdown sanitization, code validators, or API authentication as
part of unrelated work. Security changes require manual adversarial checks as well as normal
quality gates.

## Environment Variables

The current **frontend/.env.local** defines:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_RAG_API_URL=http://localhost:8000
```

Only NEXT_PUBLIC_API_URL is used for requests. NEXT_PUBLIC_RAG_API_URL is currently unused.
**frontend/lib/logger.ts** optionally reads NEXT_PUBLIC_LOG_LEVEL.

Every NEXT_PUBLIC variable is shipped to the browser. Never place secrets in these variables or in
the tracked **frontend/.env.local** file. Restart the dev server after changing environment values.

**docker-compose.yml** publishes the frontend on port 3000, supplies the two URL variables, enables
WATCHPACK_POLLING, and starts pnpm dev. The backend CORS configuration must permit the frontend
origin.

## Implemented UI Behavior and Known Gaps

Important gaps to preserve in planning and tests:

- the chat attachment button has no handler;
- folder selection is not wired into the chat header or stream request;
- the source modal normally shows a retrieved chunk, not the full stored document;
- documents are limited client-side to .md and .txt files of at most 5 MB;
- document and folder rename are not persisted;
- folder delete is not persisted;
- upload ignores its folder argument;
- direct download may produce an empty file if preview content was never loaded;
- auth protection is client-side and inconsistent across pages;
- no automated test suite is configured.

Do not present these items as completed capabilities in documentation or UI changes.

## Verification

For a normal frontend change, run:

```bash
cd frontend
pnpm lint
pnpm format:check
pnpm exec tsc --noEmit --incremental false
pnpm build
```

If dependencies are not installed:

```bash
pnpm install --frozen-lockfile
```

The build may require network access because the root layout uses next/font/google. If a command
cannot run, report the exact command and environmental error; do not imply that it passed.

There is no automated test runner. Perform a browser smoke test proportional to the change:

- /login;
- /;
- /?chat=EXISTING_DIALOGUE_ID;
- /documents;
- /documents?id=ID_FROM_THE_CURRENTLY_LOADED_FIRST_PAGE;
- /settings;
- /dev/ui-regression/chat-sidebar-trigger for sidebar-trigger visuals.

For chat work, cover streaming chunks, completion, citations, edit, regeneration, variants, first
message title refresh, stream abort, and switching dialogues. For document work, cover pagination,
all three search modes, lazy preview, upload validation, delete, single and multi-item drag and
drop, folder moves, and backend reload behavior.

After verification, inspect Git status and ensure generated output, lockfiles, and
**frontend/tsconfig.tsbuildinfo** did not change unintentionally.
