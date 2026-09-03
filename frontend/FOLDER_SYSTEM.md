# Folder and Document System: Current Implementation

This document records the folder/document behavior that is implemented now. It is not a roadmap.
In particular, folder-scoped RAG, upload-to-current-folder, and persistent folder rename/delete are
not complete.

## Source Files

The main implementation is split across:

- **frontend/app/documents/page.tsx** — documents page orchestration, dialogs, selection, search UI,
  and drag-and-drop wiring;
- **frontend/hooks/useDocuments.ts** — list/search pagination, upload, preview, delete, local rename,
  download, and document move;
- **frontend/lib/store/folder-store.ts** — folder state and actions;
- **frontend/lib/store/chat-store.ts** — selectedFolderIds state and chat streaming;
- **frontend/components/documents/folder-tree.tsx** — hierarchical navigation and tree drop targets;
- **frontend/components/documents/folder-row.tsx** — folder rows in the main list and drop targets;
- **frontend/components/documents/document-row.tsx** — document rows and drag payloads;
- **frontend/components/documents/folder-breadcrumb.tsx** — folder navigation path;
- **frontend/components/chat/folder-selector.tsx** — currently unmounted folder-selection UI;
- **frontend/lib/api/queries/folders.ts** and
  **frontend/lib/api/mutations/folders.ts** — folder transport;
- **frontend/lib/api/queries/documents.ts** and
  **frontend/lib/api/mutations/documents.ts** — source/document transport;
- **frontend/lib/api/mutations/conversations.ts** — actual chat stream request.

When these files disagree with this document, the source code and backend contract are
authoritative.

## Data Model and Root Semantics

The backend folder response is declared in **frontend/lib/api/queries/folders.ts**:

```typescript
interface BackendFolder {
  id: number
  name: string
  path: string
  parent_id: number | null
  created_at: string
  document_count: number
}
```

**frontend/lib/store/folder-store.ts** adapts it to:

```typescript
interface Folder {
  id: string
  name: string
  path: string
  parentId: string | null
  createdAt: Date
  documentCount: number
}
```

IDs are numeric at the backend boundary and strings in frontend state. Preserve explicit
conversion when adding operations.

There is no canonical backend folder whose ID is root:

- parent_id null / parentId null identifies a top-level folder;
- currentFolderId null means “All Documents” and omits the folder_id list filter;
- dropping onto the tree's “All Documents” target moves an item to parent/folder null.

Some current code still treats the string root as a synthetic node. Those paths are compatibility
remnants, not the data contract. Do not create or persist a folder with ID root to make them work.

## Persistence Matrix

| Operation               | UI                           | Backend request                   | Current status                                      |
| ----------------------- | ---------------------------- | --------------------------------- | --------------------------------------------------- |
| Load folders            | Yes                          | GET /api/folder                   | Persisted/backend-backed                            |
| Create folder/subfolder | Yes                          | POST /api/folder                  | Persisted/backend-backed                            |
| Move folder             | Yes, including drag-and-drop | PATCH /api/folder/:id             | Persisted/backend-backed, then folders are reloaded |
| Rename folder           | Yes                          | None                              | Local-only; lost on reload                          |
| Delete folder           | Yes, with confirmation       | None from the store               | Local-only; lost on reload                          |
| List/search documents   | Yes                          | GET /api/source                   | Backend-backed                                      |
| Load preview content    | Yes, lazily                  | GET /api/source/:id               | Backend-backed                                      |
| Upload document         | Yes                          | POST /api/source                  | Backend-backed, but no folder assignment is sent    |
| Move document           | Yes, including drag-and-drop | PATCH /api/source/:id             | Persisted/backend-backed                            |
| Rename document         | Yes                          | None                              | Local-only; lost on reload                          |
| Delete document         | Yes                          | DELETE /api/source/:id            | Persisted/backend-backed                            |
| Download document       | Yes                          | None                              | Builds a local Blob from currently loaded content   |
| Select folders for chat | State/component only         | No folder field in stream request | Not active end-to-end                               |

An API deleteFolder wrapper exists in **frontend/lib/api/mutations/folders.ts**, but
**frontend/lib/store/folder-store.ts** does not call it. Do not infer persistence from the wrapper.

## Folder Loading and Navigation

The documents page loads folders through useFolderStore.loadFolders. The store converts IDs and
dates and retains backend-provided paths and document counts.

Navigation behavior:

- currentFolderId null shows “All Documents” and top-level folders;
- selecting a folder sets currentFolderId to its string ID;
- the document query then sends its numeric value as folder_id;
- breadcrumbs are derived from the current store hierarchy;
- expanded tree node IDs are client-only state.

Folder state uses Zustand devtools and no persistence middleware.

## Creating Folders

createFolder accepts a name and an optional parent ID.

The frontend currently computes a path and sends:

```typescript
{
  name: string
  path: string
  parent_id: number | null
}
```

Top-level creation uses parent_id null. Subfolder creation converts the parent string ID to a
number. The returned backend folder is appended to store state.

The frontend path computation contains compatibility checks for a synthetic root ID. Treat the
path returned by the backend as authoritative after later reloads.

## Moving Folders

moveFolder sends:

```typescript
PATCH /api/folder/:id
{
  parent_id: number | null
}
```

After a successful request, the store reloads folders so paths of the moved folder and descendants
come from the backend.

Drag-and-drop is implemented in both the tree and the main folder list. It:

- supports moving to another folder or to top level;
- rejects moving a folder into itself or one of its descendants;
- supports mixed multi-selection payloads from the documents page;
- uses Promise.allSettled so one failed move does not suppress all outcomes.

Folder drag payloads use **application/x-folder-id** with comma-separated IDs. A temporary
window-level marker is used during dragover because browser dataTransfer contents may not be
readable until drop.

## Renaming and Deleting Folders

Rename and delete controls are rendered, but both operations only mutate Zustand state:

- updateFolder changes the local name/path;
- deleteFolder removes the selected folder and descendants from the local folder array.

Neither action calls the backend. Both are reversed by a reload. The delete confirmation currently
claims that nested documents will be deleted, but the frontend store does not issue such a request;
do not rely on that copy as a persistence contract.

Deleting the current folder can set currentFolderId to the legacy root string even though null is
the normal “All Documents” value. This is a known root-semantics inconsistency.

## Document Listing, Search, and Pagination

**frontend/hooks/useDocuments.ts** calls GET /api/source with:

```text
page=<number>
limit=<number>
folder_id=<numeric folder id, omitted for All Documents>
query=<trimmed query, when present>
search_in=name|content|both
```

Current behavior:

- page size is 100;
- search is debounced by 500 ms;
- the three UI modes are name, content, and both;
- pagination uses infinite scrolling;
- list results omit document content;
- changing folder or search resets to page 1;
- folder filtering is performed by the backend, not a local array filter.

The documents page accepts an id query parameter, but checks it only against the currently loaded
document page for the active folder/search (limit 100). If the ID is found, it moves to the
document's folder when needed and opens the lazy preview. If the loaded page is non-empty and the
ID is absent, it clears the query parameter; it does not call getDocument to resolve an arbitrary
source ID.

The active UI Document type only permits md and txt. Backend transport adapters also admit pdf and
docx source types but cast them into the narrower frontend type. Treat PDF/DOCX UI behavior as
unsupported until the type and rendering flow are deliberately expanded.

## Upload

The current client-side validator accepts:

- .md and .txt extensions;
- files of at most 5 MB;
- the MIME types listed in **frontend/lib/constants.ts**, with empty MIME allowed for Markdown.

Upload uses XHR through **frontend/lib/api/client.ts** to report progress and sends the file to
POST /api/source.

useDocuments.uploadFiles accepts a folderId argument, defaulting to currentFolderId, but does not
pass that value to the API mutation. The upload request therefore contains no folder assignment.
After success, the returned document is prepended to the current in-memory list even if its
backend-returned folderId does not match the current view.

Do not describe or test this as “upload to current folder” until folder_id is added to the actual
multipart request and backend contract.

## Preview, Rename, Delete, and Download

Preview content is lazy:

1. list results are mapped with content set to an empty string;
2. openPreview calls GET /api/source/:id only if content is empty;
3. the returned content is stored on the in-memory document object and shown by the Markdown
   preview.

Document rename sanitizes the filename and updates only local hook state. There is no rename API
call.

Document delete calls DELETE /api/source/:id and removes the item from local state after success.

Document download does not call a backend download route. It creates a text/plain Blob from
doc.content. If the document has not been previewed, doc.content is normally empty and the
downloaded file can be empty.

## Moving Documents and Multi-Selection

Moving a document sends:

```typescript
PATCH /api/source/:id
{
  folder_id: number | null
}
```

The hook optimistically removes the document from the current list and decrements the count. It
restores the previous state if the request fails, and reloads folders after success to refresh
document counts.

On the All Documents view, moving to a folder still removes the item from the in-memory list even
though an all-documents query would normally include it. A reload corrects the view.

The documents page supports:

- Command/Ctrl-click multi-selection;
- Shift-click range selection;
- mixed selections of folders and documents;
- bulk drag-and-drop;
- Escape to clear selection.

Document payloads use **application/x-document-id** with comma-separated IDs. Drop handlers process
document and folder moves independently with Promise.allSettled.

## Chat Folder Selection Is Not End-to-End

The chat store contains:

- selectedFolderIds;
- setSelectedFolders;
- toggleFolderSelection;
- clearFolderSelection.

An empty selectedFolderIds array is intended by the UI to mean “all documents”.

**frontend/components/chat/folder-selector.tsx** implements a popover and recursive selection, but:

1. no production component imports and renders FolderSelector;
2. it looks for a folder whose ID equals root, while backend-loaded top-level folders have
   parentId null, so its current tree would be empty;
3. **frontend/lib/api/mutations/conversations.ts** sends only dialogue_id, message, and optional
   parent_message_id;
4. selectedFolderIds is never read when constructing the stream request.

Therefore folder-scoped RAG is not implemented. The main chat continues to search according to
backend default behavior.

The actual stream request is:

```typescript
POST /api/message/stream
{
  dialogue_id: number
  message: string
  parent_message_id?: number
}
```

Adding folder-scoped RAG requires a coordinated backend/frontend contract change, not merely
mounting FolderSelector.

## API Surface Used by This Feature

| Method and path          | Request                                          | Current consumer                      |
| ------------------------ | ------------------------------------------------ | ------------------------------------- |
| GET /api/folder          | None                                             | useFolderStore.loadFolders            |
| POST /api/folder         | name, path, parent_id                            | useFolderStore.createFolder           |
| PATCH /api/folder/:id    | parent_id                                        | useFolderStore.moveFolder             |
| DELETE /api/folder/:id   | None                                             | Wrapper exists, store does not use it |
| GET /api/source          | page, limit, optional folder_id/query/search_in  | useDocuments                          |
| POST /api/source         | Multipart file only                              | useDocuments.uploadFiles              |
| GET /api/source/:id      | None                                             | Lazy preview/content                  |
| PATCH /api/source/:id    | folder_id                                        | useDocuments.moveDocument             |
| DELETE /api/source/:id   | None                                             | useDocuments.deleteDocument           |
| POST /api/message/stream | dialogue_id, message, optional parent_message_id | useChatStore                          |

Every path is passed to **frontend/lib/api/client.ts**, so NEXT_PUBLIC_API_URL must be the backend
origin without an /api suffix.

## Invariants for Changes

When changing this subsystem:

- preserve numeric backend IDs and string frontend IDs at adapter/store boundaries;
- use null, not a persisted root ID, for top-level folder relationships;
- distinguish “All Documents” from a root-only filter;
- do not claim persistence unless an action awaits a successful API request;
- reload or correctly update folder paths and counts after mutations;
- reject folder cycles for both single and bulk moves;
- keep partial failures visible during bulk operations;
- keep upload validation and backend capabilities intentionally aligned;
- do not add folder IDs only to a UI type—the actual stream request and backend schema must change;
- update API modules, adapters, stores/hooks, UI, and backend schemas together for contract changes.

## Verification

Run static checks from **frontend/**:

```bash
pnpm lint
pnpm format:check
pnpm exec tsc --noEmit --incremental false
pnpm build
```

There is no automated frontend test suite. Manually verify:

1. /documents with currentFolderId null and with a numeric folder selected.
2. Top-level and nested folder creation, followed by a reload.
3. Single and bulk document moves to a folder and to All Documents.
4. Single and bulk folder moves, including cycle rejection.
5. Folder and document rename/delete behavior before and after reload.
6. Search in name, content, and both modes.
7. Infinite scroll with more than 100 results.
8. Valid and invalid upload types/sizes and the returned folder assignment.
9. Preview before download and download without preview.
10. /documents?id=ID_FROM_THE_CURRENTLY_LOADED_FIRST_PAGE deep linking and query cleanup for a
    missing ID.

If implementing folder-scoped chat, additionally verify that the rendered selector can display
parentId-null folders, the stream payload contains the intended IDs, empty selection has an agreed
meaning, descendants follow the backend contract, and edit/regenerate preserve the same scope.
