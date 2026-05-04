// Document mutations (POST/PUT/DELETE requests)

import { api, safeRequest, uploadFile } from "../client"
import { adaptSourceToDocument, type BackendSource } from "../adapters"
import type { ApiResult, UploadDocumentResponse } from "@/types/api"

const ENDPOINTS = {
  upload: "/api/source",
  delete: (id: string) => `/api/source/${id}`,
  move: (id: string) => `/api/source/${id}`,
}

/**
 * Upload a document
 */
export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void
): Promise<ApiResult<UploadDocumentResponse>> {
  const result = await safeRequest(() => uploadFile<BackendSource>(ENDPOINTS.upload, file, onProgress))

  if (!result.success) {
    return result
  }

  const adapted = adaptSourceToDocument(result.data)

  return {
    success: true,
    data: {
      ...adapted,
      chunksCount: 0,
    },
  }
}

/**
 * Move a document to a folder (or to root when folderId is null)
 */
export async function moveDocumentToFolder(
  id: string,
  folderId: string | null
): Promise<ApiResult<{ success: boolean }>> {
  const folderIdInt = folderId ? Number.parseInt(folderId, 10) : null
  return safeRequest(async () => {
    await api.patch(ENDPOINTS.move(id), { folder_id: folderIdInt })
    return { success: true }
  })
}

/**
 * Delete a document
 */
export async function deleteDocument(id: string): Promise<ApiResult<{ success: boolean }>> {
  // Backend returns 204 No Content, so we just check if request was successful
  const result = await safeRequest(async () => {
    await api.delete(ENDPOINTS.delete(id))
    return { success: true }
  })

  return result
}
