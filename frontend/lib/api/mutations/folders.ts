import { api, safeRequest, type ApiResult } from "@/lib/api/client"
import type { BackendFolder } from "../queries/folders"

export interface CreateFolderRequest {
  name: string
  path: string
  parent_id: number | null
}

export async function createFolder(
  request: CreateFolderRequest
): Promise<ApiResult<BackendFolder>> {
  return safeRequest(() => api.post<BackendFolder>("/api/folder", request))
}

export async function moveFolderToParent(
  folderId: number,
  parentId: number | null
): Promise<ApiResult<void>> {
  return safeRequest(async () => {
    await api.patch<void>(`/api/folder/${folderId}`, { parent_id: parentId })
  })
}

export async function deleteFolder(folderId: number): Promise<ApiResult<void>> {
  return safeRequest(() => api.delete<void>(`/api/folder/${folderId}`))
}
