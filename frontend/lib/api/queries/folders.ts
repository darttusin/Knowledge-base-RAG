// Folder queries (GET requests)

import { api, safeRequest } from "../client"
import type { ApiResult } from "@/types/api"

/**
 * Backend folder response
 */
export interface BackendFolder {
  id: number
  name: string
  path: string
  parent_id: number | null
  created_at: string
  document_count: number
}

/**
 * Get all folders for the current user
 */
export async function getFolders(): Promise<ApiResult<BackendFolder[]>> {
  return safeRequest(() => api.get<BackendFolder[]>("/api/folder"))
}
