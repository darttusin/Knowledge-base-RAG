// Document queries (GET requests)

import { api, safeRequest } from "../client"
import {
  adaptSourcesList,
  adaptSourceToDocument,
  type BackendSource,
  type BackendSourcesList,
} from "../adapters"
import type {
  ApiResult,
  DocumentListItem,
  PaginatedResponse,
  PaginationParams,
} from "@/types/api"

const ENDPOINTS = {
  list: "/api/source",
  single: (id: string) => `/api/source/${id}`,
  content: (id: string) => `/api/source/${id}`,
}

export type SearchIn = "name" | "content" | "both"

/**
 * Get paginated list of documents
 */
export async function getDocuments(
  params?: PaginationParams & {
    folderId?: string | null
    query?: string
    searchIn?: SearchIn
  }
): Promise<ApiResult<PaginatedResponse<DocumentListItem>>> {
  const queryParams: Record<string, string | number | undefined> = {
    page: params?.page || 1,
    limit: params?.limit || 10,
  }

  // Add folder_id filter if specified (not null and not undefined)
  if (params?.folderId) {
    const folderId = parseInt(params.folderId)
    if (!isNaN(folderId)) {
      queryParams.folder_id = folderId
    }
  }

  // Add search parameters
  if (params?.query && params.query.trim()) {
    queryParams.query = params.query.trim()
    queryParams.search_in = params.searchIn || "both"
  }

  const result = await safeRequest(() =>
    api.get<BackendSourcesList>(ENDPOINTS.list, queryParams)
  )

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: adaptSourcesList(result.data),
  }
}

/**
 * Backend source with content
 */
export interface BackendSourceContent extends BackendSource {
  content: string
}

/**
 * Get single document details
 */
export async function getDocument(id: string): Promise<ApiResult<DocumentListItem>> {
  const result = await safeRequest(() => api.get<BackendSource>(ENDPOINTS.single(id)))

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: adaptSourceToDocument(result.data),
  }
}

/**
 * Get document content
 */
export async function getDocumentContent(id: string): Promise<ApiResult<string>> {
  const result = await safeRequest(() => api.get<BackendSourceContent>(ENDPOINTS.content(id)))

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: result.data.content,
  }
}
