// Conversation queries (GET requests)

import { api, safeRequest } from "../client"
import { adaptDialogueToConversation, type BackendDialogue } from "../adapters"
import type { ApiResult, ConversationListItem } from "@/types/api"

const ENDPOINTS = {
  list: "/api/dialogue",
  single: (id: string) => `/api/dialogue/${id}`,
}

/**
 * Get list of conversations
 */
export async function getConversations(query?: string): Promise<ApiResult<ConversationListItem[]>> {
  const result = await safeRequest(() =>
    api.get<BackendDialogue[]>(ENDPOINTS.list, query ? { query } : undefined)
  )

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: result.data.map(adaptDialogueToConversation),
  }
}

/**
 * Get single conversation
 */
export async function getConversation(
  id: string
): Promise<ApiResult<ConversationListItem>> {
  const result = await safeRequest(() => api.get<BackendDialogue>(ENDPOINTS.single(id)))

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: adaptDialogueToConversation(result.data),
  }
}
