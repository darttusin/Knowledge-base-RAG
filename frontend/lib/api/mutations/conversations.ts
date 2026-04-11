// Conversation mutations (POST/PUT/DELETE requests)

import { api, safeRequest } from "../client"
import { adaptDialogueToConversation, type BackendDialogue, type BackendMessage } from "../adapters"
import type { ApiResult, ConversationListItem } from "@/types/api"

const ENDPOINTS = {
  create: "/api/dialogue",
  update: (id: string) => `/api/dialogue/${id}`,
  delete: (id: string) => `/api/dialogue/${id}`,
  sendMessage: "/api/message",
}

export interface SendMessageRequest {
  dialogue_id: number
  message: string
}

export interface SendMessageResponse {
  userMessage: string
  assistantMessage: string
  messageId: string
  sources: string[]
}

/**
 * Create a new conversation
 */
export async function createConversation(
  name: string = "New conversation"
): Promise<ApiResult<ConversationListItem>> {
  const result = await safeRequest(() =>
    api.post<BackendDialogue>(ENDPOINTS.create, { name })
  )

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: adaptDialogueToConversation(result.data),
  }
}

/**
 * Update conversation (rename)
 */
export async function updateConversation(
  id: string,
  name: string
): Promise<ApiResult<{ success: boolean }>> {
  const result = await safeRequest(async () => {
    await api.put(ENDPOINTS.update(id), { name })
    return { success: true }
  })

  return result
}

/**
 * Delete a conversation
 */
export async function deleteConversation(id: string): Promise<ApiResult<{ success: boolean }>> {
  const result = await safeRequest(async () => {
    await api.delete(ENDPOINTS.delete(id))
    return { success: true }
  })

  return result
}

/**
 * Send a message
 */
export async function sendMessage(
  dialogueId: number,
  message: string
): Promise<ApiResult<SendMessageResponse>> {
  const result = await safeRequest(() =>
    api.post<BackendMessage>(ENDPOINTS.sendMessage, {
      dialogue_id: dialogueId,
      message,
    })
  )

  if (!result.success) {
    return result
  }

  return {
    success: true,
    data: {
      userMessage: result.data.user_message,
      assistantMessage: result.data.assistant_response,
      messageId: result.data.message_id.toString(),
      sources: result.data.sources,
    },
  }
}
