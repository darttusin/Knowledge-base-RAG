// Conversation mutations (POST/PUT/DELETE requests)

import { api, safeRequest, streamRequest, type ApiRequestError } from "../client"
import {
  adaptDialogueToConversation,
  type BackendDialogue,
  type BackendMessage,
  type BackendSourceReference,
} from "../adapters"
import type { ApiResult, ConversationListItem } from "@/types/api"

const ENDPOINTS = {
  create: "/api/dialogue",
  update: (id: string) => `/api/dialogue/${id}`,
  delete: (id: string) => `/api/dialogue/${id}`,
  sendMessage: "/api/message",
  sendMessageStream: "/api/message/stream",
}

export interface SendMessageRequest {
  dialogue_id: number
  message: string
  parent_message_id?: number
}

export interface SendMessageResponse {
  userMessage: string
  assistantMessage: string
  messageId: string
  sources: BackendSourceReference[]
}

export interface SendMessageStreamChunkEvent {
  type: "chunk"
  delta: string
}

export interface SendMessageStreamCompleteEvent {
  type: "complete"
  sources: BackendMessage["sources"]
  message_id: number
  parent_message_id?: number | null
  created_at: string
}

export type SendMessageStreamEvent = SendMessageStreamChunkEvent | SendMessageStreamCompleteEvent

export interface SendMessageStreamCallbacks {
  onChunk: (delta: string) => void
  onComplete: (payload: SendMessageStreamCompleteEvent) => void
  onError?: (error: ApiRequestError) => void
}

/**
 * Create a new conversation
 */
export async function createConversation(
  name: string = "New conversation"
): Promise<ApiResult<ConversationListItem>> {
  const result = await safeRequest(() => api.post<BackendDialogue>(ENDPOINTS.create, { name }))

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

export async function sendMessageStream(
  dialogueId: number,
  message: string,
  callbacks: SendMessageStreamCallbacks,
  signal?: AbortSignal,
  parentMessageId?: number
): Promise<void> {
  await streamRequest<SendMessageStreamEvent>(
    ENDPOINTS.sendMessageStream,
    {
      dialogue_id: dialogueId,
      message,
      parent_message_id: parentMessageId,
    },
    {
      onChunk: (event) => {
        if (event.type === "chunk") {
          callbacks.onChunk(event.delta)
          return
        }
        if (event.type === "complete") {
          callbacks.onComplete(event)
        }
      },
      onError: callbacks.onError,
    },
    signal
  )
}
