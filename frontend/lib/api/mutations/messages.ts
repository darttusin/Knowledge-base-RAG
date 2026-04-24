// Message mutations (feedback, regenerate)

import { api, safeRequest } from "../client"
import type { ApiResult } from "@/types/api"

const ENDPOINTS = {
  feedback: "/api/message/feedback",
  regenerate: "/api/message/regenerate",
}

export interface MessageFeedbackRequest {
  message_id: number
  feedback: "like" | "dislike"
}

/**
 * Send feedback for a message
 */
export async function sendMessageFeedback(
  messageId: number,
  feedback: "like" | "dislike"
): Promise<ApiResult<{ success: boolean }>> {
  const result = await safeRequest(async () => {
    await api.post(ENDPOINTS.feedback, {
      message_id: messageId,
      feedback,
    })
    return { success: true }
  })

  return result
}

/**
 * Regenerate assistant response for a message
 */
export async function regenerateMessage(
  messageId: number
): Promise<ApiResult<{ success: boolean }>> {
  const result = await safeRequest(async () => {
    await api.post(ENDPOINTS.regenerate, {
      message_id: messageId,
    })
    return { success: true }
  })

  return result
}
