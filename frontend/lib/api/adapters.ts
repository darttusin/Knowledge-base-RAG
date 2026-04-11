// Adapters to convert backend responses to frontend types

import type { DocumentListItem } from "@/types/api"

/**
 * Backend source response
 */
export interface BackendSource {
  source_id: number
  name: string
  source_type: "md" | "txt" | "pdf" | "docx"
  size_bytes: number
  created_at: string
  folder_id?: number | null
  folder_path?: string | null
}

/**
 * Backend sources list response
 */
export interface BackendSourcesList {
  total_count: number
  total_size_bytes: number
  page: number
  limit: number
  total_pages: number
  sources: BackendSource[]
}

/**
 * Backend pre-generated query
 */
export interface BackendPreGeneratedQuery {
  query: string
  icon: "database" | "doc" | "browser"
}

/**
 * Backend dialogue response
 */
export interface BackendDialogue {
  dialogue_id: number
  name: string
  created_at: string
  updated_at: string
  pre_generated_queries?: BackendPreGeneratedQuery[]
  messages?: BackendDialogueMessage[]
}

/**
 * Backend source reference
 */
export interface BackendSourceReference {
  source_id: number
  document_name: string
  chunk_text: string
  relevance_score: number
  folder_path?: string | null
}

/**
 * Backend message in dialogue response
 */
export interface BackendDialogueMessage {
  message_id: number
  parent_message_id?: number | null
  user_message: string
  assistant_response: string | null
  sources: BackendSourceReference[] | null
  feedback: string | null
  created_at: string
}

/**
 * Backend message response (from send message API)
 */
export interface BackendMessage {
  message_id: number
  user_message: string
  assistant_response: string
  sources: BackendSourceReference[]
  code_executions: Array<{
    code: string
    success: boolean
    stdout: string
    stderr: string
    result: string | null
    error: string | null
  }>
  created_at: string
}

/**
 * Convert backend source to frontend document
 */
export function adaptSourceToDocument(source: BackendSource): DocumentListItem {
  return {
    id: source.source_id.toString(),
    name: source.name,
    size: source.size_bytes,
    fileType: source.source_type as "md" | "txt",
    uploadedAt: source.created_at,
    chunksCount: 0,
    folderId: source.folder_id?.toString() || null,
    folderPath: source.folder_path || undefined,
  }
}

/**
 * Convert backend sources list to paginated response
 */
export function adaptSourcesList(backendList: BackendSourcesList) {
  return {
    items: backendList.sources.map(adaptSourceToDocument),
    total: backendList.total_count,
    totalSize: backendList.total_size_bytes,
    page: backendList.page,
    limit: backendList.limit,
    hasMore: backendList.page < backendList.total_pages,
  }
}

/**
 * Convert backend dialogue message to frontend message
 */
export function adaptDialogueMessage(message: BackendDialogueMessage) {
  const parentUserMessageId = message.parent_message_id
    ? `${message.parent_message_id}-user`
    : undefined

  if (!message.assistant_response) {
    // User message only (shouldn't happen normally)
    return {
      id: parentUserMessageId || `${message.message_id}-user`,
      role: "user" as const,
      content: message.user_message,
      timestamp: new Date(message.created_at),
    }
  }

  // Convert sources from BackendSourceReference to Source objects
  const sources =
    message.sources?.map((sourceRef, idx) => ({
      id: `source-${message.message_id}-${idx}`,
      title: sourceRef.document_name,
      content: sourceRef.chunk_text,
      relevance: sourceRef.relevance_score,
      type: "document" as const,
      documentId: sourceRef.source_id.toString(),
      folderPath: sourceRef.folder_path || undefined,
    })) || undefined

  // Regenerated versions are linked to existing user message via parent_message_id.
  // Do not duplicate user bubbles for them.
  if (parentUserMessageId) {
    return {
      id: message.message_id.toString(),
      role: "assistant" as const,
      content: message.assistant_response,
      sources,
      parentMessageId: parentUserMessageId,
      timestamp: new Date(message.created_at),
      feedback: message.feedback as "like" | "dislike" | undefined,
    }
  }

  // Regular first response: return both user and assistant messages
  return [
    {
      id: `${message.message_id}-user`,
      role: "user" as const,
      content: message.user_message,
      timestamp: new Date(message.created_at),
    },
    {
      id: message.message_id.toString(),
      role: "assistant" as const,
      content: message.assistant_response,
      sources,
      timestamp: new Date(message.created_at),
      feedback: message.feedback as "like" | "dislike" | undefined,
    },
  ]
}

/**
 * Convert backend dialogue to frontend conversation
 */
export function adaptDialogueToConversation(dialogue: BackendDialogue) {
  const messages = dialogue.messages
    ? dialogue.messages.flatMap((msg) => adaptDialogueMessage(msg))
    : []

  return {
    id: dialogue.dialogue_id.toString(),
    title: dialogue.name,
    messageCount: dialogue.messages?.length || 0,
    createdAt: dialogue.created_at,
    updatedAt: dialogue.updated_at,
    preGeneratedQueries: dialogue.pre_generated_queries,
    messages,
  }
}
