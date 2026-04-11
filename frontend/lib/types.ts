export interface MessageVersion {
  content: string
  timestamp: Date
  responseId?: string
}

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  status?: "streaming" | "completed" | "error" | "cancelled"
  sources?: Source[]
  timestamp: Date
  isEdited?: boolean
  editHistory?: MessageVersion[]
  parentMessageId?: string
  feedback?: "like" | "dislike"
}

export interface Source {
  id: string
  title: string
  content: string
  relevance: number
  type: "document"
  fileType?: "md" | "txt"
  uploadedAt?: Date
  folderId?: string | null
  folderPath?: string
  documentId: string
}

export interface PreGeneratedQuery {
  query: string
  icon: "database" | "doc" | "browser"
}

export interface Conversation {
  id: string
  title: string
  messages?: Message[] // undefined means not loaded yet, [] means loaded but empty
  preGeneratedQueries?: PreGeneratedQuery[]
  createdAt: Date
  updatedAt: Date
}

export interface ExtendedSourceData {
  author: string
  createdAt: string
  updatedAt: string
  path: string
  tags: string[]
  fullContent: string
  relatedSources: { id: string; title: string; type: "document" | "database" | "api" }[]
  metadata: { label: string; value: string }[]
}
