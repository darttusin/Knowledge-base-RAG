import { create } from "zustand"
import { devtools } from "zustand/middleware"
import type { Conversation, Message, Source, PreGeneratedQuery } from "@/lib/types"
import {
  getConversations,
  getConversation,
  createConversation,
  updateConversation as apiUpdateConversation,
  deleteConversation as apiDeleteConversation,
  sendMessageStream as apiSendMessageStream,
  getDocuments,
  getPreGeneratedQueries,
} from "@/lib/api"
import { toast } from "sonner"

interface ChatState {
  // State
  conversations: Conversation[]
  activeConversation: Conversation | null
  showSources: boolean
  selectedSources: Source[]
  sidebarOpen: boolean
  isWaitingForResponse: boolean
  waitingConversationId: string | null
  selectedFolderIds: string[]
  totalDocuments: number
  isLoading: boolean
  error: string | null
  preGeneratedQueries: PreGeneratedQuery[]
}

interface ChatActions {
  // API actions
  loadConversations: () => Promise<void>
  loadDocumentCount: () => Promise<void>
  loadPreGeneratedQueries: () => Promise<void>
  createConversation: () => Promise<void>
  selectConversation: (conversation: Conversation) => Promise<void>
  updateConversation: (id: string, title: string) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  sendMessage: (content: string) => Promise<void>
  editMessage: (messageId: string, newContent: string) => Promise<void>

  // UI actions
  toggleSources: () => void
  closeSources: () => void
  toggleSidebar: () => void
  setSidebar: (open: boolean) => void
  setSelectedFolders: (folderIds: string[]) => void
  toggleFolderSelection: (folderId: string) => void
  clearFolderSelection: () => void
}

type ChatStore = ChatState & ChatActions
let activeStreamController: AbortController | null = null

interface StreamSourceReference {
  source_id: number
  document_name: string
  chunk_text: string
  relevance_score: number
  folder_path?: string | null
}

const mapSources = (sources: StreamSourceReference[], messageId: string): Source[] =>
  sources.map((sourceRef, idx) => ({
    id: `source-${messageId}-${idx}`,
    title: sourceRef.document_name,
    content: sourceRef.chunk_text,
    relevance: sourceRef.relevance_score,
    type: "document" as const,
    documentId: sourceRef.source_id.toString(),
    folderPath: sourceRef.folder_path || undefined,
  }))

export const useChatStore = create<ChatStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      conversations: [],
      activeConversation: null,
      showSources: true,
      selectedSources: [],
      sidebarOpen: true,
      isWaitingForResponse: false,
      waitingConversationId: null,
      selectedFolderIds: [],
      totalDocuments: 0,
      isLoading: false,
      error: null,
      preGeneratedQueries: [],

      // Load conversations from API
      loadConversations: async () => {
        set({ isLoading: true, error: null })

        const result = await getConversations()

        if (result.success) {
          const conversations: Conversation[] = result.data.map((conv) => ({
            id: conv.id,
            title: conv.title,
            messages: undefined, // Mark as not loaded yet
            createdAt: new Date(conv.createdAt),
            updatedAt: new Date(conv.updatedAt),
          }))

          set({
            conversations,
            // Don't auto-select first conversation - let URL handling or user interaction do it
            isLoading: false,
          })
        } else {
          set({ error: result.error, isLoading: false })
          toast.error(`Failed to load conversations: ${result.error}`)
        }
      },

      // Load document count
      loadDocumentCount: async () => {
        const result = await getDocuments({ page: 1, limit: 1 })

        if (result.success) {
          set({ totalDocuments: result.data.total })
        }
      },

      // Load pre-generated queries
      loadPreGeneratedQueries: async () => {
        const result = await getPreGeneratedQueries()

        if (result.success) {
          set({ preGeneratedQueries: result.data })
        }
      },

      // Create new conversation
      createConversation: async () => {
        const result = await createConversation()

        if (result.success) {
          const newConvo: Conversation = {
            id: result.data.id,
            title: result.data.title,
            messages: [],
            preGeneratedQueries: result.data.preGeneratedQueries,
            createdAt: new Date(result.data.createdAt),
            updatedAt: new Date(result.data.updatedAt),
          }

          set((state) => ({
            conversations: [newConvo, ...state.conversations],
            activeConversation: newConvo,
            selectedSources: [],
            preGeneratedQueries: result.data.preGeneratedQueries || [],
          }))

          toast.success("New conversation created")
        } else {
          toast.error(`Failed to create conversation: ${result.error}`)
        }
      },

      selectConversation: async (conversation) => {
        // Immediately set as active (for UI responsiveness)
        set({ activeConversation: conversation })

        // Load full conversation with messages if not already loaded
        // undefined means not loaded, [] means loaded but empty
        if (conversation.messages === undefined) {
          set({ isLoading: true })

          const result = await getConversation(conversation.id)

          if (result.success) {
            const fullConversation: Conversation = {
              id: result.data.id,
              title: result.data.title,
              messages: result.data.messages?.map((msg) => ({
                id: msg.id,
                role: msg.role,
                content: msg.content,
                sources: msg.sources,
                timestamp: new Date(msg.timestamp),
                feedback: msg.role === "assistant" ? (msg as any).feedback : undefined,
              })) || [],
              preGeneratedQueries: result.data.preGeneratedQueries,
              createdAt: new Date(result.data.createdAt),
              updatedAt: new Date(result.data.updatedAt),
            }

            // Collect all sources from all messages
            const allSources: Source[] = []
            fullConversation.messages?.forEach((msg) => {
              if (msg.sources) {
                allSources.push(...msg.sources)
              }
            })

            set((state) => ({
              activeConversation: fullConversation,
              conversations: state.conversations.map((c) =>
                c.id === fullConversation.id ? fullConversation : c
              ),
              selectedSources: allSources,
              preGeneratedQueries: result.data.preGeneratedQueries || state.preGeneratedQueries,
              isLoading: false,
            }))
          } else {
            // On error, still set as active but with empty messages
            set({
              activeConversation: { ...conversation, messages: [] },
              selectedSources: [],
              isLoading: false
            })
            toast.error(`Failed to load conversation: ${result.error}`)
          }
        } else {
          // If already loaded, still update selectedSources
          const allSources: Source[] = []
          conversation.messages?.forEach((msg) => {
            if (msg.sources) {
              allSources.push(...msg.sources)
            }
          })
          set({ selectedSources: allSources })
        }
      },

      // Update conversation (rename)
      updateConversation: async (id, title) => {
        const result = await apiUpdateConversation(id, title)

        if (result.success) {
          set((state) => ({
            conversations: state.conversations.map((c) =>
              c.id === id ? { ...c, title } : c
            ),
            activeConversation:
              state.activeConversation?.id === id
                ? { ...state.activeConversation, title }
                : state.activeConversation,
          }))

          toast.success("Conversation renamed")
        } else {
          toast.error(`Failed to rename conversation: ${result.error}`)
        }
      },

      // Delete conversation
      deleteConversation: async (id) => {
        const result = await apiDeleteConversation(id)

        if (result.success) {
          const { conversations, activeConversation } = get()
          const newConversations = conversations.filter((c) => c.id !== id)

          set({
            conversations: newConversations,
            activeConversation:
              activeConversation?.id === id ? newConversations[0] || null : activeConversation,
          })

          toast.success("Conversation deleted")
        } else {
          toast.error(`Failed to delete conversation: ${result.error}`)
        }
      },

      // Send message
      sendMessage: async (content) => {
        let { activeConversation } = get()

        // Auto-create conversation if none exists
        if (!activeConversation) {
          const createResult = await createConversation()

          if (!createResult.success) {
            toast.error(`Failed to create conversation: ${createResult.error}`)
            return
          }

          const newConvo: Conversation = {
            id: createResult.data.id,
            title: createResult.data.title,
            messages: [],
            preGeneratedQueries: createResult.data.preGeneratedQueries,
            createdAt: new Date(createResult.data.createdAt),
            updatedAt: new Date(createResult.data.updatedAt),
          }

          set((state) => ({
            conversations: [newConvo, ...state.conversations],
            activeConversation: newConvo,
            preGeneratedQueries: createResult.data.preGeneratedQueries || [],
          }))

          activeConversation = newConvo
        }

        const userMessage: Message = {
          id: Date.now().toString(),
          role: "user",
          content,
          timestamp: new Date(),
        }

        // Add user message immediately
        set((state) => ({
          activeConversation:
            state.activeConversation?.id === activeConversation.id
              ? {
                  ...state.activeConversation,
                  messages: [...(state.activeConversation.messages || []), userMessage],
                  updatedAt: new Date(),
                }
              : state.activeConversation,
          conversations: state.conversations.map((c) =>
            c.id === activeConversation.id
              ? {
                  ...c,
                  messages: [...(c.messages || []), userMessage],
                  updatedAt: new Date(),
                }
              : c
          ),
          isWaitingForResponse: true,
          waitingConversationId: activeConversation.id,
        }))

        // Check if this is first message (for title update)
        const isFirstMessage = (activeConversation.messages?.length || 0) === 0

        const tempAssistantId = `assistant-stream-${Date.now()}`
        const tempAssistant: Message = {
          id: tempAssistantId,
          role: "assistant",
          content: "",
          status: "streaming" as const,
          timestamp: new Date(),
        }

        set((state) => ({
          activeConversation:
            state.activeConversation?.id === activeConversation.id
              ? {
                  ...state.activeConversation,
                  messages: [...(state.activeConversation.messages || []), tempAssistant],
                }
              : state.activeConversation,
          conversations: state.conversations.map((c) =>
            c.id === activeConversation.id
              ? { ...c, messages: [...(c.messages || []), tempAssistant] }
              : c
          ),
        }))

        activeStreamController?.abort()
        activeStreamController = new AbortController()

        let completed = false
        await apiSendMessageStream(
          parseInt(activeConversation.id),
          content,
          {
            onChunk: (delta) => {
              set((state) => {
                const patchMessages = (messages: Message[] = []) =>
                  messages.map((msg) =>
                    msg.id === tempAssistantId ? { ...msg, content: msg.content + delta } : msg
                  )
                return {
                  activeConversation:
                    state.activeConversation?.id === activeConversation.id
                      ? { ...state.activeConversation, messages: patchMessages(state.activeConversation.messages) }
                      : state.activeConversation,
                  conversations: state.conversations.map((c) =>
                    c.id === activeConversation.id ? { ...c, messages: patchMessages(c.messages) } : c
                  ),
                }
              })
            },
            onComplete: (event) => {
              completed = true
              const messageId = event.message_id.toString()
              const sources = mapSources(event.sources, messageId)
              set((state) => {
                const patchMessages = (messages: Message[] = []) =>
                  messages.map((msg) =>
                    msg.id === tempAssistantId
                      ? { ...msg, id: messageId, sources, status: "completed" as const, timestamp: new Date(event.created_at) }
                      : msg
                  )
                const updatedActiveConversation =
                  state.activeConversation?.id === activeConversation.id
                    ? { ...state.activeConversation, messages: patchMessages(state.activeConversation.messages), updatedAt: new Date() }
                    : state.activeConversation
                const updatedConversations = state.conversations.map((c) =>
                  c.id === activeConversation.id ? { ...c, messages: patchMessages(c.messages), updatedAt: new Date() } : c
                )
                const allSources = (updatedActiveConversation?.messages || []).flatMap((msg) => msg.sources || [])
                return {
                  activeConversation: updatedActiveConversation,
                  conversations: updatedConversations,
                  selectedSources: allSources,
                  isWaitingForResponse: false,
                  waitingConversationId: null,
                }
              })
            },
            onError: (error) => {
              const isCancelled = error.code === "ABORT"
              set((state) => {
                const patchMessages = (messages: Message[] = []) =>
                  messages.map((msg) =>
                    msg.id === tempAssistantId
                      ? { ...msg, status: (isCancelled ? "cancelled" : "error") as "cancelled" | "error" }
                      : msg
                  )
                return {
                  activeConversation:
                    state.activeConversation?.id === activeConversation.id
                      ? { ...state.activeConversation, messages: patchMessages(state.activeConversation.messages) }
                      : state.activeConversation,
                  conversations: state.conversations.map((c) =>
                    c.id === activeConversation.id ? { ...c, messages: patchMessages(c.messages) } : c
                  ),
                  isWaitingForResponse: false,
                  waitingConversationId: null,
                }
              })
              if (!isCancelled) {
                toast.error(`Failed to send message: ${error.message}`)
              }
            },
          },
          activeStreamController.signal
        )

        if (!completed) {
          set({ isWaitingForResponse: false, waitingConversationId: null })
        }

        // If this was first message, reload conversation to get updated title
        if (completed && isFirstMessage) {
          const updatedConvo = await getConversation(activeConversation.id)
          if (updatedConvo.success) {
            set((state) => ({
              activeConversation: state.activeConversation
                ? { ...state.activeConversation, title: updatedConvo.data.title }
                : null,
              conversations: state.conversations.map((c) =>
                c.id === activeConversation.id ? { ...c, title: updatedConvo.data.title } : c
              ),
            }))
          }
        }
      },

      toggleSources: () => {
        set((state) => ({ showSources: !state.showSources }))
      },

      closeSources: () => {
        set({ showSources: false })
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarOpen: !state.sidebarOpen }))
      },

      setSidebar: (open) => {
        set({ sidebarOpen: open })
      },

      setSelectedFolders: (folderIds) => {
        set({ selectedFolderIds: folderIds })
      },

      toggleFolderSelection: (folderId) => {
        set((state) => {
          const isSelected = state.selectedFolderIds.includes(folderId)
          return {
            selectedFolderIds: isSelected
              ? state.selectedFolderIds.filter((id) => id !== folderId)
              : [...state.selectedFolderIds, folderId],
          }
        })
      },

      clearFolderSelection: () => {
        set({ selectedFolderIds: [] })
      },

      // Edit message (or regenerate response for that specific message group)
      editMessage: async (messageId, newContent) => {
        const { activeConversation } = get()
        if (!activeConversation?.messages) return

        const originalMessages = activeConversation.messages
        const userMessageIndex = originalMessages.findIndex(
          (msg) => msg.id === messageId && msg.role === "user"
        )

        if (userMessageIndex === -1) return

        const nextUserMessageIndex = originalMessages.findIndex(
          (msg, idx) => idx > userMessageIndex && msg.role === "user"
        )

        const updatedUserMessage: Message = {
          ...originalMessages[userMessageIndex],
          content: newContent,
          isEdited: originalMessages[userMessageIndex].content !== newContent,
        }

        const messagesBefore = originalMessages.slice(0, userMessageIndex)
        const messagesAfter =
          nextUserMessageIndex === -1 ? [] : originalMessages.slice(nextUserMessageIndex)
        const messagesWithoutCurrentResponses = [
          ...messagesBefore,
          updatedUserMessage,
          ...messagesAfter,
        ]

        // Optimistically replace current user block and remove old assistant responses for it
        set((state) => {
          const updatedConversation = state.activeConversation
            ? {
                ...state.activeConversation,
                messages: messagesWithoutCurrentResponses,
                updatedAt: new Date(),
              }
            : null

          return {
            activeConversation: updatedConversation,
            conversations: updatedConversation
              ? state.conversations.map((c) =>
                  c.id === updatedConversation.id ? updatedConversation : c
                )
              : state.conversations,
            isWaitingForResponse: true,
            waitingConversationId: activeConversation.id,
          }
        })

        const tempAssistantId = `assistant-stream-${Date.now()}`
        const tempAssistant: Message = {
          id: tempAssistantId,
          role: "assistant",
          content: "",
          status: "streaming" as const,
          parentMessageId: updatedUserMessage.id,
          timestamp: new Date(),
        }

        set((state) => {
          const conversationMessages = [...messagesBefore, updatedUserMessage, tempAssistant, ...messagesAfter]
          const updatedConversation = state.activeConversation
            ? { ...state.activeConversation, messages: conversationMessages, updatedAt: new Date() }
            : null
          return {
            activeConversation: updatedConversation,
            conversations: updatedConversation
              ? state.conversations.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
              : state.conversations,
          }
        })

        activeStreamController?.abort()
        activeStreamController = new AbortController()

        let completed = false
        await apiSendMessageStream(
          parseInt(activeConversation.id),
          newContent,
          {
            onChunk: (delta) => {
              set((state) => {
                const patchMessages = (messages: Message[] = []) =>
                  messages.map((msg) =>
                    msg.id === tempAssistantId ? { ...msg, content: msg.content + delta } : msg
                  )
                return {
                  activeConversation: state.activeConversation
                    ? { ...state.activeConversation, messages: patchMessages(state.activeConversation.messages) }
                    : null,
                  conversations: state.conversations.map((c) =>
                    c.id === activeConversation.id ? { ...c, messages: patchMessages(c.messages) } : c
                  ),
                }
              })
            },
            onComplete: (event) => {
              completed = true
              const messageId = event.message_id.toString()
              const sources = mapSources(event.sources, messageId)
              set((state) => {
                const patchMessages = (messages: Message[] = []) =>
                  messages.map((msg) =>
                    msg.id === tempAssistantId
                      ? { ...msg, id: messageId, status: "completed" as const, sources, timestamp: new Date(event.created_at) }
                      : msg
                  )
                const updatedConversation = state.activeConversation
                  ? { ...state.activeConversation, messages: patchMessages(state.activeConversation.messages), updatedAt: new Date() }
                  : null
                const allSources = (updatedConversation?.messages || []).flatMap((msg) => msg.sources || [])
                return {
                  activeConversation: updatedConversation,
                  conversations: updatedConversation
                    ? state.conversations.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
                    : state.conversations,
                  selectedSources: allSources,
                  isWaitingForResponse: false,
                  waitingConversationId: null,
                }
              })
            },
            onError: (error) => {
              const isCancelled = error.code === "ABORT"
              if (!isCancelled) {
                set((state) => {
                  const revertedConversation = state.activeConversation
                    ? { ...state.activeConversation, messages: originalMessages }
                    : null
                  return {
                    activeConversation: revertedConversation,
                    conversations: revertedConversation
                      ? state.conversations.map((c) => (c.id === revertedConversation.id ? revertedConversation : c))
                      : state.conversations,
                    isWaitingForResponse: false,
                    waitingConversationId: null,
                  }
                })
                toast.error(`Failed to regenerate message: ${error.message}`)
                return
              }
              set({ isWaitingForResponse: false, waitingConversationId: null })
            },
          },
          activeStreamController.signal
        )

        if (!completed) {
          set({ isWaitingForResponse: false, waitingConversationId: null })
        }
      },
    }),
    { name: "chat-store" }
  )
)

// Селекторы
export const selectConversations = (state: ChatStore) => state.conversations
export const selectActiveConversation = (state: ChatStore) => state.activeConversation
export const selectShowSources = (state: ChatStore) => state.showSources
export const selectSelectedSources = (state: ChatStore) => state.selectedSources
export const selectSidebarOpen = (state: ChatStore) => state.sidebarOpen
export const selectIsWaitingForResponse = (state: ChatStore) => state.isWaitingForResponse
export const selectIsWaitingForActiveConversation = (state: ChatStore) =>
  Boolean(state.activeConversation?.id) &&
  state.waitingConversationId === state.activeConversation?.id
export const selectSelectedFolderIds = (state: ChatStore) => state.selectedFolderIds
export const selectIsLoading = (state: ChatStore) => state.isLoading
export const selectError = (state: ChatStore) => state.error
