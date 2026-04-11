import { create } from "zustand"
import { devtools } from "zustand/middleware"
import type { Conversation, Message, Source, PreGeneratedQuery } from "@/lib/types"
import {
  getConversations,
  getConversation,
  createConversation,
  updateConversation as apiUpdateConversation,
  deleteConversation as apiDeleteConversation,
  sendMessage as apiSendMessage,
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
          activeConversation: state.activeConversation
            ? {
                ...state.activeConversation,
                messages: [...(state.activeConversation.messages || []), userMessage],
                updatedAt: new Date(),
              }
            : null,
          isWaitingForResponse: true,
        }))

        // Check if this is first message (for title update)
        const isFirstMessage = (activeConversation.messages?.length || 0) === 0

        // Send to API
        const result = await apiSendMessage(parseInt(activeConversation.id), content)

        if (result.success) {
          // Convert backend source references to frontend Source type
          const sources = result.data.sources.map((sourceRef, idx) => ({
            id: `source-${result.data.messageId}-${idx}`,
            title: sourceRef.document_name,
            content: sourceRef.chunk_text,
            relevance: sourceRef.relevance_score,
            type: "document" as const,
            documentId: sourceRef.source_id.toString(),
            folderPath: sourceRef.folder_path || undefined,
          }))

          const assistantMessage: Message = {
            id: result.data.messageId,
            role: "assistant",
            content: result.data.assistantMessage,
            sources,
            timestamp: new Date(),
          }

          set((state) => {
            const updatedConversation = state.activeConversation
              ? {
                  ...state.activeConversation,
                  messages: [...(state.activeConversation.messages || []), assistantMessage],
                  updatedAt: new Date(),
                }
              : null

            // Add new sources to existing ones
            const newSources = assistantMessage.sources || []
            const allSources = [...state.selectedSources, ...newSources]

            return {
              activeConversation: updatedConversation,
              conversations: updatedConversation
                ? state.conversations.map((c) =>
                    c.id === updatedConversation.id ? updatedConversation : c
                  )
                : state.conversations,
              selectedSources: allSources,
              isWaitingForResponse: false,
            }
          })

          // If this was first message, reload conversation to get updated title
          if (isFirstMessage) {
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
        } else {
          set({ isWaitingForResponse: false })
          toast.error(`Failed to send message: ${result.error}`)
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

      // Edit message (regenerates response)
      editMessage: async (messageId, newContent) => {
        // Simply re-send the message to regenerate
        await get().sendMessage(newContent)
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
export const selectSelectedFolderIds = (state: ChatStore) => state.selectedFolderIds
export const selectIsLoading = (state: ChatStore) => state.isLoading
export const selectError = (state: ChatStore) => state.error
