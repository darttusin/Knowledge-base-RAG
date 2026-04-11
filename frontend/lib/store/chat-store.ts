import { create } from "zustand"
import { devtools } from "zustand/middleware"
import type { Conversation, Message, Source } from "@/lib/types"
import {
  getConversations,
  getConversation,
  createConversation,
  deleteConversation as apiDeleteConversation,
  sendMessage as apiSendMessage,
  getDocuments,
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
}

interface ChatActions {
  // API actions
  loadConversations: () => Promise<void>
  loadDocumentCount: () => Promise<void>
  createConversation: () => Promise<void>
  selectConversation: (conversation: Conversation) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  sendMessage: (content: string) => Promise<void>

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

      // Create new conversation
      createConversation: async () => {
        const result = await createConversation()

        if (result.success) {
          const newConvo: Conversation = {
            id: result.data.id,
            title: result.data.title,
            messages: [],
            createdAt: new Date(result.data.createdAt),
            updatedAt: new Date(result.data.updatedAt),
          }

          set((state) => ({
            conversations: [newConvo, ...state.conversations],
            activeConversation: newConvo,
            selectedSources: [],
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
              createdAt: new Date(result.data.createdAt),
              updatedAt: new Date(result.data.updatedAt),
            }

            set((state) => ({
              activeConversation: fullConversation,
              conversations: state.conversations.map((c) =>
                c.id === fullConversation.id ? fullConversation : c
              ),
              isLoading: false,
            }))
          } else {
            // On error, still set as active but with empty messages
            set({
              activeConversation: { ...conversation, messages: [] },
              isLoading: false
            })
            toast.error(`Failed to load conversation: ${result.error}`)
          }
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
            createdAt: new Date(createResult.data.createdAt),
            updatedAt: new Date(createResult.data.updatedAt),
          }

          set((state) => ({
            conversations: [newConvo, ...state.conversations],
            activeConversation: newConvo,
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

        // Send to API
        const result = await apiSendMessage(parseInt(activeConversation.id), content)

        if (result.success) {
          const assistantMessage: Message = {
            id: result.data.messageId,
            role: "assistant",
            content: result.data.assistantMessage,
            sources: result.data.sources.map((url, idx) => ({
              id: `source-${idx}`,
              title: `Source ${idx + 1}`,
              content: url,
              relevance: 0.9,
              type: "document" as const,
            })),
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

            return {
              activeConversation: updatedConversation,
              conversations: updatedConversation
                ? state.conversations.map((c) =>
                    c.id === updatedConversation.id ? updatedConversation : c
                  )
                : state.conversations,
              selectedSources: assistantMessage.sources || [],
              isWaitingForResponse: false,
            }
          })
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
