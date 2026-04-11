"use client"

import { useEffect, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { useChatStore } from "@/lib/store/chat-store"
import { useFolderStore } from "@/lib/store/folder-store"
import { ChatSidebar } from "@/components/chat-sidebar"
import { ChatMain } from "@/components/chat-main"
import { SourcesPanel } from "@/components/sources-panel"
import { isAuthenticated } from "@/lib/auth"

function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const showSources = useChatStore((s) => s.showSources)
  const conversations = useChatStore((s) => s.conversations)
  const activeConversation = useChatStore((s) => s.activeConversation)
  const isLoading = useChatStore((s) => s.isLoading)
  const loadConversations = useChatStore((s) => s.loadConversations)
  const loadDocumentCount = useChatStore((s) => s.loadDocumentCount)
  const selectConversation = useChatStore((s) => s.selectConversation)
  const loadFolders = useFolderStore((s) => s.loadFolders)

  // Check authentication and load data
  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login")
      return
    }

    // Load conversations, folders, and document count from API
    loadConversations()
    loadFolders()
    loadDocumentCount()
  }, [router, loadConversations, loadFolders, loadDocumentCount])

  // Handle URL-driven conversation selection
  useEffect(() => {
    if (conversations.length === 0 || isLoading) return

    const urlChatId = searchParams.get("chat")

    if (urlChatId) {
      // URL has chat ID - select if different from current
      if (!activeConversation || activeConversation.id !== urlChatId) {
        const conversation = conversations.find((c) => c.id === urlChatId)
        if (conversation) {
          selectConversation(conversation)
        }
      }
    } else if (!activeConversation && conversations.length > 0) {
      // No URL and no active conversation - select first and update URL
      const firstConv = conversations[0]
      router.replace(`/?chat=${firstConv.id}`, { scroll: false })
    }
    // Only depend on conversations.length and searchParams to avoid loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversations.length, searchParams, isLoading])

  // Update URL when active conversation changes (from user interaction)
  useEffect(() => {
    if (!activeConversation) return

    const urlChatId = searchParams.get("chat")
    if (urlChatId !== activeConversation.id) {
      router.replace(`/?chat=${activeConversation.id}`, { scroll: false })
    }
    // Only depend on activeConversation.id to avoid loops with searchParams
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversation?.id])

  return (
    <div className="bg-background flex h-screen overflow-hidden">
      <ChatSidebar />
      <ChatMain />
      {showSources && <SourcesPanel />}
    </div>
  )
}

export default function Home() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center">Loading...</div>}>
      <HomeContent />
    </Suspense>
  )
}
