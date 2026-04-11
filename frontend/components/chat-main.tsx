"use client"

import { useState } from "react"
import { ChatHeader } from "@/components/chat/chat-header"
import { MessageList } from "@/components/chat/message-list"
import { MessageInput } from "@/components/chat/message-input"
import { SourceDetailModal } from "@/components/source-detail-modal"
import { useChatStore } from "@/lib/store/chat-store"
import type { Source } from "@/lib/types"

export interface ChatMainProps {
  onSourceClick?: (source: Source) => void
}

export function ChatMain({ onSourceClick: externalSourceClick }: ChatMainProps) {
  const [selectedSource, setSelectedSource] = useState<Source | null>(null)
  const [sourceModalOpen, setSourceModalOpen] = useState(false)
  const sendMessage = useChatStore((s) => s.sendMessage)

  const handleSourceClick = (source: Source) => {
    setSelectedSource(source)
    setSourceModalOpen(true)
    externalSourceClick?.(source)
  }

  const handleSuggestionClick = async (suggestion: string) => {
    // Send message will auto-create conversation if needed
    await sendMessage(suggestion)
  }

  return (
    <div
      className="from-background to-muted/30 flex min-h-0 flex-1 flex-col overflow-hidden bg-linear-to-b"
      role="main"
    >
      <ChatHeader />

      <MessageList onSourceClick={handleSourceClick} onSuggestionClick={handleSuggestionClick} />

      <MessageInput />

      <SourceDetailModal
        source={selectedSource}
        open={sourceModalOpen}
        onOpenChange={setSourceModalOpen}
      />
    </div>
  )
}
