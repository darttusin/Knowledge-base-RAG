"use client"

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { TooltipButton } from "@/components/tooltip-button"
import { ParagraphRenderer } from "@/components/paragraph-renderer"
import { User, Pencil, History, ChevronLeft, ChevronRight } from "lucide-react"
import type { Message } from "@/lib/types"
import { cn } from "@/lib/utils"

interface UserMessageBubbleProps {
  message: Message
  onEditMessage: (messageId: string, newContent: string) => void
}

export function UserMessageBubble({ message, onEditMessage }: UserMessageBubbleProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(message.content)
  const [showHistory, setShowHistory] = useState(false)
  const [historyIndex, setHistoryIndex] = useState(0)
  const [bubbleSize, setBubbleSize] = useState<{ width: number; height: number } | null>(null)
  const displayBubbleRef = useRef<HTMLDivElement>(null)
  const editTextareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSaveEdit = useCallback(() => {
    if (editContent.trim() && editContent !== message.content) {
      onEditMessage(message.id, editContent)
    }
    setIsEditing(false)
    setBubbleSize(null)
  }, [editContent, message.content, message.id, onEditMessage])

  const handleCancelEdit = useCallback(() => {
    setIsEditing(false)
    setEditContent(message.content)
    setBubbleSize(null)
  }, [message.content])

  // Sync editContent with message.content when not editing
  useEffect(() => {
    if (!isEditing) {
      setEditContent(message.content)
    }
  }, [message.content, isEditing])

  // Auto-resize textarea
  useEffect(() => {
    if (!isEditing || !editTextareaRef.current) return

    const textarea = editTextareaRef.current
    textarea.style.height = "0px"
    textarea.style.height = `${textarea.scrollHeight}px`
  }, [editContent, isEditing])

  // Global ESC handler for edit mode
  useEffect(() => {
    if (!isEditing) return

    const handleEscape = (e: Event) => {
      const keyEvent = e as globalThis.KeyboardEvent
      if (keyEvent.key === "Escape") {
        keyEvent.preventDefault()
        keyEvent.stopPropagation()
        setIsEditing(false)
        setEditContent(message.content)
        setBubbleSize(null)
      }
    }

    document.addEventListener("keydown", handleEscape, true)
    return () => document.removeEventListener("keydown", handleEscape, true)
  }, [isEditing, message.content])

  const handleEditKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      e.preventDefault()
      handleCancelEdit()
      return
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSaveEdit()
    }
  }

  const history = message.editHistory || []
  const hasHistory = history.length > 0

  const handleStartEdit = () => {
    if (displayBubbleRef.current) {
      const { width, height } = displayBubbleRef.current.getBoundingClientRect()
      setBubbleSize({ width, height })
    }
    setIsEditing(true)
  }

  return (
    <div className="flex justify-end gap-2 sm:gap-4">
      <div className="flex max-w-[85%] flex-1 justify-end sm:max-w-[80%]">
        <div className="space-y-2">
          {isEditing ? (
            <div
              className="bg-primary text-primary-foreground rounded-2xl px-3 py-2.5 sm:px-4 sm:py-3"
              style={{
                width: bubbleSize ? `${bubbleSize.width}px` : undefined,
                minHeight: bubbleSize ? `${bubbleSize.height}px` : undefined,
              }}
            >
              <Textarea
                ref={editTextareaRef}
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                onKeyDown={handleEditKeyDown}
                rows={1}
                className="text-primary-foreground min-h-[1.25rem] w-full resize-none overflow-hidden border-0 bg-transparent p-0 text-sm leading-relaxed shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
                autoFocus
              />
            </div>
          ) : (
            <>
              <div
                ref={displayBubbleRef}
                className="bg-primary text-primary-foreground rounded-2xl px-3 py-2.5 sm:px-4 sm:py-3"
              >
                <ParagraphRenderer
                  content={message.content}
                  className="text-sm leading-relaxed"
                  paragraphClassName="mb-1.5 last:mb-0"
                />
              </div>

              <div className="flex items-center justify-end gap-0.5 sm:gap-1">
                {message.isEdited && (
                  <span className="text-muted-foreground mr-1 text-xs">(edited)</span>
                )}
                {hasHistory && (
                  <TooltipButton
                    tooltip="View edit history"
                    variant="ghost"
                    size="icon"
                    className={cn(
                      "text-muted-foreground hover:text-foreground hover:bg-muted/50 h-7 w-7",
                      showHistory && "bg-muted text-foreground"
                    )}
                    onClick={() => setShowHistory(!showHistory)}
                  >
                    <History className="h-3.5 w-3.5" />
                  </TooltipButton>
                )}
                <TooltipButton
                  tooltip="Edit message"
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground hover:text-foreground hover:bg-muted/50 h-7 w-7"
                  onClick={handleStartEdit}
                >
                  <Pencil className="h-3.5 w-3.5" />
                </TooltipButton>
              </div>

              {/* Edit History Panel */}
              {showHistory && hasHistory && (
                <div className="bg-muted/30 space-y-2 rounded-xl p-2 sm:p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground text-xs font-medium">
                      Previous versions
                    </span>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="hover:bg-muted h-6 w-6"
                        onClick={() => setHistoryIndex((i) => Math.max(0, i - 1))}
                        disabled={historyIndex === 0}
                      >
                        <ChevronLeft className="h-3 w-3" />
                      </Button>
                      <span className="text-muted-foreground min-w-[30px] text-center text-xs">
                        {historyIndex + 1} / {history.length}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="hover:bg-muted h-6 w-6"
                        onClick={() => setHistoryIndex((i) => Math.min(history.length - 1, i + 1))}
                        disabled={historyIndex === history.length - 1}
                      >
                        <ChevronRight className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-2 sm:p-3">
                    <p className="text-foreground text-sm">{history[historyIndex].content}</p>
                    <p className="text-muted-foreground mt-2 text-xs">
                      {history[historyIndex].timestamp.toLocaleString()}
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <div className="bg-muted flex h-7 w-7 shrink-0 items-center justify-center rounded-full sm:h-8 sm:w-8">
        <User className="text-muted-foreground h-3.5 w-3.5 sm:h-4 sm:w-4" />
      </div>
    </div>
  )
}
