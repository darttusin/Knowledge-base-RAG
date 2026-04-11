"use client"

import type React from "react"
import { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { SourceBadge } from "./source-badge"
import { CodeBlock } from "@/components/code-block"
import { useCopyFeedback } from "@/hooks/useCopyFeedback"
import { Bot, Copy, Check, ThumbsUp, ThumbsDown, RefreshCw } from "lucide-react"
import type { Message, Source } from "@/lib/types"
import { cn } from "@/lib/utils"

interface AssistantMessageBubbleProps {
  message: Message
  onSourceClick: (source: Source) => void
}

export function AssistantMessageBubble({ message, onSourceClick }: AssistantMessageBubbleProps) {
  const [feedback, setFeedback] = useState<"like" | "dislike" | null>(null)
  const { copied, copy } = useCopyFeedback()

  const handleCopy = () => copy(message.content)

  const handleFeedback = (type: "like" | "dislike") => {
    setFeedback(feedback === type ? null : type)
  }

  // Render markdown with custom code block component
  const renderContent = () => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "")
            const language = match ? match[1] : "javascript"
            const codeString = String(children).replace(/\n$/, "")

            return !inline ? (
              <CodeBlock code={codeString} language={language} />
            ) : (
              <code
                className="bg-muted text-foreground rounded px-1.5 py-0.5 text-sm font-mono before:content-none after:content-none"
                {...props}
              >
                {children}
              </code>
            )
          },
        }}
      >
        {message.content}
      </ReactMarkdown>
    )
  }

  return (
    <div className="flex gap-2 sm:gap-4">
      <div className="from-primary to-primary/80 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-linear-to-br sm:h-8 sm:w-8">
        <Bot className="text-primary-foreground h-3.5 w-3.5 sm:h-4 sm:w-4" />
      </div>
      <div className="min-w-0 flex-1 pt-0.5">
        <div
          className={cn(
            "prose prose-sm dark:prose-invert max-w-none",
            "prose-p:leading-relaxed prose-p:my-2 prose-p:text-foreground",
            "prose-headings:font-semibold prose-headings:text-foreground",
            "prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg",
            "prose-strong:font-semibold prose-strong:text-foreground",
            "prose-em:italic prose-em:text-foreground",
            "prose-a:text-primary prose-a:no-underline hover:prose-a:underline",
            "prose-code:bg-muted prose-code:text-foreground prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none",
            "prose-pre:bg-transparent prose-pre:p-0 prose-pre:m-0",
            "prose-blockquote:border-l-primary prose-blockquote:border-l-4 prose-blockquote:italic prose-blockquote:text-foreground/80",
            "prose-ul:list-disc prose-ul:text-foreground prose-ol:list-decimal prose-ol:text-foreground",
            "prose-li:text-foreground prose-li:my-1",
            "prose-table:border prose-table:border-border",
            "prose-th:bg-muted prose-th:border prose-th:border-border prose-th:px-3 prose-th:py-2",
            "prose-td:border prose-td:border-border prose-td:px-3 prose-td:py-2",
            "prose-hr:border-border"
          )}
        >
          {renderContent()}
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="border-border/50 mt-3 border-t pt-3 sm:mt-4 sm:pt-4">
            <p className="text-muted-foreground mb-2 text-xs font-medium">Referenced sources</p>
            <div className="flex flex-wrap gap-1.5 sm:gap-2">
              {message.sources.map((source) => (
                <SourceBadge
                  key={source.id}
                  source={source}
                  onClick={() => onSourceClick(source)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div
          className="mt-2 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 sm:mt-3 sm:gap-1"
          style={{ opacity: 1 }}
        >
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-7 w-7 transition-all duration-200 sm:h-8 sm:w-8",
                    copied
                      ? "text-green-500 hover:bg-green-500/10 hover:text-green-500"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  )}
                  onClick={handleCopy}
                >
                  {copied ? (
                    <Check className="animate-in zoom-in-50 h-3.5 w-3.5 duration-200 sm:h-4 sm:w-4" />
                  ) : (
                    <Copy className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{copied ? "Copied!" : "Copy"}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-7 w-7 transition-all duration-200 sm:h-8 sm:w-8",
                    feedback === "like"
                      ? "scale-110 text-green-500 hover:bg-green-500/10 hover:text-green-500"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  )}
                  onClick={() => handleFeedback("like")}
                >
                  <ThumbsUp
                    className={cn(
                      "h-3.5 w-3.5 transition-transform duration-200 sm:h-4 sm:w-4",
                      feedback === "like" && "animate-in zoom-in-50 fill-current"
                    )}
                  />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {feedback === "like" ? "Thanks for feedback!" : "Good response"}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-7 w-7 transition-all duration-200 sm:h-8 sm:w-8",
                    feedback === "dislike"
                      ? "scale-110 text-red-500 hover:bg-red-500/10 hover:text-red-500"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  )}
                  onClick={() => handleFeedback("dislike")}
                >
                  <ThumbsDown
                    className={cn(
                      "h-3.5 w-3.5 transition-transform duration-200 sm:h-4 sm:w-4",
                      feedback === "dislike" && "animate-in zoom-in-50 fill-current"
                    )}
                  />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {feedback === "dislike" ? "Thanks for feedback!" : "Bad response"}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground hover:text-foreground hover:bg-muted/50 h-7 w-7 transition-all duration-200 active:rotate-180 sm:h-8 sm:w-8"
                >
                  <RefreshCw className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Regenerate</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </div>
  )
}
