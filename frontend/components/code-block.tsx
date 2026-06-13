"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import {
  Check,
  Copy,
  Play,
  Loader2,
  XCircle,
  CheckCircle2,
  Square,
  Pencil,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { executeCode, isExecutableLanguage } from "@/lib/code-executor"
import { tokenize, TOKEN_COLORS, PLOT_DATA_REGEX } from "@/lib/syntax-highlighting"
import { useCopyFeedback } from "@/hooks/useCopyFeedback"

interface CodeBlockProps {
  code: string
  language?: string
}

type ExecutionStatus = "idle" | "running" | "success" | "error"

// Highlighted code component using React elements
function HighlightedCode({ code, language }: { code: string; language: string }) {
  const tokens = tokenize(code, language)

  return (
    <code className="font-mono text-sm">
      {tokens.map((token, i) => (
        <span key={i} className={TOKEN_COLORS[token.type]}>
          {token.value}
        </span>
      ))}
    </code>
  )
}

// Component to render output with support for matplotlib plots
function OutputRenderer({ output }: { output: string }) {
  // Check for embedded plot data
  const plotMatch = output.match(PLOT_DATA_REGEX)

  if (plotMatch) {
    const base64Data = plotMatch[1]
    const textOutput = output.replace(/\[PLOT_DATA\][\s\S]*?\[\/PLOT_DATA\]/, "").trim()

    return (
      <div className="mt-1 space-y-2">
        {textOutput && (
          <div className="overflow-x-auto">
            <pre className="font-mono text-xs whitespace-pre text-neutral-300">{textOutput}</pre>
          </div>
        )}
        <div className="overflow-hidden rounded-lg bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${base64Data}`}
            alt="Plot output"
            className="h-auto max-w-full"
          />
        </div>
      </div>
    )
  }

  return (
    <div className="mt-1 overflow-x-auto">
      <pre className="font-mono text-xs whitespace-pre text-neutral-300">{output}</pre>
    </div>
  )
}

export function CodeBlock({ code, language = "javascript" }: CodeBlockProps) {
  const { copied, copy } = useCopyFeedback()
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus>("idle")
  const [executionOutput, setExecutionOutput] = useState<string>("")
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  // Editable code state. `currentCode` is what gets copied/run; it starts as the
  // model's output and follows streaming updates until the user edits it.
  const [currentCode, setCurrentCode] = useState(code)
  const [isEditing, setIsEditing] = useState(false)
  const [draftCode, setDraftCode] = useState(code)
  const [isEdited, setIsEdited] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // While the message is still streaming, keep syncing with the incoming code —
  // but stop once the user has made their own edits.
  useEffect(() => {
    if (!isEditing && !isEdited) {
      setCurrentCode(code)
      setDraftCode(code)
    }
  }, [code, isEditing, isEdited])

  const canExecute = isExecutableLanguage(language)

  const handleCopy = () => copy(currentCode)

  const startEditing = () => {
    setDraftCode(currentCode)
    setIsEditing(true)
  }

  const cancelEditing = () => {
    setDraftCode(currentCode)
    setIsEditing(false)
  }

  const saveEditing = () => {
    setCurrentCode(draftCode)
    setIsEdited(draftCode !== code)
    setIsEditing(false)
  }

  // Focus the textarea and place the caret at the end when entering edit mode.
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      const el = textareaRef.current
      el.focus()
      el.setSelectionRange(el.value.length, el.value.length)
    }
  }, [isEditing])

  const handleEditorKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      e.preventDefault()
      cancelEditing()
      return
    }
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      saveEditing()
      return
    }
    // Insert 4 spaces on Tab instead of leaving the field.
    if (e.key === "Tab") {
      e.preventDefault()
      const el = e.currentTarget
      const start = el.selectionStart
      const end = el.selectionEnd
      const next = draftCode.slice(0, start) + "    " + draftCode.slice(end)
      setDraftCode(next)
      requestAnimationFrame(() => {
        el.selectionStart = el.selectionEnd = start + 4
      })
    }
  }

  const handleStop = () => {
    if (abortController) {
      abortController.abort()
      setAbortController(null)
      setExecutionStatus("error")
      setExecutionOutput("Execution stopped by user")
    }
  }

  const handleRun = async () => {
    if (!canExecute) {
      setExecutionStatus("error")
      setExecutionOutput(`Language "${language}" is not supported for execution`)
      return
    }

    setExecutionStatus("running")
    setExecutionOutput("")

    const controller = new AbortController()
    setAbortController(controller)

    try {
      const result = await executeCode(currentCode, language)

      // Check if aborted
      if (controller.signal.aborted) return

      setExecutionStatus(result.status)
      setExecutionOutput(result.output)
    } catch (error) {
      if (controller.signal.aborted) return

      setExecutionStatus("error")
      setExecutionOutput(error instanceof Error ? error.message : String(error))
    } finally {
      setAbortController(null)
    }
  }

  return (
    <div className="border-border/50 my-4 overflow-hidden rounded-xl border bg-neutral-900 dark:bg-neutral-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-700/50 bg-neutral-800 px-4 py-2 dark:bg-neutral-900">
        <span className="flex items-center gap-2 text-xs font-medium tracking-wide text-neutral-400">
          {language}
          {isEdited && !isEditing && (
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
              edited
            </span>
          )}
        </span>
        <div className="flex items-center gap-0.5">
          {isEditing ? (
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-emerald-400 hover:bg-neutral-700/50 hover:text-emerald-300"
                      onClick={saveEditing}
                    >
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Save changes (⌘/Ctrl+Enter)</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-neutral-400 hover:bg-neutral-700/50 hover:text-neutral-200"
                      onClick={cancelEditing}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Cancel (Esc)</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          ) : (
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-neutral-400 hover:bg-neutral-700/50 hover:text-neutral-200"
                      onClick={startEditing}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Edit code</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-neutral-400 hover:bg-neutral-700/50 hover:text-neutral-200"
                      onClick={handleCopy}
                    >
                      {copied ? (
                        <Check className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{copied ? "Copied!" : "Copy code"}</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          )}
          {canExecute && !isEditing && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  {executionStatus === "running" ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-blue-400 hover:bg-neutral-700/50 hover:text-red-400"
                      onClick={handleStop}
                    >
                      <Square className="h-3 w-3 fill-current" />
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "h-7 w-7",
                        executionStatus === "success" && "text-emerald-400",
                        executionStatus === "error" && "text-red-400",
                        executionStatus === "idle" &&
                          "text-neutral-400 hover:bg-neutral-700/50 hover:text-neutral-200"
                      )}
                      onClick={handleRun}
                    >
                      <Play className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {executionStatus === "running" ? "Stop" : "Run code"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>

      {/* Code */}
      {isEditing ? (
        <textarea
          ref={textareaRef}
          value={draftCode}
          onChange={(e) => setDraftCode(e.target.value)}
          onKeyDown={handleEditorKeyDown}
          spellCheck={false}
          rows={Math.min(Math.max(draftCode.split("\n").length, 3), 30)}
          className="w-full resize-y border-0 bg-neutral-900 p-4 font-mono text-sm text-neutral-100 outline-none focus:ring-1 focus:ring-blue-500/40 dark:bg-neutral-950"
        />
      ) : (
        <div className="overflow-x-auto">
          <pre className="min-w-fit p-4">
            <HighlightedCode code={currentCode} language={language} />
          </pre>
        </div>
      )}

      {/* Execution Result */}
      {executionStatus !== "idle" && (
        <div
          className={cn(
            "flex items-start gap-2 border-t border-neutral-700/50 px-4 py-3",
            executionStatus === "success" && "bg-emerald-500/10",
            executionStatus === "error" && "bg-red-500/10",
            executionStatus === "running" && "bg-blue-500/10"
          )}
        >
          {executionStatus === "running" && (
            <>
              <Loader2 className="mt-0.5 h-4 w-4 animate-spin text-blue-400" />
              <span className="text-sm text-blue-400">Running...</span>
            </>
          )}
          {executionStatus === "success" && (
            <>
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <div className="min-w-0 flex-1">
                <span className="text-sm font-medium text-emerald-400">Output</span>
                <OutputRenderer output={executionOutput} />
              </div>
            </>
          )}
          {executionStatus === "error" && (
            <>
              <XCircle className="mt-0.5 h-4 w-4 text-red-400" />
              <div className="min-w-0 flex-1">
                <span className="text-sm font-medium text-red-400">Error</span>
                <div className="mt-1 overflow-x-auto">
                  <pre className="font-mono text-xs whitespace-pre text-red-400/80">
                    {executionOutput}
                  </pre>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
