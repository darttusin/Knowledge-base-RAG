"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { useCopyFeedback } from "@/hooks/useCopyFeedback"
import {
  FileText,
  Database,
  Globe,
  Copy,
  Hash,
  CheckCircle2,
  Clock,
  Layers,
  ArrowRight,
} from "lucide-react"
import type { Source } from "@/lib/types"
import { extendedSourceData } from "@/lib/mock-data"
import { SOURCE_TYPE_COLORS, SOURCE_TYPE_LABELS } from "@/lib/constants"
import { cn } from "@/lib/utils"

interface SourceDetailModalProps {
  source: Source | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

const SOURCE_TYPE_ICONS = {
  document: FileText,
  database: Database,
  api: Globe,
}

export function SourceDetailModal({ source, open, onOpenChange }: SourceDetailModalProps) {
  const { copied, copy } = useCopyFeedback()

  if (!source) return null

  const extendedData = extendedSourceData[source.id] || {
    author: "Unknown",
    createdAt: new Date().toISOString().split("T")[0],
    updatedAt: new Date().toISOString().split("T")[0],
    path: "/unknown",
    tags: [],
    fullContent: source.content,
    relatedSources: [],
    metadata: [],
  }

  const Icon = SOURCE_TYPE_ICONS[source.type]
  const typeLabel = SOURCE_TYPE_LABELS[source.type]
  const colorClass = SOURCE_TYPE_COLORS[source.type]

  const handleCopy = () => copy(extendedData.fullContent)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-2xl gap-0 overflow-hidden p-0">
        <DialogHeader className="border-border bg-card/50 border-b p-6 pb-4">
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border",
                colorClass
              )}
            >
              <Icon className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <DialogTitle className="text-foreground text-lg font-semibold">
                {source.title}
              </DialogTitle>
              <p className="text-muted-foreground mt-0.5 text-sm">{typeLabel} Source</p>
            </div>
          </div>
        </DialogHeader>

        <ScrollArea className="max-h-[calc(85vh-120px)] flex-1">
          <div className="space-y-6 p-6">
            {/* Folder path if exists */}
            {source.folderPath && (
              <>
                <div className="flex items-center gap-2 text-sm min-w-0">
                  <Hash className="text-muted-foreground h-4 w-4 shrink-0" />
                  <span className="text-muted-foreground shrink-0">Folder:</span>
                  <code className="bg-muted text-foreground rounded px-2 py-0.5 font-mono text-xs break-all overflow-wrap-anywhere">
                    {source.folderPath}
                  </code>
                </div>
                <Separator />
              </>
            )}

            {/* Content */}
            <div>
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-foreground flex items-center gap-2 text-sm font-semibold">
                  <Layers className="text-muted-foreground h-4 w-4" />
                  Retrieved Content
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 gap-1.5 text-xs"
                  onClick={handleCopy}
                >
                  {copied ? (
                    <>
                      <CheckCircle2 className="text-chart-5 h-3 w-3" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      Copy
                    </>
                  )}
                </Button>
              </div>
              <div className="border-border bg-muted/30 rounded-xl border p-4 overflow-hidden max-w-full">
                <div className="text-foreground text-sm leading-relaxed max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                  {source.title.endsWith(".md") ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        h1: ({ children }) => <h1 className="text-base font-bold mb-2 mt-4 first:mt-0 max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</h1>,
                        h2: ({ children }) => <h2 className="text-sm font-semibold mb-2 mt-3 first:mt-0 max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0 max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</h3>,
                        p: ({ children }) => <p className="mb-2 max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</p>,
                        ul: ({ children }) => <ul className="list-disc ml-4 mb-2 max-w-full">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal ml-4 mb-2 max-w-full">{children}</ol>,
                        li: ({ children }) => <li className="mb-1 max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</li>,
                        code: ({ className, children }) => {
                          const isBlock = className?.includes('language-')
                          return isBlock ? (
                            <pre className="bg-muted/50 p-2 rounded mb-2 text-xs max-w-full whitespace-pre-wrap" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                              <code className="max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{children}</code>
                            </pre>
                          ) : (
                            <code className="bg-muted/50 px-1 py-0.5 rounded text-xs break-all max-w-full">{children}</code>
                          )
                        },
                        a: ({ href, children }) => (
                          <a href={href} className="text-primary underline break-all max-w-full" target="_blank" rel="noopener noreferrer">
                            {children}
                          </a>
                        ),
                      }}
                    >
                      {extendedData.fullContent}
                    </ReactMarkdown>
                  ) : (
                    <div className="whitespace-pre-wrap max-w-full" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{extendedData.fullContent}</div>
                  )}
                </div>
              </div>
            </div>

            {/* Metadata */}
            {extendedData.metadata.length > 0 && (
              <>
                <Separator />
                <div>
                  <h3 className="text-foreground mb-3 flex items-center gap-2 text-sm font-semibold">
                    <Clock className="text-muted-foreground h-4 w-4" />
                    Metadata
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    {extendedData.metadata.map((item) => (
                      <div key={item.label} className="border-border bg-card rounded-lg border p-3">
                        <p className="text-muted-foreground text-xs">{item.label}</p>
                        <p className="text-foreground mt-0.5 text-sm font-medium">{item.value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Related Sources */}
            {extendedData.relatedSources.length > 0 && (
              <>
                <Separator />
                <div>
                  <h3 className="text-foreground mb-3 text-sm font-semibold">Related Sources</h3>
                  <div className="space-y-2">
                    {extendedData.relatedSources.map((related) => {
                      const RelatedIcon = SOURCE_TYPE_ICONS[related.type]
                      const relatedColorClass = SOURCE_TYPE_COLORS[related.type]
                      return (
                        <button
                          key={related.id}
                          className="border-border bg-card hover:bg-muted/50 hover:border-primary/20 group flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-all"
                        >
                          <div
                            className={cn(
                              "flex h-8 w-8 items-center justify-center rounded-lg",
                              relatedColorClass
                            )}
                          >
                            <RelatedIcon className="h-3.5 w-3.5" />
                          </div>
                          <span className="text-foreground flex-1 text-sm font-medium">
                            {related.title}
                          </span>
                          <ArrowRight className="text-muted-foreground group-hover:text-primary h-4 w-4 transition-colors" />
                        </button>
                      )
                    })}
                  </div>
                </div>
              </>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
