"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Folder, MoreVertical, Edit, Trash2, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import type { Folder as FolderType } from "@/lib/store/folder-store"
import { useFolderStore } from "@/lib/store/folder-store"
import { DOCUMENT_DRAG_MIME } from "@/components/documents/document-row"
import { FOLDER_DRAG_MIME, collectDescendantIds } from "@/components/documents/folder-tree"

interface FolderRowProps {
  folder: FolderType
  onRename: (folder: FolderType) => void
  onDelete: (folder: FolderType) => void
  formatDate: (date: Date) => string
  onDocumentDrop?: (documentId: string, targetFolderId: string | null) => Promise<void> | void
  isSelected?: boolean
  onSelectClick?: (e: React.MouseEvent) => void
  getDragPayload?: () => { docIds: string[]; folderIds: string[] }
}

export function FolderRow({
  folder,
  onRename,
  onDelete,
  formatDate,
  onDocumentDrop,
  isSelected = false,
  onSelectClick,
  getDragPayload,
}: FolderRowProps) {
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)
  const allFolders = useFolderStore((s) => s.folders)
  const moveFolder = useFolderStore((s) => s.moveFolder)
  const [isDragging, setIsDragging] = useState(false)
  const [isDropTarget, setIsDropTarget] = useState(false)

  useEffect(() => {
    const reset = () => setIsDropTarget(false)
    window.addEventListener("dragend", reset)
    window.addEventListener("drop", reset)
    return () => {
      window.removeEventListener("dragend", reset)
      window.removeEventListener("drop", reset)
    }
  }, [])

  const handleClick = (e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey) {
      e.preventDefault()
      e.stopPropagation()
      onSelectClick?.(e)
      return
    }
    setCurrentFolder(folder.id)
  }

  const isDocumentDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(DOCUMENT_DRAG_MIME)
  const isFolderDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(FOLDER_DRAG_MIME)

  const isInvalidFolderDropTarget = (e: React.DragEvent): boolean => {
    if (!isFolderDrag(e)) return false
    const draggedIds =
      (window as unknown as { __draggedFolderIds?: string[] }).__draggedFolderIds ?? []
    if (draggedIds.length === 0) return false
    return draggedIds.some(
      (id) => id === folder.id || collectDescendantIds(allFolders, id).has(folder.id)
    )
  }

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
    const payload = getDragPayload?.() ?? { docIds: [], folderIds: [folder.id] }
    if (payload.docIds.length > 0) {
      e.dataTransfer.setData(DOCUMENT_DRAG_MIME, payload.docIds.join(","))
    }
    e.dataTransfer.setData(FOLDER_DRAG_MIME, payload.folderIds.join(","))
    const totalCount = payload.docIds.length + payload.folderIds.length
    e.dataTransfer.setData(
      "text/plain",
      totalCount > 1 ? `${totalCount} items` : folder.name
    )
    e.dataTransfer.effectAllowed = "move"
    // Track ALL dragged folder ids globally so drop targets can validate descendants for any of them
    ;(window as unknown as { __draggedFolderIds?: string[] }).__draggedFolderIds =
      payload.folderIds
    setIsDragging(true)
  }

  const handleDragEnd = () => {
    setIsDragging(false)
    delete (window as unknown as { __draggedFolderIds?: string[] }).__draggedFolderIds
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    const docDrag = onDocumentDrop && isDocumentDrag(e)
    const folderDrag = isFolderDrag(e) && !isInvalidFolderDropTarget(e)
    if (!docDrag && !folderDrag) return
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
    if (!isDropTarget) setIsDropTarget(true)
  }

  const handleDragLeave = () => {
    if (isDropTarget) setIsDropTarget(false)
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDropTarget(false)

    const docIds = (e.dataTransfer.getData(DOCUMENT_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)
    const folderIdsRaw = (e.dataTransfer.getData(FOLDER_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)

    // Skip self and descendant moves for folders
    const folderIds = folderIdsRaw.filter((id) => {
      if (id === folder.id) return false
      if (collectDescendantIds(allFolders, id).has(folder.id)) {
        toast.error(`Cannot move folder into its own descendant`)
        return false
      }
      return true
    })

    if (docIds.length === 0 && folderIds.length === 0) return

    const errors: string[] = []
    if (onDocumentDrop) {
      const docResults = await Promise.allSettled(
        docIds.map((id) => Promise.resolve(onDocumentDrop(id, folder.id)))
      )
      docResults.forEach((r) => {
        if (r.status === "rejected") errors.push(String(r.reason))
      })
    }
    const folderResults = await Promise.allSettled(
      folderIds.map((id) => moveFolder(id, folder.id))
    )
    folderResults.forEach((r) => {
      if (r.status === "rejected") errors.push(String(r.reason))
    })

    const totalMoved = docIds.length + folderIds.length - errors.length
    if (totalMoved > 0) {
      toast.success(
        totalMoved === 1
          ? `Moved to "${folder.name}"`
          : `Moved ${totalMoved} items to "${folder.name}"`
      )
    }
    if (errors.length > 0) {
      toast.error(`Failed to move ${errors.length} item${errors.length > 1 ? "s" : ""}`)
    }
  }

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "hover:bg-muted/30 group flex cursor-pointer items-center gap-4 rounded-xl p-3 transition-colors",
        isDragging && "opacity-40",
        isSelected && "bg-primary/15 hover:bg-primary/20",
        isDropTarget && "outline-primary bg-primary/10 outline-2 -outline-offset-2"
      )}
      onClick={handleClick}
    >
      <div
        className={cn(
          "bg-primary/10 text-primary flex h-10 w-10 shrink-0 items-center justify-center rounded-xl"
        )}
      >
        <Folder className="h-5 w-5" />
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-foreground truncate text-sm font-medium">{folder.name}</p>
        <div className="text-muted-foreground mt-0.5 flex items-center gap-2 text-xs">
          <Badge variant="secondary" className="bg-muted/50 border-0 px-1.5 py-0 text-[10px]">
            FOLDER
          </Badge>
          <span>
            {folder.documentCount} {folder.documentCount === 1 ? "document" : "documents"}
          </span>
          <span className="text-muted-foreground/50">•</span>
          <span>{formatDate(folder.createdAt)}</span>
        </div>
      </div>

      <div className="flex items-center gap-1">
        <ChevronRight className="text-muted-foreground h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        {folder.id !== "root" && (
          <div
            className="opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100"
            onClick={(e) => e.stopPropagation()}
          >
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground hover:text-foreground hover:bg-muted/50 h-8 w-8"
                  aria-label={`More actions for ${folder.name}`}
                >
                  <MoreVertical className="h-4 w-4" aria-hidden="true" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => onRename(folder)}>
                  <Edit className="mr-2 h-4 w-4" />
                  Rename
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive"
                  onClick={() => onDelete(folder)}
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>
    </div>
  )
}
