"use client"

import React, { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import {
  useFolderStore,
  selectFolderChildren,
  type Folder,
} from "@/lib/store/folder-store"
import {
  Folder as FolderIcon,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  MoreVertical,
  Plus,
  Edit,
  Trash,
  FileText,
} from "lucide-react"
import { toast } from "sonner"
import { DOCUMENT_DRAG_MIME } from "@/components/documents/document-row"

export const FOLDER_DRAG_MIME = "application/x-folder-id"

type DocumentDropHandler = (documentId: string, targetFolderId: string | null) => Promise<void> | void

interface FolderTreeProps {
  onDocumentDrop?: DocumentDropHandler
}

interface FolderNodeProps {
  folder: Folder
  level?: number
  onDocumentDrop?: DocumentDropHandler
}

/** Returns set of all descendant folder ids (not including the folder itself). */
export function collectDescendantIds(folders: Folder[], folderId: string): Set<string> {
  const childrenMap = new Map<string, Folder[]>()
  for (const f of folders) {
    if (f.parentId !== null) {
      const list = childrenMap.get(f.parentId) ?? []
      list.push(f)
      childrenMap.set(f.parentId, list)
    }
  }
  const descendants = new Set<string>()
  const stack: string[] = [folderId]
  while (stack.length > 0) {
    const current = stack.pop()!
    for (const child of childrenMap.get(current) ?? []) {
      if (!descendants.has(child.id)) {
        descendants.add(child.id)
        stack.push(child.id)
      }
    }
  }
  return descendants
}

function FolderNode({ folder, level = 0, onDocumentDrop }: FolderNodeProps) {
  const currentFolderId = useFolderStore((s) => s.currentFolderId)
  const expandedFolders = useFolderStore((s) => s.expandedFolders)
  const allFolders = useFolderStore((s) => s.folders)
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)
  const toggleFolderExpanded = useFolderStore((s) => s.toggleFolderExpanded)
  const deleteFolder = useFolderStore((s) => s.deleteFolder)
  const updateFolder = useFolderStore((s) => s.updateFolder)
  const moveFolder = useFolderStore((s) => s.moveFolder)

  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(folder.name)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [isDropTarget, setIsDropTarget] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  // Reset drop highlight whenever any drag ends globally (drop / cancel / escape).
  // dragleave alone is unreliable: it doesn't fire if the drag ends outside this row.
  useEffect(() => {
    const reset = () => setIsDropTarget(false)
    window.addEventListener("dragend", reset)
    window.addEventListener("drop", reset)
    return () => {
      window.removeEventListener("dragend", reset)
      window.removeEventListener("drop", reset)
    }
  }, [])

  // Memoize children to prevent re-filtering on every render
  const children = React.useMemo(
    () => allFolders.filter((f) => f.parentId === folder.id),
    [allFolders, folder.id]
  )

  const isExpanded = expandedFolders.includes(folder.id)
  const isSelected = currentFolderId === folder.id
  const hasChildren = children.length > 0

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (hasChildren) {
      toggleFolderExpanded(folder.id)
    }
  }

  const handleSelect = () => {
    setCurrentFolder(folder.id)
  }

  const handleRename = () => {
    if (editName.trim() && editName !== folder.name) {
      updateFolder(folder.id, editName.trim())
      toast.success("Folder renamed")
    }
    setIsEditing(false)
  }

  const handleDelete = () => {
    deleteFolder(folder.id)
    toast.success("Folder deleted")
    setDeleteDialogOpen(false)
  }

  const isDocumentDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(DOCUMENT_DRAG_MIME)
  const isFolderDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(FOLDER_DRAG_MIME)

  /** True when any of the dragged folders is this folder or one of its descendants. */
  const isInvalidFolderDropTarget = (e: React.DragEvent): boolean => {
    if (!isFolderDrag(e)) return false
    // dataTransfer.getData() is empty during dragover for security reasons,
    // so we rely on dragged folder ids passed via a window-level marker.
    const draggedIds =
      (window as unknown as { __draggedFolderIds?: string[] }).__draggedFolderIds ?? []
    if (draggedIds.length === 0) return false
    return draggedIds.some(
      (id) => id === folder.id || collectDescendantIds(allFolders, id).has(folder.id)
    )
  }

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
    e.stopPropagation()
    e.dataTransfer.setData(FOLDER_DRAG_MIME, folder.id)
    e.dataTransfer.setData("text/plain", folder.name)
    e.dataTransfer.effectAllowed = "move"
    ;(window as unknown as { __draggedFolderIds?: string[] }).__draggedFolderIds = [
      folder.id,
    ]
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
    e.stopPropagation()
    e.dataTransfer.dropEffect = "move"
    if (!isDropTarget) setIsDropTarget(true)
  }

  const handleDragLeave = () => {
    if (isDropTarget) setIsDropTarget(false)
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDropTarget(false)

    const docIds = (e.dataTransfer.getData(DOCUMENT_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)
    const folderIdsRaw = (e.dataTransfer.getData(FOLDER_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)

    const folderIds = folderIdsRaw.filter((id) => {
      if (id === folder.id) return false
      if (collectDescendantIds(allFolders, id).has(folder.id)) {
        toast.error("Cannot move folder into its own descendant")
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
    <>
      <div className="select-none">
        <div
          draggable
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          className={cn(
            "hover:bg-muted/50 group flex items-center gap-1 rounded-lg px-2 py-1.5 transition-colors cursor-pointer",
            isSelected && "bg-primary/10 hover:bg-primary/15",
            isDropTarget && "outline-primary bg-primary/10 outline-2 -outline-offset-2",
            isDragging && "opacity-40"
          )}
          style={{ paddingLeft: `${level * 16 + 8}px` }}
          onClick={handleSelect}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {hasChildren ? (
            <button
              onClick={handleToggle}
              className="hover:bg-muted flex h-5 w-5 shrink-0 items-center justify-center rounded transition-colors"
              aria-label={isExpanded ? "Collapse folder" : "Expand folder"}
            >
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
            </button>
          ) : (
            <div className="h-5 w-5 shrink-0" />
          )}

          {isExpanded ? (
            <FolderOpen className="text-muted-foreground h-4 w-4 shrink-0" />
          ) : (
            <FolderIcon className="text-muted-foreground h-4 w-4 shrink-0" />
          )}

          {isEditing ? (
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onBlur={handleRename}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleRename()
                if (e.key === "Escape") setIsEditing(false)
              }}
              className="text-foreground h-6 flex-1 border-0 bg-transparent px-1 py-0 text-sm focus-visible:ring-1"
              autoFocus
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className={cn(
              "flex-1 truncate text-sm font-medium",
              folder.documentCount === 0 ? "text-muted-foreground" : "text-foreground"
            )}>
              {folder.name}
            </span>
          )}

          <div className="ml-auto flex items-center gap-1">
            <span className={cn(
              "text-xs",
              folder.documentCount === 0 ? "text-muted-foreground/50" : "text-muted-foreground"
            )}>
              {folder.documentCount}
            </span>

            {true && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    aria-label="Folder options"
                  >
                    <MoreVertical className="h-3.5 w-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation()
                      setIsEditing(true)
                    }}
                  >
                    <Edit className="mr-2 h-4 w-4" />
                    Rename
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeleteDialogOpen(true)
                    }}
                  >
                    <Trash className="mr-2 h-4 w-4" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </div>

        {isExpanded && hasChildren && (
          <div>
            {children.map((child) => (
              <FolderNode
                key={child.id}
                folder={child}
                level={level + 1}
                onDocumentDrop={onDocumentDrop}
              />
            ))}
          </div>
        )}
      </div>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete folder</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{folder.name}&quot;? This will also delete all
              subfolders and documents inside. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

export function FolderTree({ onDocumentDrop }: FolderTreeProps = {}) {
  const folders = useFolderStore((s) => s.folders)
  const createFolder = useFolderStore((s) => s.createFolder)
  const moveFolder = useFolderStore((s) => s.moveFolder)
  const currentFolderId = useFolderStore((s) => s.currentFolderId)
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)

  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [newFolderName, setNewFolderName] = useState("")
  const [isRootDropTarget, setIsRootDropTarget] = useState(false)

  useEffect(() => {
    const reset = () => setIsRootDropTarget(false)
    window.addEventListener("dragend", reset)
    window.addEventListener("drop", reset)
    return () => {
      window.removeEventListener("dragend", reset)
      window.removeEventListener("drop", reset)
    }
  }, [])

  const isDocumentDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(DOCUMENT_DRAG_MIME)
  const isFolderDrag = (e: React.DragEvent) =>
    e.dataTransfer.types.includes(FOLDER_DRAG_MIME)

  const handleRootDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    const docDrag = onDocumentDrop && isDocumentDrag(e)
    const folderDrag = isFolderDrag(e)
    if (!docDrag && !folderDrag) return
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
    if (!isRootDropTarget) setIsRootDropTarget(true)
  }

  const handleRootDragLeave = () => {
    if (isRootDropTarget) setIsRootDropTarget(false)
  }

  const handleRootDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsRootDropTarget(false)

    const docIds = (e.dataTransfer.getData(DOCUMENT_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)
    const folderIds = (e.dataTransfer.getData(FOLDER_DRAG_MIME) || "")
      .split(",")
      .filter(Boolean)

    if (docIds.length === 0 && folderIds.length === 0) return

    const errors: string[] = []
    if (onDocumentDrop) {
      const docResults = await Promise.allSettled(
        docIds.map((id) => Promise.resolve(onDocumentDrop(id, null)))
      )
      docResults.forEach((r) => {
        if (r.status === "rejected") errors.push(String(r.reason))
      })
    }
    const folderResults = await Promise.allSettled(
      folderIds.map((id) => moveFolder(id, null))
    )
    folderResults.forEach((r) => {
      if (r.status === "rejected") errors.push(String(r.reason))
    })

    const totalMoved = docIds.length + folderIds.length - errors.length
    if (totalMoved > 0) {
      toast.success(
        totalMoved === 1 ? "Moved to All Documents" : `Moved ${totalMoved} items to All Documents`
      )
    }
    if (errors.length > 0) {
      toast.error(`Failed to move ${errors.length} item${errors.length > 1 ? "s" : ""}`)
    }
  }

  // Get top-level folders (no parent)
  const rootFolders = folders.filter((f) => f.parentId === null)

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return
    try {
      await createFolder(newFolderName.trim(), currentFolderId)
      toast.success("Folder created")
      setNewFolderName("")
      setCreateDialogOpen(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create folder")
    }
  }

  return (
    <>
      <div className="flex h-full flex-col gap-2 overflow-hidden">
        <div className="flex items-center justify-between px-2 flex-shrink-0">
          <div className="flex items-center gap-2">
            <FolderIcon className="text-muted-foreground h-4 w-4" />
            <span className="text-foreground text-sm font-semibold">Folders</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setCreateDialogOpen(true)}
            aria-label="Create new folder"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full">
            <div className="space-y-1 pr-4">
              {/* All Documents button */}
              <div
                className={cn(
                  "hover:bg-muted/50 group flex items-center gap-1 rounded-lg px-2 py-1.5 transition-colors cursor-pointer",
                  currentFolderId === null && "bg-primary/10 hover:bg-primary/15",
                  isRootDropTarget && "outline-primary bg-primary/10 outline-2 -outline-offset-2"
                )}
                onClick={() => {
                  setCurrentFolder(null)
                }}
                onDragOver={handleRootDragOver}
                onDragLeave={handleRootDragLeave}
                onDrop={handleRootDrop}
              >
                <div className="h-5 w-5 shrink-0" />
                <FileText className="text-muted-foreground h-4 w-4 shrink-0" />
                <span className="text-foreground flex-1 truncate text-sm font-medium">
                  All Documents
                </span>
              </div>

              {rootFolders.map((folder) => (
                <FolderNode key={folder.id} folder={folder} onDocumentDrop={onDocumentDrop} />
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create new folder</DialogTitle>
            <DialogDescription>Enter a name for the new folder.</DialogDescription>
          </DialogHeader>
          <Input
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            placeholder="Folder name"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreateFolder()
            }}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateFolder} disabled={!newFolderName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
