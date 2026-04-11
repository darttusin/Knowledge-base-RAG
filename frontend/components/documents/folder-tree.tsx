"use client"

import React, { useState } from "react"
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

interface FolderNodeProps {
  folder: Folder
  level?: number
}

function FolderNode({ folder, level = 0 }: FolderNodeProps) {
  const currentFolderId = useFolderStore((s) => s.currentFolderId)
  const expandedFolders = useFolderStore((s) => s.expandedFolders)
  const allFolders = useFolderStore((s) => s.folders)
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)
  const toggleFolderExpanded = useFolderStore((s) => s.toggleFolderExpanded)
  const deleteFolder = useFolderStore((s) => s.deleteFolder)
  const updateFolder = useFolderStore((s) => s.updateFolder)

  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(folder.name)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)

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
    console.log('Selected folder:', { id: folder.id, name: folder.name, path: folder.path })
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

  return (
    <>
      <div className="select-none">
        <div
          className={cn(
            "hover:bg-muted/50 group flex items-center gap-1 rounded-lg px-2 py-1.5 transition-colors cursor-pointer",
            isSelected && "bg-primary/10 hover:bg-primary/15"
          )}
          style={{ paddingLeft: `${level * 16 + 8}px` }}
          onClick={handleSelect}
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
              <FolderNode key={child.id} folder={child} level={level + 1} />
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

export function FolderTree() {
  const folders = useFolderStore((s) => s.folders)
  const createFolder = useFolderStore((s) => s.createFolder)
  const currentFolderId = useFolderStore((s) => s.currentFolderId)
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)

  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [newFolderName, setNewFolderName] = useState("")

  // Get top-level folders (no parent)
  const rootFolders = folders.filter((f) => f.parentId === null)

  const handleCreateFolder = () => {
    if (newFolderName.trim()) {
      createFolder(newFolderName.trim(), currentFolderId)
      toast.success("Folder created")
      setNewFolderName("")
      setCreateDialogOpen(false)
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
                  currentFolderId === null && "bg-primary/10 hover:bg-primary/15"
                )}
                onClick={() => {
                  console.log('Selected: All Documents (currentFolderId = null)')
                  setCurrentFolder(null)
                }}
              >
                <div className="h-5 w-5 shrink-0" />
                <FileText className="text-muted-foreground h-4 w-4 shrink-0" />
                <span className="text-foreground flex-1 truncate text-sm font-medium">
                  All Documents
                </span>
              </div>

              {rootFolders.map((folder) => (
                <FolderNode key={folder.id} folder={folder} />
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
