"use client"

import { Suspense, useState, useEffect, useRef, useCallback } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { EmptyState } from "@/components/empty-state"
import {
  DocumentRow,
  FolderRow,
  UploadArea,
  DocumentStats,
  FolderTree,
  FolderBreadcrumb,
  MarkdownPreview,
} from "@/components/documents"
import { ThemeToggle } from "@/components/theme-toggle"
import { useDocuments, type Document } from "@/hooks/useDocuments"
import { useDateFormat } from "@/hooks/useDateFormat"
import type { SearchIn } from "@/lib/api"
import {
  useFolderStore,
  selectCurrentFolder,
  type Folder,
} from "@/lib/store/folder-store"
import { ArrowLeft, FileText, Search, Plus, FolderOpen, Download, Loader2 } from "lucide-react"
import { toast } from "sonner"
import Loading from "./loading"

export default function DocumentsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const documentIdFromUrl = searchParams.get("id")
  const hasProcessedDocId = useRef(false)

  const folders = useFolderStore((s) => s.folders)
  const currentFolderId = useFolderStore((s) => s.currentFolderId)
  const currentFolder = useFolderStore(selectCurrentFolder)
  const loadFolders = useFolderStore((s) => s.loadFolders)
  const updateFolder = useFolderStore((s) => s.updateFolder)
  const deleteFolder = useFolderStore((s) => s.deleteFolder)
  const setCurrentFolder = useFolderStore((s) => s.setCurrentFolder)

  // Load folders on mount
  useEffect(() => {
    loadFolders()
  }, [loadFolders])

  const {
    documents,
    selectedDocument,
    previewOpen,
    deleteDialogOpen,
    documentToDelete,
    isDragging,
    fileInputRef,
    totalSize,
    totalCount,
    hasMore,
    isLoadingMore,
    loadMore,
    searchQuery,
    setSearchQuery,
    searchIn,
    setSearchIn,
    uploadFiles,
    deleteDocument,
    renameDocument,
    openPreview,
    closePreview,
    confirmDelete,
    cancelDelete,
    downloadDocument,
    openFilePicker,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    findDocumentById,
  } = useDocuments(currentFolderId)

  // Handle document ID from URL (only once)
  useEffect(() => {
    if (documentIdFromUrl && !hasProcessedDocId.current) {
      const doc = findDocumentById(documentIdFromUrl)
      if (doc) {
        hasProcessedDocId.current = true
        // Set current folder to document's folder
        if (doc.folderId && doc.folderId !== currentFolderId) {
          setCurrentFolder(doc.folderId)
        }
        // Open preview
        openPreview(doc)
        // Clear the URL parameter after opening
        router.replace("/documents", { scroll: false })
      } else if (documents.length > 0) {
        // Document not found after documents loaded, clear the parameter
        hasProcessedDocId.current = true
        router.replace("/documents", { scroll: false })
      }
    }
  }, [documentIdFromUrl, documents, findDocumentById, openPreview, setCurrentFolder, currentFolderId, router])

  // Folder dialogs state
  const [renameFolderDialogOpen, setRenameFolderDialogOpen] = useState(false)
  const [deleteFolderDialogOpen, setDeleteFolderDialogOpen] = useState(false)
  const [folderToEdit, setFolderToEdit] = useState<Folder | null>(null)
  const [newFolderName, setNewFolderName] = useState("")

  // Document rename dialog state
  const [renameDocumentDialogOpen, setRenameDocumentDialogOpen] = useState(false)
  const [documentToRename, setDocumentToRename] = useState<Document | null>(null)
  const [newDocumentName, setNewDocumentName] = useState("")

  // Get child folders of current folder
  const childFolders = folders.filter((f) => f.parentId === currentFolderId)

  const { formatRelativeDate, formatFileSize } = useDateFormat()

  // Search mode options
  const searchModes: { value: SearchIn; label: string }[] = [
    { value: "both", label: "All" },
    { value: "name", label: "Name" },
    { value: "content", label: "Content" },
  ]

  // Infinite scroll handler
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  const handleScroll = useCallback(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea || isLoadingMore || !hasMore) return

    const scrollPosition = scrollArea.scrollTop + scrollArea.clientHeight
    const scrollHeight = scrollArea.scrollHeight

    // Load more when 80% scrolled
    if (scrollPosition >= scrollHeight * 0.8) {
      loadMore()
    }
  }, [isLoadingMore, hasMore, loadMore])

  useEffect(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return

    scrollArea.addEventListener('scroll', handleScroll)
    return () => scrollArea.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  // Folder actions
  const handleRenameFolder = (folder: Folder) => {
    setFolderToEdit(folder)
    setNewFolderName(folder.name)
    setRenameFolderDialogOpen(true)
  }

  const handleDeleteFolder = (folder: Folder) => {
    setFolderToEdit(folder)
    setDeleteFolderDialogOpen(true)
  }

  const confirmRenameFolder = () => {
    if (folderToEdit && newFolderName.trim() && newFolderName !== folderToEdit.name) {
      updateFolder(folderToEdit.id, newFolderName.trim())
      toast.success("Folder renamed")
    }
    setRenameFolderDialogOpen(false)
    setFolderToEdit(null)
    setNewFolderName("")
  }

  const confirmDeleteFolder = () => {
    if (folderToEdit) {
      deleteFolder(folderToEdit.id)
      toast.success("Folder deleted")
    }
    setDeleteFolderDialogOpen(false)
    setFolderToEdit(null)
  }

  const cancelFolderAction = () => {
    setRenameFolderDialogOpen(false)
    setDeleteFolderDialogOpen(false)
    setFolderToEdit(null)
    setNewFolderName("")
  }

  // Document rename actions
  const handleRenameDocument = (doc: Document) => {
    setDocumentToRename(doc)
    setNewDocumentName(doc.name)
    setRenameDocumentDialogOpen(true)
  }

  const confirmRenameDocument = () => {
    if (documentToRename && newDocumentName.trim() && newDocumentName !== documentToRename.name) {
      renameDocument(documentToRename.id, newDocumentName.trim())
      toast.success("Document renamed")
    }
    setRenameDocumentDialogOpen(false)
    setDocumentToRename(null)
    setNewDocumentName("")
  }

  const cancelRenameDocument = () => {
    setRenameDocumentDialogOpen(false)
    setDocumentToRename(null)
    setNewDocumentName("")
  }

  return (
    <Suspense fallback={<Loading />}>
      <div className="bg-background flex h-screen flex-col">
        {/* Header */}
        <header className="border-border/50 flex items-center justify-between border-b px-6 py-4">
          <div className="flex items-center gap-4">
            <Link href="/" aria-label="Go back to chat">
              <Button
                variant="ghost"
                size="icon"
                className="hover:bg-muted/50 h-8 w-8"
                aria-label="Back to chat"
              >
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              </Button>
            </Link>
            <div>
              <h1 className="text-foreground text-xl font-semibold">Documents</h1>
              <p className="text-muted-foreground text-sm">
                {currentFolder?.name || "All Documents"} - {totalCount} documents
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <Button onClick={openFilePicker} className="gap-2" aria-label="Upload new document">
              <Plus className="h-4 w-4" aria-hidden="true" />
              Upload
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md,.txt"
              multiple
              className="hidden"
              onChange={(e) => uploadFiles(e.target.files)}
            />
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Folder Sidebar */}
          <div className="border-border/50 flex w-64 shrink-0 flex-col border-r overflow-hidden">
            <div className="flex h-full flex-col p-4">
              <FolderTree />
            </div>
          </div>

          {/* Documents Area */}
          <div className="flex-1 overflow-hidden p-6">
            <div className="mx-auto flex h-full max-w-4xl flex-col">
            {/* Stats */}
            <DocumentStats
              documentCount={totalCount}
              totalSize={formatFileSize(totalSize)}
              lastUpload={
                documents.length > 0 ? formatRelativeDate(documents[0].uploadedAt) : "N/A"
              }
            />

            {/* Breadcrumb */}
            <FolderBreadcrumb />

            {/* Upload Area */}
            <UploadArea
              isDragging={isDragging}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onBrowse={openFilePicker}
            />

            {/* Search */}
            <div className="mb-4 space-y-2" role="search">
              <div className="relative">
                <Search
                  className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2"
                  aria-hidden="true"
                />
                <Input
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-muted/30 focus-visible:ring-primary/30 border-0 pl-9 focus-visible:ring-1"
                  aria-label="Search documents"
                />
              </div>
              {/* Search mode selector */}
              <div className="flex gap-1">
                {searchModes.map((mode) => (
                  <Button
                    key={mode.value}
                    variant={searchIn === mode.value ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSearchIn(mode.value)}
                    className="text-xs"
                  >
                    {mode.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* Documents Count */}
            {totalCount > 0 && (
              <div className="mb-2 text-muted-foreground text-sm">
                Loaded {documents.length} of {totalCount} documents
              </div>
            )}

            {/* Documents List */}
            <div ref={scrollAreaRef} className="flex-1 overflow-y-auto" role="region" aria-label="Documents list">
              {!searchQuery && childFolders.length === 0 && documents.length === 0 ? (
                <EmptyState
                  icon={FolderOpen}
                  title="No documents yet"
                  description="Upload your first document to get started"
                  className="py-16"
                  iconClassName="h-16 w-16 rounded-2xl mb-4"
                />
              ) : searchQuery && documents.length === 0 ? (
                <EmptyState
                  icon={FolderOpen}
                  title="No documents found"
                  description="Try a different search term or mode"
                  className="py-16"
                  iconClassName="h-16 w-16 rounded-2xl mb-4"
                />
              ) : (
                <div className="space-y-1">
                  {/* Show folders first (only when not searching) */}
                  {!searchQuery &&
                    childFolders.map((folder) => (
                      <FolderRow
                        key={folder.id}
                        folder={folder}
                        onRename={handleRenameFolder}
                        onDelete={handleDeleteFolder}
                        formatDate={formatRelativeDate}
                      />
                    ))}

                  {/* Then show documents */}
                  {documents.map((doc) => (
                    <DocumentRow
                      key={doc.id}
                      document={doc}
                      onPreview={openPreview}
                      onDownload={downloadDocument}
                      onRename={handleRenameDocument}
                      onDelete={confirmDelete}
                      formatFileSize={formatFileSize}
                      formatDate={formatRelativeDate}
                    />
                  ))}
                </div>
              )}

              {/* Loading indicator */}
              {isLoadingMore && (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  <span className="ml-2 text-sm text-muted-foreground">Loading more...</span>
                </div>
              )}

              {/* End of list indicator */}
              {!hasMore && documents.length > 0 && (
                <div className="text-center py-6 text-sm text-muted-foreground">
                  All documents loaded
                </div>
              )}
            </div>
            </div>
          </div>
        </div>

        {/* Preview Dialog */}
        <Dialog open={previewOpen} onOpenChange={closePreview}>
          <DialogContent className="max-w-[98vw] w-[1800px]">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                {selectedDocument?.name}
              </DialogTitle>
              <DialogDescription>
                {selectedDocument && formatFileSize(selectedDocument.size)} • Uploaded{" "}
                {selectedDocument && formatRelativeDate(selectedDocument.uploadedAt)}
              </DialogDescription>
            </DialogHeader>
            <div className="mt-4 h-[75vh] w-full overflow-auto">
              {selectedDocument?.type === "md" ? (
                <div className="bg-muted/30 rounded-xl p-8">
                  <MarkdownPreview content={selectedDocument.content} />
                </div>
              ) : (
                <pre className="text-foreground bg-muted/30 rounded-xl p-6 font-mono text-sm whitespace-pre-wrap break-words">
                  {selectedDocument?.content}
                </pre>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={closePreview} className="bg-transparent">
                Close
              </Button>
              {selectedDocument && (
                <Button onClick={() => downloadDocument(selectedDocument)}>
                  <Download className="mr-2 h-4 w-4" />
                  Download
                </Button>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Document Dialog */}
        <Dialog open={deleteDialogOpen} onOpenChange={cancelDelete}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete document</DialogTitle>
              <DialogDescription>
                Are you sure you want to delete &quot;{documentToDelete?.name}&quot;? This action
                cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={cancelDelete} className="bg-transparent">
                Cancel
              </Button>
              <Button variant="destructive" onClick={deleteDocument}>
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Rename Folder Dialog */}
        <Dialog open={renameFolderDialogOpen} onOpenChange={cancelFolderAction}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename folder</DialogTitle>
              <DialogDescription>Enter a new name for the folder.</DialogDescription>
            </DialogHeader>
            <Input
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="Folder name"
              onKeyDown={(e) => {
                if (e.key === "Enter") confirmRenameFolder()
              }}
            />
            <DialogFooter>
              <Button variant="outline" onClick={cancelFolderAction} className="bg-transparent">
                Cancel
              </Button>
              <Button onClick={confirmRenameFolder} disabled={!newFolderName.trim()}>
                Rename
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Folder Dialog */}
        <Dialog open={deleteFolderDialogOpen} onOpenChange={cancelFolderAction}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete folder</DialogTitle>
              <DialogDescription>
                Are you sure you want to delete &quot;{folderToEdit?.name}&quot;? This will also
                delete all subfolders and documents inside. This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={cancelFolderAction} className="bg-transparent">
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmDeleteFolder}>
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Rename Document Dialog */}
        <Dialog open={renameDocumentDialogOpen} onOpenChange={cancelRenameDocument}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename document</DialogTitle>
              <DialogDescription>Enter a new name for the document.</DialogDescription>
            </DialogHeader>
            <Input
              value={newDocumentName}
              onChange={(e) => setNewDocumentName(e.target.value)}
              placeholder="Document name"
              onKeyDown={(e) => {
                if (e.key === "Enter") confirmRenameDocument()
              }}
            />
            <DialogFooter>
              <Button variant="outline" onClick={cancelRenameDocument} className="bg-transparent">
                Cancel
              </Button>
              <Button onClick={confirmRenameDocument} disabled={!newDocumentName.trim()}>
                Rename
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Suspense>
  )
}
