"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { toast } from "sonner"
import { MAX_FILE_SIZE, ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES } from "@/lib/constants"
import {
  getDocuments,
  getDocumentContent,
  uploadDocument as apiUploadDocument,
  deleteDocument as apiDeleteDocument,
  type SearchIn,
} from "@/lib/api"

// Debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])

  return debouncedValue
}

export interface Document {
  id: string
  name: string
  type: "md" | "txt"
  size: number
  content: string
  uploadedAt: Date
  folderId: string | null
  folderPath?: string
}

function validateFile(file: File): { valid: boolean; error?: string } {
  // Size check
  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, error: `File "${file.name}" exceeds 5MB limit` }
  }

  // Extension check
  const extension = file.name.split(".").pop()?.toLowerCase()
  if (
    !extension ||
    !ALLOWED_EXTENSIONS.includes(extension as (typeof ALLOWED_EXTENSIONS)[number])
  ) {
    return { valid: false, error: "Only .md and .txt files are supported" }
  }

  // MIME type check (with fallback for .md files that may have empty type)
  if (file.type && !ALLOWED_MIME_TYPES.includes(file.type) && file.type !== "") {
    return { valid: false, error: `Invalid file type: ${file.type}` }
  }

  return { valid: true }
}

function sanitizeFilename(filename: string): string {
  return (
    filename
      .replace(/\.\.[\/\\]/g, "") // path traversal
      .replace(/[\/\\]/g, "") // slashes
      .replace(/[\x00-\x1f\x7f]/g, "") // control chars
      .slice(0, 255) || "download"
  )
}

export function useDocuments(currentFolderId: string | null = null) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [totalSizeBytes, setTotalSizeBytes] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Search state
  const [searchQuery, setSearchQuery] = useState("")
  const [searchIn, setSearchIn] = useState<SearchIn>("both")
  const debouncedSearchQuery = useDebounce(searchQuery, 500)

  const ITEMS_PER_PAGE = 100

  // Load documents from API
  const loadDocuments = useCallback(
    async (page: number, append: boolean = false, query?: string, search_in?: SearchIn) => {
      if (append) {
        setIsLoadingMore(true)
      } else {
        setIsLoading(true)
      }

      const result = await getDocuments({
        page,
        limit: ITEMS_PER_PAGE,
        folderId: currentFolderId,
        query: query || undefined,
        searchIn: search_in || searchIn,
      })

      if (result.success) {
        const docs: Document[] = result.data.items.map((item) => ({
          id: item.id,
          name: item.name,
          type: item.fileType,
          size: item.size,
          content: "", // Content not loaded until preview
          uploadedAt: new Date(item.uploadedAt),
          folderId: item.folderId,
          folderPath: item.folderPath,
        }))

        if (append) {
          setDocuments((prev) => [...prev, ...docs])
        } else {
          setDocuments(docs)
        }

        setTotalCount(result.data.total)
        setTotalSizeBytes(result.data.totalSize || 0)
        setHasMore(docs.length === ITEMS_PER_PAGE)
        console.log(
          "Loaded documents - Page:",
          page,
          "Total:",
          result.data.total,
          "TotalSize:",
          result.data.totalSize,
          "Loaded:",
          docs.length
        )
      } else {
        toast.error(`Failed to load documents: ${result.error}`)
      }

      setIsLoading(false)
      setIsLoadingMore(false)
    },
    [currentFolderId, searchIn]
  )

  // Load more documents (infinite scroll)
  const loadMore = useCallback(() => {
    if (!isLoadingMore && hasMore) {
      const nextPage = currentPage + 1
      setCurrentPage(nextPage)
      loadDocuments(nextPage, true)
    }
  }, [currentPage, hasMore, isLoadingMore, loadDocuments])

  // Reload documents when folder or search changes
  useEffect(() => {
    setCurrentPage(1)
    setDocuments([])
    setHasMore(true)

    // Load first page
    const load = async () => {
      setIsLoading(true)

      const result = await getDocuments({
        page: 1,
        limit: ITEMS_PER_PAGE,
        folderId: currentFolderId,
        query: debouncedSearchQuery || undefined,
        searchIn: searchIn,
      })

      if (result.success) {
        const docs: Document[] = result.data.items.map((item) => ({
          id: item.id,
          name: item.name,
          type: item.fileType,
          size: item.size,
          content: "",
          uploadedAt: new Date(item.uploadedAt),
          folderId: item.folderId,
          folderPath: item.folderPath,
        }))

        setDocuments(docs)
        setTotalCount(result.data.total)
        setTotalSizeBytes(result.data.totalSize || 0)
        setHasMore(docs.length === ITEMS_PER_PAGE)
      } else {
        toast.error(`Failed to load documents: ${result.error}`)
      }

      setIsLoading(false)
    }

    load()
  }, [currentFolderId, debouncedSearchQuery, searchIn])

  // No need to filter - documents are already filtered by backend
  const filteredDocuments = documents

  const uploadFiles = useCallback(
    async (files: FileList | null, folderId: string | null = currentFolderId) => {
      if (!files) return

      const uploadPromises = Array.from(files).map(async (file) => {
        const validation = validateFile(file)
        if (!validation.valid) {
          toast.error(validation.error)
          return null
        }

        try {
          const result = await apiUploadDocument(file, (progress) => {
            // You could add a progress toast here if needed
          })

          if (result.success) {
            toast.success(`Uploaded: ${file.name}`)

            const newDoc: Document = {
              id: result.data.id,
              name: result.data.name,
              type: result.data.fileType,
              size: result.data.size,
              content: "",
              uploadedAt: new Date(result.data.uploadedAt),
              folderId: result.data.folderId,
              folderPath: result.data.folderPath,
            }

            return newDoc
          } else {
            toast.error(`Failed to upload ${file.name}: ${result.error}`)
            return null
          }
        } catch (error) {
          toast.error(error instanceof Error ? error.message : "Failed to upload file")
          return null
        }
      })

      const results = await Promise.all(uploadPromises)
      const validDocs = results.filter((doc): doc is Document => doc !== null)

      if (validDocs.length > 0) {
        setDocuments((prev) => [...validDocs, ...prev])
      }
    },
    [currentFolderId]
  )

  const deleteDocument = useCallback(async () => {
    if (documentToDelete) {
      const result = await apiDeleteDocument(documentToDelete.id)

      if (result.success) {
        setDocuments((prev) => prev.filter((doc) => doc.id !== documentToDelete.id))
        toast.success("Document deleted")
      } else {
        toast.error(`Failed to delete document: ${result.error}`)
      }

      setDocumentToDelete(null)
      setDeleteDialogOpen(false)
    }
  }, [documentToDelete])

  const renameDocument = useCallback((documentId: string, newName: string) => {
    // TODO: Implement API call for renaming
    setDocuments((prev) =>
      prev.map((doc) =>
        doc.id === documentId ? { ...doc, name: sanitizeFilename(newName) } : doc
      )
    )
    toast.success("Document renamed")
  }, [])

  const openPreview = useCallback(async (doc: Document) => {
    // Load content if not already loaded
    if (!doc.content) {
      const result = await getDocumentContent(doc.id)
      if (result.success) {
        doc.content = result.data
      } else {
        toast.error(`Failed to load document content: ${result.error}`)
        return
      }
    }
    setSelectedDocument(doc)
    setPreviewOpen(true)
  }, [])

  const closePreview = useCallback(() => {
    setPreviewOpen(false)
  }, [])

  const confirmDelete = useCallback((doc: Document) => {
    setDocumentToDelete(doc)
    setDeleteDialogOpen(true)
  }, [])

  const cancelDelete = useCallback(() => {
    setDeleteDialogOpen(false)
    setDocumentToDelete(null)
  }, [])

  const downloadDocument = useCallback((doc: Document) => {
    // TODO: Use API endpoint for downloading
    const blob = new Blob([doc.content], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = sanitizeFilename(doc.name)
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      uploadFiles(e.dataTransfer.files)
    },
    [uploadFiles]
  )

  const moveDocument = useCallback((documentId: string, targetFolderId: string | null) => {
    // TODO: Implement API call for moving
    setDocuments((prev) =>
      prev.map((doc) =>
        doc.id === documentId
          ? { ...doc, folderId: targetFolderId || "root", folderPath: "/" }
          : doc
      )
    )
  }, [])

  const findDocumentById = useCallback(
    (documentId: string) => {
      return documents.find((doc) => doc.id === documentId) || null
    },
    [documents]
  )

  return {
    // State
    documents: filteredDocuments,
    allDocuments: documents,
    selectedDocument,
    previewOpen,
    deleteDialogOpen,
    documentToDelete,
    isDragging,
    fileInputRef,
    totalSize: totalSizeBytes,
    isLoading,
    isLoadingMore,

    // Search
    searchQuery,
    setSearchQuery,
    searchIn,
    setSearchIn,

    // Infinite scroll
    totalCount,
    hasMore,
    loadMore,

    // Actions
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
    moveDocument,
    findDocumentById,
  }
}
