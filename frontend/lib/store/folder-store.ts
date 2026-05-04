import { create } from "zustand"
import { devtools } from "zustand/middleware"
import {
  createFolder as createFolderApi,
  getFolders,
  moveFolderToParent as moveFolderApi,
  type BackendFolder,
} from "@/lib/api"

export interface Folder {
  id: string
  name: string
  path: string
  parentId: string | null
  createdAt: Date
  documentCount: number
}

interface FolderState {
  folders: Folder[]
  currentFolderId: string | null
  expandedFolders: string[]
  isLoading: boolean
}

interface FolderActions {
  loadFolders: () => Promise<void>
  createFolder: (name: string, parentId?: string | null) => Promise<void>
  updateFolder: (id: string, name: string) => void
  deleteFolder: (id: string) => void
  moveFolder: (id: string, targetParentId: string | null) => Promise<void>
  setCurrentFolder: (id: string | null) => void
  toggleFolderExpanded: (id: string) => void
}

type FolderStore = FolderState & FolderActions

/**
 * Convert backend folder to frontend folder
 */
function adaptBackendFolder(backendFolder: BackendFolder): Folder {
  return {
    id: backendFolder.id.toString(),
    name: backendFolder.name,
    path: backendFolder.path,
    parentId: backendFolder.parent_id?.toString() || null,
    createdAt: new Date(backendFolder.created_at),
    documentCount: backendFolder.document_count,
  }
}

function buildPath(folders: Folder[], folderId: string): string {
  const folder = folders.find((f) => f.id === folderId)
  if (!folder || !folder.parentId) return "/"

  const parent = folders.find((f) => f.id === folder.parentId)
  if (!parent || parent.id === "root") return `/${folder.name}`

  return `${buildPath(folders, parent.id)}/${folder.name}`
}

export const useFolderStore = create<FolderStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      folders: [],
      currentFolderId: null, // null = показывать все документы
      expandedFolders: [],
      isLoading: false,

      // Actions
      loadFolders: async () => {
        set({ isLoading: true })
        const result = await getFolders()

        if (result.success) {
          const folders = result.data.map(adaptBackendFolder)
          set({ folders, isLoading: false })
        } else {
          console.error("Failed to load folders:", result.error)
          set({ isLoading: false })
        }
      },

      createFolder: async (name, parentId = null) => {
        const folders = get().folders
        const isTopLevel = !parentId || parentId === "root"
        const path = isTopLevel ? `/${name}` : `${buildPath(folders, parentId!)}/${name}`
        const parentIdInt = isTopLevel ? null : Number.parseInt(parentId!, 10)

        const result = await createFolderApi({ name, path, parent_id: parentIdInt })
        if (!result.success) {
          throw new Error(result.error)
        }

        const newFolder = adaptBackendFolder(result.data)
        set((state) => ({
          folders: [...state.folders, newFolder],
        }))
      },

      updateFolder: (id, name) => {
        set((state) => {
          const folders = state.folders.map((f) => {
            if (f.id === id) {
              const parentPath =
                f.parentId && f.parentId !== "root" ? buildPath(state.folders, f.parentId) : ""
              return {
                ...f,
                name,
                path: parentPath ? `${parentPath}/${name}` : `/${name}`,
              }
            }
            return f
          })

          // Update paths of all children
          const updatedFolders = folders.map((f) => {
            if (f.parentId === id) {
              return {
                ...f,
                path: buildPath(folders, f.id),
              }
            }
            return f
          })

          return { folders: updatedFolders }
        })
      },

      deleteFolder: (id) => {
        set((state) => {
          // Recursively get all folder IDs to delete (folder + all descendants)
          const getDescendantIds = (folderId: string): string[] => {
            const children = state.folders.filter((f) => f.parentId === folderId)
            return [
              folderId,
              ...children.flatMap((child) => getDescendantIds(child.id)),
            ]
          }

          const idsToDelete = getDescendantIds(id)
          return {
            folders: state.folders.filter((f) => !idsToDelete.includes(f.id)),
            currentFolderId:
              state.currentFolderId && idsToDelete.includes(state.currentFolderId)
                ? "root"
                : state.currentFolderId,
          }
        })
      },

      moveFolder: async (id, targetParentId) => {
        const folderIdInt = Number.parseInt(id, 10)
        const isTopLevel = !targetParentId || targetParentId === "root"
        const parentIdInt = isTopLevel ? null : Number.parseInt(targetParentId!, 10)

        const result = await moveFolderApi(folderIdInt, parentIdInt)
        if (!result.success) {
          throw new Error(result.error)
        }

        // Re-fetch from backend so paths of moved folder + descendants are authoritative
        const refreshed = await getFolders()
        if (refreshed.success) {
          set({ folders: refreshed.data.map(adaptBackendFolder) })
        }
      },

      setCurrentFolder: (id) => {
        set({ currentFolderId: id })
      },

      toggleFolderExpanded: (id) => {
        set((state) => {
          const isExpanded = state.expandedFolders.includes(id)
          return {
            expandedFolders: isExpanded
              ? state.expandedFolders.filter((folderId) => folderId !== id)
              : [...state.expandedFolders, id],
          }
        })
      },
    }),
    { name: "folder-store" }
  )
)

// Selectors
export const selectFolders = (state: FolderStore) => state.folders
export const selectCurrentFolderId = (state: FolderStore) => state.currentFolderId
export const selectExpandedFolders = (state: FolderStore) => state.expandedFolders
export const selectCurrentFolder = (state: FolderStore) =>
  state.folders.find((f) => f.id === state.currentFolderId) || null

// Helper to get children of a folder
export const selectFolderChildren = (folderId: string) => (state: FolderStore) =>
  state.folders.filter((f) => f.parentId === folderId)

// Helper to get folder by id
export const selectFolderById = (id: string) => (state: FolderStore) =>
  state.folders.find((f) => f.id === id) || null
