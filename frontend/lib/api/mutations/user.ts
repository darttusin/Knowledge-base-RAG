import { api, safeRequest, type ApiResult } from "@/lib/api/client"

export interface UserChanges {
  email?: string
  username?: string
  old_password?: string
  new_password?: string
}

/**
 * Update user profile (email, username, or password)
 */
export async function updateUser(changes: UserChanges): Promise<ApiResult<void>> {
  return safeRequest(() => api.put<void>("/api/user", changes))
}

/**
 * Delete user account
 */
export async function deleteUser(): Promise<ApiResult<void>> {
  return safeRequest(() => api.delete<void>("/api/user"))
}
