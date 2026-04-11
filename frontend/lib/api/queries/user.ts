import { api, safeRequest, type ApiResult } from "@/lib/api/client"

export interface User {
  user_id: number
  email: string
  username: string
}

/**
 * Get current user data
 */
export async function getCurrentUser(): Promise<ApiResult<User>> {
  return safeRequest(() => api.get<User>("/api/user/me"))
}
