// Authentication mutations

import { api, safeRequest } from "../client"
import type { ApiResult } from "@/types/api"
import { saveAuthData, clearAuthData, type AuthData } from "@/lib/auth/token"

const ENDPOINTS = {
  login: "/api/user/auth",
  me: "/api/user/me",
}

export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  user: {
    user_id: number
    email: string
    username: string
  }
}

/**
 * Login user and save auth data
 */
export async function login(credentials: LoginRequest): Promise<ApiResult<AuthData>> {
  const result = await safeRequest(() =>
    api.post<LoginResponse>(ENDPOINTS.login, credentials)
  )

  if (result.success) {
    // Save token and user data to localStorage
    saveAuthData(result.data)
  }

  return result
}

/**
 * Logout user and clear auth data
 */
export function logout(): void {
  clearAuthData()
}

/**
 * Get current user info
 */
export async function getCurrentUser(): Promise<ApiResult<AuthData["user"]>> {
  return safeRequest(() => api.get<AuthData["user"]>(ENDPOINTS.me))
}
