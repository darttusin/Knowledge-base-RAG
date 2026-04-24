// JWT token storage and retrieval utilities

const TOKEN_KEY = "auth_token"
const USER_KEY = "auth_user"

export interface AuthUser {
  user_id: number
  email: string
  username: string
}

export interface AuthData {
  access_token: string
  token_type: string
  user: AuthUser
}

/**
 * Save auth data to localStorage
 */
export function saveAuthData(data: AuthData): void {
  if (typeof window === "undefined") return

  localStorage.setItem(TOKEN_KEY, data.access_token)
  localStorage.setItem(USER_KEY, JSON.stringify(data.user))
}

/**
 * Get JWT token from localStorage
 */
export function getToken(): string | null {
  if (typeof window === "undefined") return null

  return localStorage.getItem(TOKEN_KEY)
}

/**
 * Get user data from localStorage
 */
export function getUser(): AuthUser | null {
  if (typeof window === "undefined") return null

  const userJson = localStorage.getItem(USER_KEY)
  if (!userJson) return null

  try {
    return JSON.parse(userJson) as AuthUser
  } catch {
    return null
  }
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return getToken() !== null
}

/**
 * Clear auth data from localStorage
 */
export function clearAuthData(): void {
  if (typeof window === "undefined") return

  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

/**
 * Get Authorization header value
 */
export function getAuthHeader(): string | null {
  const token = getToken()
  return token ? `Bearer ${token}` : null
}
