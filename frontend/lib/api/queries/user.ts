export interface User {
  user_id: number
  email: string
  username: string
}

// getCurrentUser moved to mutations/auth.ts to avoid duplicate exports
