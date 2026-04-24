"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Eye, EyeOff, Database, Loader2, Mail, Lock, ArrowRight } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { ThemeToggle } from "@/components/theme-toggle"
import { login } from "@/lib/api"

export default function LoginPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = React.useState(false)
  const [showPassword, setShowPassword] = React.useState(false)
  const [email, setEmail] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [rememberMe, setRememberMe] = React.useState(false)
  const [error, setError] = React.useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    // Call real login API
    const result = await login({ email, password })

    if (result.success) {
      toast.success("Login successful!")
      router.push("/")
    } else {
      const isInvalidCredentialsError =
        result.code === "HTTP_401" ||
        result.code === "HTTP_404" ||
        result.code === "INVALID_CREDENTIALS"

      const errorMessage = isInvalidCredentialsError
        ? "Invalid email or password"
        : result.error || "Failed to sign in"

      setError(errorMessage)
      toast.error(errorMessage)
    }

    setIsLoading(false)
  }

  return (
    <div className="flex min-h-screen">
      {/* Left side - Branding */}
      <div className="bg-primary relative hidden overflow-hidden lg:flex lg:w-1/2">
        <div className="from-primary via-primary to-primary/80 absolute inset-0 bg-gradient-to-br" />

        {/* Decorative elements */}
        <div className="bg-primary-foreground/10 absolute top-20 left-20 h-72 w-72 rounded-full blur-3xl" />
        <div className="bg-primary-foreground/5 absolute right-20 bottom-20 h-96 w-96 rounded-full blur-3xl" />

        <div className="text-primary-foreground relative z-10 flex flex-col justify-between p-12">
          <div className="flex items-center gap-3">
            <div className="bg-primary-foreground/20 flex h-10 w-10 items-center justify-center rounded-xl">
              <Database className="h-5 w-5" />
            </div>
            <span className="text-xl font-semibold">DataMind AI</span>
          </div>

          <div className="max-w-md">
            <h1 className="mb-6 text-4xl leading-tight font-bold text-balance">
              Intelligent insights from your private data
            </h1>
            <p className="text-primary-foreground/80 text-lg leading-relaxed">
              Connect your databases, documents, and knowledge bases. Ask questions in natural
              language and get accurate answers with source citations.
            </p>

            <div className="mt-12 grid grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold">99.9%</div>
                <div className="text-primary-foreground/70 mt-1 text-sm">Uptime</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold">500K+</div>
                <div className="text-primary-foreground/70 mt-1 text-sm">Queries/day</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold">50ms</div>
                <div className="text-primary-foreground/70 mt-1 text-sm">Avg response</div>
              </div>
            </div>
          </div>

          <div className="text-primary-foreground/60 flex items-center gap-2 text-sm">
            <Lock className="h-4 w-4" />
            <span>Enterprise-grade security with SOC 2 Type II compliance</span>
          </div>
        </div>
      </div>

      {/* Right side - Login form */}
      <div className="bg-background flex w-full flex-col lg:w-1/2">
        <div className="flex items-center justify-between p-6">
          <div className="flex items-center gap-3 lg:hidden">
            <div className="bg-primary flex h-9 w-9 items-center justify-center rounded-xl">
              <Database className="text-primary-foreground h-4 w-4" />
            </div>
            <span className="text-lg font-semibold">DataMind AI</span>
          </div>
          <div className="lg:ml-auto">
            <ThemeToggle />
          </div>
        </div>

        <div className="flex flex-1 items-center justify-center p-6 sm:p-12">
          <div className="w-full max-w-md space-y-8">
            <div className="text-center lg:text-left">
              <h2 className="text-2xl font-bold tracking-tight">Welcome back</h2>
              <p className="text-muted-foreground mt-2">
                Sign in to access your AI-powered database assistant
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="name@company.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="h-11 pl-10"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="text-muted-foreground absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="h-11 pr-10 pl-10"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="text-muted-foreground hover:text-foreground absolute top-1/2 right-3 -translate-y-1/2 transition-colors"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      <span className="sr-only">
                        {showPassword ? "Hide password" : "Show password"}
                      </span>
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="remember"
                  checked={rememberMe}
                  onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                />
                <Label
                  htmlFor="remember"
                  className="text-muted-foreground cursor-pointer text-sm font-normal"
                >
                  Remember me for 30 days
                </Label>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive rounded-lg px-4 py-3 text-sm">
                  {error}
                </div>
              )}

              <Button type="submit" className="h-11 w-full text-base" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    Sign in
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </form>

          </div>
        </div>

        <div className="text-muted-foreground p-6 text-center text-xs">
          By signing in, you agree to our{" "}
          <Link href="/terms" className="hover:text-foreground underline transition-colors">
            Terms of Service
          </Link>{" "}
          and{" "}
          <Link href="/privacy" className="hover:text-foreground underline transition-colors">
            Privacy Policy
          </Link>
        </div>
      </div>
    </div>
  )
}
