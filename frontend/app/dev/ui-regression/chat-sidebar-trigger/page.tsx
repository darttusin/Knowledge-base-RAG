import { MoreHorizontal } from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const triggerBaseClass =
  "hover:bg-background focus-visible:ring-ring data-[state=open]:bg-background h-6 w-6 transition-opacity opacity-100 md:opacity-0 md:group-hover:opacity-100 md:focus-visible:opacity-100 md:data-[state=open]:opacity-100"

function TriggerPreview({
  label,
  forceVisible,
  active,
  open,
}: {
  label: string
  forceVisible?: boolean
  active?: boolean
  open?: boolean
}) {
  return (
    <div className="bg-background flex items-center justify-between rounded-lg border p-3">
      <span className={cn("text-sm", active ? "text-foreground" : "text-muted-foreground")}>
        {label}
      </span>
      <div className={cn("group", forceVisible && "hover")}>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Conversation actions"
          data-state={open ? "open" : "closed"}
          className={cn(
            triggerBaseClass,
            forceVisible && "opacity-100 md:opacity-100",
            active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
          )}
        >
          <MoreHorizontal className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

export default function ChatSidebarTriggerRegressionPage() {
  return (
    <main className="space-y-8 p-6">
      <h1 className="text-xl font-semibold">Chat sidebar trigger UI regression</h1>

      <section className="space-y-3">
        <h2 className="text-base font-medium">Light theme</h2>
        <TriggerPreview label="Normal / mobile-visible" />
        <TriggerPreview label="Hover (desktop simulation)" forceVisible />
        <TriggerPreview label="Active conversation" active />
        <TriggerPreview label="Menu open state" open forceVisible />
      </section>

      <section className="dark space-y-3 rounded-xl border p-4">
        <h2 className="text-foreground text-base font-medium">Dark theme</h2>
        <TriggerPreview label="Normal / mobile-visible" />
        <TriggerPreview label="Hover (desktop simulation)" forceVisible />
        <TriggerPreview label="Active conversation" active />
        <TriggerPreview label="Menu open state" open forceVisible />
      </section>
    </main>
  )
}
