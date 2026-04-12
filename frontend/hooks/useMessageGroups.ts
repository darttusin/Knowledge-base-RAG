import type { Conversation, Message } from "@/lib/types"

export interface MessageGroup {
  userMessage: Message
  responses: Message[]
}

export function useMessageGroups(conversation: Conversation | null): MessageGroup[] {
  if (!conversation || !conversation.messages) return []

  const groups: MessageGroup[] = conversation.messages
    .filter((msg): msg is Message & { role: "user" } => msg.role === "user")
    .map((userMessage) => ({ userMessage, responses: [] }))

  if (groups.length === 0) {
    return []
  }

  const groupByUserId = new Map(groups.map((group) => [group.userMessage.id, group]))

  conversation.messages.forEach((msg, index) => {
    if (msg.role !== "assistant") return

    if (msg.parentMessageId) {
      const parentGroup = groupByUserId.get(msg.parentMessageId)
      if (parentGroup) {
        parentGroup.responses.push(msg)
      }
      return
    }

    // Backward compatibility: if message has no parentMessageId, attach it to the
    // nearest preceding user message.
    for (let i = index - 1; i >= 0; i--) {
      const prev = conversation.messages?.[i]
      if (prev?.role === "user") {
        const fallbackGroup = groupByUserId.get(prev.id)
        if (fallbackGroup) {
          fallbackGroup.responses.push(msg)
        }
        break
      }
    }
  })

  return groups
}
