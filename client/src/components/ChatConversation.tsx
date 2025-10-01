import gdLogo from "@/assets/gd-logo.png";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { ChatMessage } from "@/components/ChatMessage";
import type { ChatStatus, UIMessage } from "ai";

interface ChatConversationProps {
  messages: UIMessage[];
  onRegenerate: () => void;
  status: ChatStatus;
}

export function ChatConversation({
  messages,
  onRegenerate,
  status,
}: ChatConversationProps) {
  return (
    <Conversation className="flex-1">
      <ConversationContent className="h-full">
        {messages.length === 0 ? (
          <ConversationEmptyState
            title="Welcome to A Gentleman's Disagreement"
            description="Start a conversation to explore the podcast"
            icon={
              <img
                src={gdLogo}
                alt="Gentleman's Disagreement Logo"
                className="w-52 h-52 rounded-2xl object-contain"
              />
            }
          />
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id}>
                <ChatMessage
                  message={message}
                  isLastMessage={message.id === messages.at(-1)?.id}
                  onRegenerate={onRegenerate}
                />
              </div>
            ))}
            {/* Show loader when status indicates request is in progress but no assistant message exists */}
            {status === "submitted" &&
              messages[messages.length - 1]?.role !== "assistant" && (
                <div key="loading-placeholder">
                  <ChatMessage
                    message={{
                      id: "loading-placeholder",
                      role: "assistant",
                      parts: [],
                    }}
                    isLastMessage={true}
                    onRegenerate={onRegenerate}
                  />
                </div>
              )}
          </>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
