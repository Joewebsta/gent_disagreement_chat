import { type PromptInputMessage } from "@/components/ai-elements/prompt-input";
import { ChatConversation } from "@/components/ChatConversation";
import { ChatInput } from "@/components/ChatInput";
import { InfoDialog } from "@/components/InfoDialog";
import { getMessageText } from "@/lib/message-utils";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useState } from "react";

function App() {
  const [input, setInput] = useState<string>("");

  const suggestions = [
    "What is a Gentleman's Disagreement?",
    "Summarize the most recent episodes",
  ];

  const { messages, sendMessage, stop, status } = useChat({
    transport: new TextStreamChatTransport({
      api: import.meta.env.VITE_API_URL || "http://localhost:8000/api/chat",
    }),
    onFinish: () => {},
    onError: () => {},
  });

  const handleSubmit = (message: PromptInputMessage) => {
    const hasText = Boolean(message.text);
    const hasAttachments = Boolean(message.files?.length);

    if (!(hasText || hasAttachments)) {
      return;
    }

    sendMessage({
      text: message.text || "Sent with attachments",
    });
    setInput("");
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (status === "streaming") {
      return;
    }

    sendMessage({ text: suggestion });
    setInput("");
  };

  const regenerate = () => {
    if (messages.length > 0) {
      const lastUserMessage = messages
        .slice()
        .reverse()
        .find((m) => m.role === "user");
      if (lastUserMessage) {
        sendMessage({ text: getMessageText(lastUserMessage) });
      }
    }
  };

  return (
    <>
      {messages.length === 0 && <InfoDialog />}

      <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
        <div className="flex flex-col h-full">
          <ChatConversation
            messages={messages}
            onRegenerate={regenerate}
            status={status}
          />
          <ChatInput
            input={input}
            status={status}
            suggestions={suggestions}
            onInputChange={setInput}
            onSubmit={handleSubmit}
            onSuggestionClick={handleSuggestionClick}
            onStop={stop}
          />
        </div>
      </div>
    </>
  );
}

export default App;
