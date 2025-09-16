import gentLogo from "@/assets/gent-disagreement-logo.png";
import { AIBranch, AIBranchMessages } from "@/components/ui/kibo-ui/ai/branch";
import {
  AIConversation,
  AIConversationContent,
  AIConversationScrollButton,
} from "@/components/ui/kibo-ui/ai/conversation";
import {
  AIInput,
  AIInputSubmit,
  AIInputTextarea,
  AIInputToolbar,
  AIInputTools,
} from "@/components/ui/kibo-ui/ai/input";
import {
  AIMessage,
  AIMessageAvatar,
  AIMessageContent,
} from "@/components/ui/kibo-ui/ai/message";
import { AIResponse } from "@/components/ui/kibo-ui/ai/response";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { User } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

function App() {
  const [input, setInput] = useState<string>("");

  // Use the Vercel AI SDK's useChat hook with text streaming - replaces all manual streaming logic
  const { messages, sendMessage, status } = useChat({
    transport: new TextStreamChatTransport({
      api: "http://localhost:8000/api/chat",
    }),
    onFinish: () => {
      toast.success("Response received");
    },
    onError: (error) => {
      toast.error("Error", {
        description: error.message,
      });
    },
  });

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!input.trim()) {
      return;
    }

    sendMessage({
      text: input,
    });

    setInput("");
    toast.success("Message sent");
  };

  return (
    <div className="flex min-h-svh flex-col">
      <div className="max-w-screen-lg mx-auto w-full flex flex-col flex-1">
        <div className="flex-1 overflow-hidden">
          <AIConversation>
            <AIConversationContent>
              {messages.map((message) => (
                <AIBranch defaultBranch={0} key={message.id}>
                  <AIBranchMessages>
                    <AIMessage
                      from={message.role as "user" | "assistant"}
                      key={message.id}
                    >
                      <div>
                        <AIMessageContent>
                          <AIResponse>
                            {message.parts
                              ?.filter((part) => part.type === "text")
                              .map(
                                (part: { type: string; text: string }) =>
                                  part.text
                              )
                              .join("") || ""}
                          </AIResponse>
                        </AIMessageContent>
                      </div>
                      <div className="hidden sm:block">
                        {message.role === "user" ? (
                          <div className="size-8 rounded-full bg-black flex items-center justify-center">
                            <User
                              strokeWidth={3}
                              className="size-4 text-white"
                            />
                          </div>
                        ) : (
                          <AIMessageAvatar name="Assistant" src={gentLogo} />
                        )}
                      </div>
                    </AIMessage>
                  </AIBranchMessages>
                </AIBranch>
              ))}
            </AIConversationContent>
            <AIConversationScrollButton />
          </AIConversation>
        </div>
        <div className="shrink-0 pt-4">
          <div className="w-full px-4 pb-4">
            <AIInput onSubmit={handleSubmit}>
              <AIInputTextarea
                onChange={(event) => setInput(event.target.value)}
                value={input}
              />
              <AIInputToolbar>
                <AIInputTools></AIInputTools>
                <AIInputSubmit disabled={!input.trim()} status={status} />
              </AIInputToolbar>
            </AIInput>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
