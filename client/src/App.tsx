import {
  AIBranch,
  AIBranchMessages,
  AIBranchNext,
  AIBranchPage,
  AIBranchPrevious,
  AIBranchSelector,
} from "@/components/ui/kibo-ui/ai/branch";
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
import {
  AIReasoning,
  AIReasoningContent,
  AIReasoningTrigger,
} from "@/components/ui/kibo-ui/ai/reasoning";
import { AIResponse } from "@/components/ui/kibo-ui/ai/response";
import {
  AISource,
  AISources,
  AISourcesContent,
  AISourcesTrigger,
} from "@/components/ui/kibo-ui/ai/source";
import {
  AITool,
  AIToolContent,
  AIToolHeader,
  AIToolParameters,
  AIToolResult,
  type AIToolStatus,
} from "@/components/ui/kibo-ui/ai/tool";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import { useState } from "react";
import { toast } from "sonner";

// Legacy message type for UI compatibility
type LegacyMessage = {
  from: "user" | "assistant";
  sources?: { href: string; title: string }[];
  versions: {
    id: string;
    content: string;
  }[];
  reasoning?: {
    content: string;
    duration: number;
  };
  tools?: {
    name: string;
    description: string;
    status: AIToolStatus;
    parameters: Record<string, unknown>;
    result: string | undefined;
    error: string | undefined;
  }[];
  avatar: string;
  name: string;
};

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

  console.log("messages", messages);

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

  // Convert AI SDK messages to legacy format for UI compatibility
  const legacyMessages: LegacyMessage[] = messages.map((message) => ({
    from: message.role as "user" | "assistant",
    versions: [
      {
        id: message.id,
        content: message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join(""),
      },
    ],
    avatar:
      message.role === "user"
        ? "https://github.com/haydenbleasel.png"
        : "https://github.com/openai.png",
    name: message.role === "user" ? "User" : "Assistant",
  }));

  return (
    <div className="flex min-h-svh flex-col">
      <div className="max-w-screen-lg mx-auto w-full px-4 flex flex-col flex-1">
        <div className="flex-1 overflow-hidden">
          <AIConversation>
            <AIConversationContent>
              {legacyMessages.map(({ versions, ...message }, index) => (
                <AIBranch defaultBranch={0} key={index}>
                  <AIBranchMessages>
                    {versions.map((version) => (
                      <AIMessage from={message.from} key={version.id}>
                        <div>
                          {message.sources?.length && (
                            <AISources>
                              <AISourcesTrigger
                                count={message.sources.length}
                              />
                              <AISourcesContent>
                                {message.sources.map((source) => (
                                  <AISource
                                    href={source.href}
                                    key={source.href}
                                    title={source.title}
                                  />
                                ))}
                              </AISourcesContent>
                            </AISources>
                          )}
                          {message.tools?.map((toolCall) => (
                            <AITool key={toolCall.name}>
                              <AIToolHeader
                                description={toolCall.description}
                                name={`Called MCP tool: ${toolCall.name}`}
                                status={toolCall.status}
                              />
                              <AIToolContent>
                                <AIToolParameters
                                  parameters={toolCall.parameters}
                                />
                                {(toolCall.result || toolCall.error) && (
                                  <AIToolResult
                                    error={toolCall.error}
                                    result={toolCall.result}
                                  />
                                )}
                              </AIToolContent>
                            </AITool>
                          ))}
                          {message.reasoning && (
                            <AIReasoning duration={message.reasoning.duration}>
                              <AIReasoningTrigger />
                              <AIReasoningContent>
                                {message.reasoning.content}
                              </AIReasoningContent>
                            </AIReasoning>
                          )}
                          <AIMessageContent>
                            <AIResponse>{version.content}</AIResponse>
                          </AIMessageContent>
                        </div>
                        <AIMessageAvatar
                          name={message.name}
                          src={message.avatar}
                        />
                      </AIMessage>
                    ))}
                  </AIBranchMessages>
                  {versions.length > 1 && (
                    <AIBranchSelector from={message.from}>
                      <AIBranchPrevious />
                      <AIBranchPage />
                      <AIBranchNext />
                    </AIBranchSelector>
                  )}
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
                <AIInputTools>
                  {/* Removed attachment controls for simplicity */}
                </AIInputTools>
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
