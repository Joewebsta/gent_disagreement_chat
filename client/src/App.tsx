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
import { type FormEventHandler, useState } from "react";
import { toast } from "sonner";

type Message = {
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
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState<string>("");
  const [status, setStatus] = useState<
    "submitted" | "streaming" | "ready" | "error"
  >("ready");

  const handleSubmit: FormEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault();

    if (!text) {
      return;
    }

    const newMessage: Message = {
      from: "user",
      versions: [
        {
          id: Date.now().toString(),
          content: text,
        },
      ],
      avatar: "https://github.com/haydenbleasel.png",
      name: "User",
    };

    setMessages((prev) => [...prev, newMessage]);
    setText("");

    toast.success("Message submitted", {
      description: text,
    });

    setStatus("submitted");
    // setTimeout(() => {
    //   setStatus("streaming");
    // }, 200);

    // Create initial empty assistant message
    const assistantMessage: Message = {
      from: "assistant",
      versions: [
        {
          id: Date.now().toString(),
          content: "",
        },
      ],
      avatar: "https://github.com/openai.png",
      name: "OpenAI",
    };
    setMessages((prev) => [...prev, assistantMessage]);

    // Start streaming
    try {
      const response = await fetch("http://localhost:8000/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newMessage),
      });

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        console.log("Buffer:", JSON.stringify(buffer));

        // Process complete SSE events separated by blank line (\n\n)
        while (true) {
          const eventEnd = buffer.indexOf("\n\n");
          if (eventEnd === -1) break;

          const rawEvent = buffer.slice(0, eventEnd);
          buffer = buffer.slice(eventEnd + 2);

          // Collect all data lines in the event
          const dataLines = rawEvent
            .split("\n")
            .filter((l) => l.startsWith("data: "))
            .map((l) => l.slice(6));
          const data = dataLines.join("\n");

          if (data === "[DONE]") {
            setStatus("ready");
            return;
          }

          if (data) {
            setMessages((prev) =>
              prev.map((msg, index) =>
                index === prev.length - 1
                  ? {
                      ...msg,
                      versions: [
                        {
                          ...msg.versions[0],
                          content: msg.versions[0].content + data,
                        },
                      ],
                    }
                  : msg
              )
            );
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      setStatus("error");
      toast.error("Error", {
        description: "Failed to stream response",
      });
    }
  };

  // Intentionally minimal demo UI – suggestions and attachment controls removed

  return (
    <div className="flex min-h-svh flex-col">
      <div className="max-w-screen-lg mx-auto w-full px-4 flex flex-col flex-1">
        <div className="flex-1 overflow-hidden">
          <AIConversation>
            <AIConversationContent>
              {messages.map(({ versions, ...message }, index) => (
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
          {/* <AISuggestions className="px-4">
            {suggestions.map((suggestion) => (
              <AISuggestion
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                suggestion={suggestion}
              />
            ))}
          </AISuggestions> */}
          <div className="w-full px-4 pb-4">
            <AIInput onSubmit={handleSubmit}>
              <AIInputTextarea
                onChange={(event) => setText(event.target.value)}
                value={text}
              />
              <AIInputToolbar>
                <AIInputTools>
                  {/* <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <AIInputButton>
                        <PlusIcon size={16} />
                        <span className="sr-only">Add attachment</span>
                      </AIInputButton>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      <DropdownMenuItem
                        onClick={() => handleFileAction("upload-file")}
                      >
                        <FileIcon className="mr-2" size={16} />
                        Upload file
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFileAction("upload-photo")}
                      >
                        <ImageIcon className="mr-2" size={16} />
                        Upload photo
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFileAction("take-screenshot")}
                      >
                        <ScreenShareIcon className="mr-2" size={16} />
                        Take screenshot
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFileAction("take-photo")}
                      >
                        <CameraIcon className="mr-2" size={16} />
                        Take photo
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu> */}
                  {/* <AIInputButton
                    onClick={() => setUseMicrophone(!useMicrophone)}
                    variant={useMicrophone ? "default" : "ghost"}
                  >
                    <MicIcon size={16} />
                    <span className="sr-only">Microphone</span>
                  </AIInputButton> */}
                  {/* <AIInputButton
                    onClick={() => setUseWebSearch(!useWebSearch)}
                    variant={useWebSearch ? "default" : "ghost"}
                  >
                    <GlobeIcon size={16} />
                    <span>Search</span>
                  </AIInputButton> */}
                </AIInputTools>
                <AIInputSubmit disabled={!text} status={status} />
              </AIInputToolbar>
            </AIInput>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
