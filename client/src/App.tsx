import gentLogo from "@/assets/assistant-logo.png";
import gdLogo from "@/assets/gd-logo.png";
import { Action, Actions } from "@/components/ai-elements/actions";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Loader } from "@/components/ai-elements/loader";
import { Message, MessageContent } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Response } from "@/components/ai-elements/response";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport, type TextUIPart, type UIMessage } from "ai";
import { CopyIcon, RefreshCcwIcon, User } from "lucide-react";
import { Fragment, useState } from "react";
import { toast } from "sonner";

function isTextPart(part: unknown): part is TextUIPart {
  if (!part || typeof part !== "object" || part === null) {
    return false;
  }

  const candidate = part as Record<string, unknown>;
  return candidate.type === "text" && typeof candidate.text === "string";
}

function getMessageText(message: UIMessage): string {
  return (
    message.parts
      ?.filter(isTextPart)
      .map((part) => part.text)
      .join("") || ""
  );
}

function hasMessageContent(message: UIMessage): boolean {
  return (
    message.parts?.some(
      (part) => isTextPart(part) && part.text && part.text.trim().length > 0
    ) || false
  );
}

function App() {
  const [input, setInput] = useState<string>("");

  const { messages, sendMessage, stop, status } = useChat({
    transport: new TextStreamChatTransport({
      api: import.meta.env.VITE_API_URL || "http://localhost:8000/api/chat",
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
    toast.success("Message sent");
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
    <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
      <div className="flex flex-col h-full">
        {/* <header className="mb-2 text-center">
          <h1 className="text-sm font-semibold text-gray-900">
            A Gentleman's Disagreement Chatbot
          </h1>
        </header> */}
        <Conversation className="flex-1">
          <ConversationContent className="h-full">
            {messages.length === 0 ? (
              <ConversationEmptyState
                title="Welcome to A Gentleman's Disagreement"
                description="Start a conversation to begin exploring ideas and perspectives"
                icon={
                  <img
                    src={gdLogo}
                    alt="Gentleman's Disagreement Logo"
                    className="w-52 h-52 rounded-2xl object-contain"
                  />
                }
              />
            ) : (
              messages.map((message) => (
                <div key={message.id}>
                  {message.role === "assistant" &&
                  !hasMessageContent(message) ? (
                    <Message from="assistant" key={`${message.id}-loading`}>
                      <MessageContent>
                        <Loader className="my-[2px]" />
                      </MessageContent>
                      <div className="hidden sm:block">
                        <Avatar className={cn("size-8")}>
                          <AvatarImage
                            alt=""
                            className="mt-0 mb-0"
                            src={gentLogo}
                          />
                        </Avatar>
                      </div>
                    </Message>
                  ) : (
                    message.parts.map((part, i) => {
                      switch (part.type) {
                        case "text":
                          return (
                            <Fragment key={`${message.id}-${i}`}>
                              <Message
                                from={message.role as "user" | "assistant"}
                              >
                                <MessageContent>
                                  <Response>{part.text}</Response>
                                </MessageContent>
                                <div className="hidden sm:block">
                                  {message.role === "user" ? (
                                    <div className="size-8 rounded-full bg-black flex items-center justify-center">
                                      <User
                                        strokeWidth={3}
                                        className="size-4 text-white"
                                      />
                                    </div>
                                  ) : (
                                    <Avatar className={cn("size-8")}>
                                      <AvatarImage
                                        alt=""
                                        className="mt-0 mb-0"
                                        src={gentLogo}
                                      />
                                    </Avatar>
                                  )}
                                </div>
                              </Message>
                              {message.role === "assistant" &&
                                i === message.parts.length - 1 &&
                                message.id === messages.at(-1)?.id && (
                                  <Actions className="mt-2">
                                    <Action
                                      onClick={() => regenerate()}
                                      label="Retry"
                                    >
                                      <RefreshCcwIcon className="size-3" />
                                    </Action>
                                    <Action
                                      onClick={() =>
                                        navigator.clipboard.writeText(part.text)
                                      }
                                      label="Copy"
                                    >
                                      <CopyIcon className="size-3" />
                                    </Action>
                                  </Actions>
                                )}
                            </Fragment>
                          );
                        default:
                          return null;
                      }
                    })
                  )}
                </div>
              ))
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <PromptInput onSubmit={handleSubmit} className="mt-4">
          <PromptInputBody>
            <PromptInputTextarea
              onChange={(e) => setInput(e.target.value)}
              value={input}
            />
          </PromptInputBody>
          <PromptInputToolbar>
            <PromptInputTools></PromptInputTools>
            <PromptInputSubmit disabled={!input && !status} status={status} onStop={stop} />
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  );
}

export default App;
