import gentLogo from "@/assets/gent-disagreement-logo.png";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent } from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import {
  Action,
  Actions,
} from '@/components/ai-elements/actions';
import { Response } from '@/components/ai-elements/response';
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from '@/components/ai-elements/sources';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import { Loader } from '@/components/ai-elements/loader';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport, type UIMessage, type TextUIPart } from "ai";
import { User, RefreshCcwIcon, CopyIcon } from "lucide-react";
import { Fragment, useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";


function isTextPart(part: unknown): part is TextUIPart {
  if (!part || typeof part !== "object" || part === null) {
    return false;
  }

  const candidate = part as Record<string, unknown>;
  return (
    candidate.type === "text" &&
    typeof candidate.text === "string"
  );
}

function getMessageText(message: UIMessage): string {
  return message.parts
    ?.filter(isTextPart)
    .map(part => part.text)
    .join("") || "";
}

function App() {
  const [input, setInput] = useState<string>("");

  // Use the Vercel AI SDK's useChat hook with text streaming - replaces all manual streaming logic
  const { messages, sendMessage, status } = useChat({
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
      text: message.text || 'Sent with attachments'
    });
    setInput('');
    toast.success("Message sent");
  };

  const regenerate = () => {
    if (messages.length > 0) {
      const lastUserMessage = messages.slice().reverse().find(m => m.role === 'user');
      if (lastUserMessage) {
        sendMessage({ text: getMessageText(lastUserMessage) });
      }
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
      <div className="flex flex-col h-full">
        <Conversation className="h-full">
          <ConversationContent>
            {messages.map((message) => (
              <div key={message.id}>
                {message.role === 'assistant' && message.parts.filter((part) => part.type === 'source-url').length > 0 && (
                  <Sources>
                    <SourcesTrigger
                      count={
                        message.parts.filter(
                          (part) => part.type === 'source-url',
                        ).length
                      }
                    />
                    {message.parts.filter((part) => part.type === 'source-url').map((part, i) => (
                      <SourcesContent key={`${message.id}-${i}`}>
                        <Source
                          key={`${message.id}-${i}`}
                          href={part.url}
                          title={part.url}
                        />
                      </SourcesContent>
                    ))}
                  </Sources>
                )}
                {message.parts.map((part, i) => {
                  switch (part.type) {
                    case 'text':
                      return (
                        <Fragment key={`${message.id}-${i}`}>
                          <Message from={message.role as "user" | "assistant"}>
                            <MessageContent>
                              <Response>
                                {part.text}
                              </Response>
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
                                  <AvatarImage alt="" className="mt-0 mb-0" src={gentLogo} />
                                  <AvatarFallback>AI</AvatarFallback>
                                </Avatar>
                              )}
                            </div>
                          </Message>
                          {message.role === 'assistant' && i === message.parts.length - 1 && message.id === messages.at(-1)?.id && (
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
                    case 'reasoning':
                      return (
                        <Reasoning
                          key={`${message.id}-${i}`}
                          className="w-full"
                          isStreaming={status === 'streaming' && i === message.parts.length - 1 && message.id === messages.at(-1)?.id}
                        >
                          <ReasoningTrigger />
                          <ReasoningContent>{part.text}</ReasoningContent>
                        </Reasoning>
                      );
                    default:
                      return null;
                  }
                })}
              </div>
            ))}
            {status === 'submitted' && <Loader />}
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
            <PromptInputSubmit disabled={!input && !status} status={status} />
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  );
}

export default App;
