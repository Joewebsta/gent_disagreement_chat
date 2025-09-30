import gentLogo from "@/assets/assistant-logo.png";
import { Action, Actions } from "@/components/ai-elements/actions";
import { Loader } from "@/components/ai-elements/loader";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { Response } from "@/components/ai-elements/response";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { ChatAvatar } from "@/components/ChatAvatar";
import { hasMessageContent, isTextPart } from "@/lib/message-utils";
import { cn } from "@/lib/utils";
import type { UIMessage } from "ai";
import { CopyIcon, RefreshCcwIcon } from "lucide-react";
import { Fragment } from "react";

interface ChatMessageProps {
  message: UIMessage;
  isLastMessage: boolean;
  onRegenerate: () => void;
}

export function ChatMessage({
  message,
  isLastMessage,
  onRegenerate,
}: ChatMessageProps) {
  if (message.role === "assistant" && !hasMessageContent(message)) {
    return (
      <Message from="assistant" key={`${message.id}-loading`}>
        <MessageContent>
          <Loader className="my-[2px]" />
        </MessageContent>
        <div className="hidden sm:block">
          <Avatar className={cn("size-8")}>
            <AvatarImage alt="" className="mt-0 mb-0" src={gentLogo} />
          </Avatar>
        </div>
      </Message>
    );
  }

  return (
    <>
      {message.parts.map((part, i) => {
        switch (part.type) {
          case "text":
            return (
              <Fragment key={`${message.id}-${i}`}>
                <Message from={message.role as "user" | "assistant"}>
                  <MessageContent>
                    <Response>{part.text}</Response>
                  </MessageContent>
                  <div className="hidden sm:block">
                    <ChatAvatar role={message.role as "user" | "assistant"} />
                  </div>
                </Message>
                {message.role === "assistant" &&
                  i === message.parts.length - 1 &&
                  isLastMessage && (
                    <Actions className="mt-2">
                      <Action onClick={onRegenerate} label="Retry">
                        <RefreshCcwIcon className="size-3" />
                      </Action>
                      <Action
                        onClick={() =>
                          navigator.clipboard.writeText(
                            isTextPart(part) ? part.text : ""
                          )
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
      })}
    </>
  );
}