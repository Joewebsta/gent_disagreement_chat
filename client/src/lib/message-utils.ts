import type { TextUIPart, UIMessage } from "ai";

export function isTextPart(part: unknown): part is TextUIPart {
  if (!part || typeof part !== "object" || part === null) {
    return false;
  }

  const candidate = part as Record<string, unknown>;
  return candidate.type === "text" && typeof candidate.text === "string";
}

export function getMessageText(message: UIMessage): string {
  return (
    message.parts
      ?.filter(isTextPart)
      .map((part) => part.text)
      .join("") || ""
  );
}

export function hasMessageContent(message: UIMessage): boolean {
  return (
    message.parts?.some(
      (part) => isTextPart(part) && part.text && part.text.trim().length > 0
    ) || false
  );
}
