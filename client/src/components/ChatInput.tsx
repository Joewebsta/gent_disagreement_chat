import {
  PromptInput,
  PromptInputBody,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import type { ChatStatus } from "ai";

interface ChatInputProps {
  input: string;
  status: ChatStatus;
  suggestions: string[];
  onInputChange: (value: string) => void;
  onSubmit: (message: PromptInputMessage) => void;
  onSuggestionClick: (suggestion: string) => void;
  onStop: () => void;
}

export function ChatInput({
  input,
  status,
  suggestions,
  onInputChange,
  onSubmit,
  onSuggestionClick,
  onStop,
}: ChatInputProps) {
  return (
    <div className="mt-4">
      <Suggestions>
        {suggestions.map((suggestion) => (
          <Suggestion
            key={suggestion}
            onClick={onSuggestionClick}
            suggestion={suggestion}
          />
        ))}
      </Suggestions>

      <PromptInput onSubmit={onSubmit} className="mt-4">
        <PromptInputBody>
          <PromptInputTextarea
            onChange={(e) => onInputChange(e.target.value)}
            value={input}
          />
        </PromptInputBody>
        <PromptInputToolbar>
          <PromptInputTools></PromptInputTools>
          <PromptInputSubmit
            disabled={!input && !status}
            status={status}
            onStop={onStop}
          />
        </PromptInputToolbar>
      </PromptInput>
    </div>
  );
}
