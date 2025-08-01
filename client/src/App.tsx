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
  AIInputButton,
  AIInputModelSelect,
  AIInputModelSelectContent,
  AIInputModelSelectItem,
  AIInputModelSelectTrigger,
  AIInputModelSelectValue,
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
  AISuggestion,
  AISuggestions,
} from "@/components/ui/kibo-ui/ai/suggestion";
import {
  AITool,
  AIToolContent,
  AIToolHeader,
  AIToolParameters,
  AIToolResult,
  type AIToolStatus,
} from "@/components/ui/kibo-ui/ai/tool";
import {
  CameraIcon,
  FileIcon,
  GlobeIcon,
  ImageIcon,
  MicIcon,
  PlusIcon,
  ScreenShareIcon,
} from "lucide-react";
import { type FormEventHandler, useState } from "react";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const messages: {
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
}[] = [
  {
    from: "user",
    versions: [
      {
        id: "1",
        content: "Can you explain how to use React hooks effectively?",
      },
    ],
    avatar: "https://github.com/haydenbleasel.png",
    name: "Hayden Bleasel",
  },
  {
    from: "assistant",
    // sources: [],
    // tools: [],
    versions: [
      {
        id: "1",
        content: `# React Hooks Best Practices
React hooks are a powerful feature that let you use state and other React features without writing classes. Here are some tips for using them effectively:
## Rules of Hooks
1. **Only call hooks at the top level** of your component or custom hooks
2. **Don't call hooks inside loops, conditions, or nested functions**
## Common Hooks
- **useState**: For local component state
- **useEffect**: For side effects like data fetching
- **useContext**: For consuming context
- **useReducer**: For complex state logic
- **useCallback**: For memoizing functions
- **useMemo**: For memoizing values
## Example of useState and useEffect
\`\`\`jsx
function ProfilePage({ userId }) {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    // This runs after render and when userId changes
    fetchUser(userId).then(userData => {
      setUser(userData);
    });
  }, [userId]);
  
  return user ? <Profile user={user} /> : <Loading />;
}
\`\`\`
Would you like me to explain any specific hook in more detail?`,
      },
    ],
    avatar: "https://github.com/openai.png",
    name: "OpenAI",
  },
];
const models = [
  { id: "gpt-4", name: "GPT-4", provider: "openai.com" },
  { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo", provider: "openai.com" },
  { id: "claude-2", name: "Claude 2", provider: "anthropic.com" },
  { id: "claude-instant", name: "Claude Instant", provider: "anthropic.com" },
  { id: "palm-2", name: "PaLM 2", provider: "google.com" },
  { id: "llama-2-70b", name: "Llama 2 70B", provider: "meta.com" },
  { id: "llama-2-13b", name: "Llama 2 13B", provider: "meta.com" },
  { id: "cohere-command", name: "Command", provider: "cohere.com" },
  { id: "mistral-7b", name: "Mistral 7B", provider: "mistral.ai" },
];
const suggestions = [
  "What are the latest trends in AI?",
  "How does machine learning work?",
];
function App() {
  const [model, setModel] = useState<string>(models[0].id);
  const [text, setText] = useState<string>("");
  const [useWebSearch, setUseWebSearch] = useState<boolean>(false);
  const [useMicrophone, setUseMicrophone] = useState<boolean>(false);
  const [status, setStatus] = useState<
    "submitted" | "streaming" | "ready" | "error"
  >("ready");
  const handleSubmit: FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    if (!text) {
      return;
    }
    toast.success("Message submitted", {
      description: text,
    });
    setStatus("submitted");
    setTimeout(() => {
      setStatus("streaming");
    }, 200);
    setTimeout(() => {
      setStatus("ready");
    }, 2000);
  };
  const handleFileAction = (action: string) => {
    toast.success("File action", {
      description: action,
    });
  };
  const handleSuggestionClick = (suggestion: string) => {
    toast.success("Suggestion clicked", {
      description: suggestion,
    });
    setStatus("submitted");
    setTimeout(() => {
      setStatus("streaming");
    }, 200);
    setTimeout(() => {
      setStatus("ready");
    }, 2000);
  };

  return (
    <div className="flex min-h-svh flex-col">
      <div className="max-w-screen-lg mx-auto">
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
                            <AISourcesTrigger count={message.sources.length} />
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
        <div className="grid shrink-0 gap-4 pt-4">
          <AISuggestions className="px-4">
            {suggestions.map((suggestion) => (
              <AISuggestion
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                suggestion={suggestion}
              />
            ))}
          </AISuggestions>
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
                  <AIInputModelSelect onValueChange={setModel} value={model}>
                    <AIInputModelSelectTrigger>
                      <AIInputModelSelectValue />
                    </AIInputModelSelectTrigger>
                    <AIInputModelSelectContent>
                      {models.map((model) => (
                        <AIInputModelSelectItem key={model.id} value={model.id}>
                          {/* <img
                          alt={model.provider}
                          className="inline-flex size-4"
                          height={16}
                          src={`https://img.logo.dev/${model.provider}?token=${process.env.NEXT_PUBLIC_LOGO_DEV_TOKEN}`}
                          width={16}
                        /> */}
                          {model.name}
                        </AIInputModelSelectItem>
                      ))}
                    </AIInputModelSelectContent>
                  </AIInputModelSelect>
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
