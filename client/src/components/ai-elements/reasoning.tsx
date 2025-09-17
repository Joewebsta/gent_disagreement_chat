import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDownIcon } from "lucide-react";
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type ReasoningProps = HTMLAttributes<HTMLDivElement> & {
  isStreaming?: boolean;
};

export const Reasoning = ({ className, children, ...props }: ReasoningProps) => (
  <Collapsible>
    <div className={cn("mb-4 border rounded-lg", className)} {...props}>
      {children}
    </div>
  </Collapsible>
);

export type ReasoningTriggerProps = HTMLAttributes<HTMLButtonElement>;

export const ReasoningTrigger = ({ className, ...props }: ReasoningTriggerProps) => (
  <CollapsibleTrigger
    className={cn(
      "flex w-full items-center justify-between p-3 text-sm font-medium hover:bg-muted/50 transition-colors",
      className
    )}
    {...props}
  >
    <span>Reasoning</span>
    <ChevronDownIcon className="size-4" />
  </CollapsibleTrigger>
);

export type ReasoningContentProps = HTMLAttributes<HTMLDivElement>;

export const ReasoningContent = ({ className, children, ...props }: ReasoningContentProps) => (
  <CollapsibleContent>
    <div className={cn("p-3 pt-0 text-sm text-muted-foreground", className)} {...props}>
      {children}
    </div>
  </CollapsibleContent>
);