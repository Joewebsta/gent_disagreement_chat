import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ExternalLinkIcon } from "lucide-react";
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type SourcesProps = HTMLAttributes<HTMLDivElement>;

export const Sources = ({ className, children, ...props }: SourcesProps) => (
  <Collapsible>
    <div className={cn("mb-4", className)} {...props}>
      {children}
    </div>
  </Collapsible>
);

export type SourcesTriggerProps = {
  count: number;
  className?: string;
};

export const SourcesTrigger = ({ count, className }: SourcesTriggerProps) => (
  <CollapsibleTrigger asChild>
    <Button variant="outline" size="sm" className={cn("mb-2", className)}>
      <ExternalLinkIcon className="size-3 mr-1" />
      {count} source{count !== 1 ? 's' : ''}
    </Button>
  </CollapsibleTrigger>
);

export type SourcesContentProps = HTMLAttributes<HTMLDivElement>;

export const SourcesContent = ({ className, children, ...props }: SourcesContentProps) => (
  <CollapsibleContent>
    <div className={cn("space-y-2", className)} {...props}>
      {children}
    </div>
  </CollapsibleContent>
);

export type SourceProps = {
  href: string;
  title: string;
  className?: string;
};

export const Source = ({ href, title, className }: SourceProps) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className={cn(
      "block p-2 rounded border text-sm hover:bg-muted transition-colors",
      className
    )}
  >
    <div className="flex items-center gap-2">
      <ExternalLinkIcon className="size-3 flex-shrink-0" />
      <span className="truncate">{title}</span>
    </div>
  </a>
);