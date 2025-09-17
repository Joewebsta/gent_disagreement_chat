import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type LoaderProps = HTMLAttributes<HTMLDivElement>;

export const Loader = ({ className, ...props }: LoaderProps) => (
  <div className={cn("flex items-center gap-2 p-4", className)} {...props}>
    <div className="flex space-x-1">
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-bounce"></div>
    </div>
    <span className="text-sm text-muted-foreground">Thinking...</span>
  </div>
);