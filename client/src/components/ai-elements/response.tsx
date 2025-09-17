import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type ResponseProps = HTMLAttributes<HTMLDivElement>;

export const Response = ({ className, children, ...props }: ResponseProps) => (
  <div className={cn("prose prose-sm max-w-none", className)} {...props}>
    {children}
  </div>
);