import { Button } from "@/components/ui/button";
import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export type ActionsProps = HTMLAttributes<HTMLDivElement>;

export const Actions = ({ className, children, ...props }: ActionsProps) => (
  <div className={cn("flex gap-2", className)} {...props}>
    {children}
  </div>
);

export type ActionProps = {
  onClick: () => void;
  label: string;
  children: ReactNode;
  className?: string;
};

export const Action = ({ onClick, label, children, className }: ActionProps) => (
  <Button
    variant="ghost"
    size="sm"
    onClick={onClick}
    className={cn("h-8 px-2", className)}
    title={label}
  >
    {children}
  </Button>
);