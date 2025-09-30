import gentLogo from "@/assets/assistant-logo.png";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { User } from "lucide-react";

interface ChatAvatarProps {
  role: "user" | "assistant";
}

export function ChatAvatar({ role }: ChatAvatarProps) {
  if (role === "user") {
    return (
      <div className="size-8 rounded-full bg-black flex items-center justify-center">
        <User strokeWidth={3} className="size-4 text-white" />
      </div>
    );
  }

  return (
    <Avatar className={cn("size-8")}>
      <AvatarImage alt="" className="mt-0 mb-0" src={gentLogo} />
    </Avatar>
  );
}