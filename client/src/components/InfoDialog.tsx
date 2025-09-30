import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { InfoIcon } from "lucide-react";

export function InfoDialog() {
  return (
    <Dialog>
      <DialogTrigger className="cursor-pointer" asChild>
        <Button
          variant="ghost"
          size="icon"
          className="fixed top-4 right-4 z-10"
          aria-label="Information"
        >
          <InfoIcon className="h-4 w-4 text-muted-foreground" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>About A Gentleman's Disagreement</DialogTitle>
          <DialogDescription>
            Welcome to the A Gentleman's Disagreement chatbot! This AI
            assistant can help you explore topics and perspectives discussed on
            the podcast.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Ask questions about specific episodes, guests, topics, or general
            podcast information. The AI has been trained on podcast content to
            provide relevant and helpful responses.
          </p>
          <p className="text-sm text-muted-foreground">
            Try asking about recent episodes, specific topics of interest, or
            request summaries of discussions.
          </p>
          <p className="text-sm text-muted-foreground">
            This chatbot was created by friend of the pod,{" "}
            <a
              href="https://www.linkedin.com/in/joeswebster/"
              className="text-blue-500"
              target="_blank"
            >
              Joe Webster
            </a>
            .
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}