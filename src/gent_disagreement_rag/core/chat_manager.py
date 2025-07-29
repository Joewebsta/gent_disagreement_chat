from .rag_service import RAGService


class ChatManager:
    def __init__(self):
        self.rag_service = RAGService()

    def run(self):
        print("Welcome to the A Gentleman's Disagreement Chatbot!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("-" * 50)

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("Thinking...")

            response = self.rag_service.ask_question(user_input)
            print(f"Assistant: {response}")

            print("-" * 50)
