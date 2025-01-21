from openai import OpenAI
import sys

class ConversationManager:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_conversation_history(self):
        return self.messages

def create_streaming_client(ip_address):
    """Create an OpenAI client configured for the iLab server."""
    return OpenAI(
        api_key="EMPTY",  # iLab doesn't require authentication
        base_url=f"http://{ip_address}:8000/v1"
    )

def stream_response(client, conversation):
    """Stream the response from the model with conversation history."""
    try:
        response = client.chat.completions.create(
            model='granite-7b-lab-Q4_K_M.gguf',
            messages=conversation.get_conversation_history(),
            temperature=0,
            stream=True
        )
        
        # Stream the response chunks
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")  # Add a newline after the complete response
        return full_response
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None

def main():
    # Replace with your iLab server IP
    SERVER_IP = "129.40.95.233"
    
    # Initialize the client and conversation manager
    client = create_streaming_client(SERVER_IP)
    conversation = ConversationManager()
    
    # Main interaction loop
    print("Chat session started. Each message will maintain context from previous messages.")
    print("Type 'exit' to quit or 'clear' to start a new conversation.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            conversation = ConversationManager()
            print("\nConversation history cleared. Starting new chat.\n")
            continue
            
        # Add user message to history
        conversation.add_message("user", user_input)
        
        # Get and display response
        assistant_response = stream_response(client, conversation)
        if assistant_response:
            # Add assistant response to history
            conversation.add_message("assistant", assistant_response)

if __name__ == "__main__":
    main()
