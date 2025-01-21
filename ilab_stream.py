from openai import OpenAI
import sys

def create_streaming_client(ip_address):
    """Create an OpenAI client configured for the iLab server."""
    return OpenAI(
        api_key="EMPTY",  # iLab doesn't require authentication
        base_url=f"http://{ip_address}:8000/v1"
    )

def stream_response(client, message):
    """Stream the response from the model."""
    try:
        response = client.chat.completions.create(
            model='granite-7b-lab-Q4_K_M.gguf',
            messages=[{"role": "user", "content": message}],
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
    
    # Initialize the client
    client = create_streaming_client(SERVER_IP)
    
    # Main interaction loop
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
            
        stream_response(client, user_input)

if __name__ == "__main__":
    main()