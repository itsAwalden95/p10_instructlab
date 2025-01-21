from openai import OpenAI
import os
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage

checkpointer = MemorySaver()


IP = "129.40.94.153"  # IP of your TechZone LPAR
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://{IP}:8000/v1",
)

class State(TypedDict):
    messages: Annotated[list, add_messages] = []
    current_input: str = ""
    response: str = ""


def get_user_input(state):
    user_prompt = input("Enter your question (or 'exit' to quit): ")
    return {
        "current_input": user_prompt,
        "messages": state.get("messages", []) + [HumanMessage(content=user_prompt)],
        "response": ""
    }



def generate_response(state):
    user_input = state["current_input"]
    messages = state.get("messages", [])
    
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            formatted_messages.append(msg)
        elif isinstance(msg, dict) and 'content' in msg:
            formatted_messages.append({"role": "user", "content": msg['content']})
        elif isinstance(msg, str):
            formatted_messages.append({"role": "user", "content": msg})
    
    formatted_messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model='granite-7b-lab-Q4_K_M.gguf',
        messages=formatted_messages,
        temperature=0,
        stream=True
    )
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            print(content, end="", flush=True)
    print("\n")
    
    return {
        "current_input": state["current_input"],
        "response": full_response,
        "messages": messages + [HumanMessage(content=user_input), AIMessage(content=full_response)]
    }


# Define the graph
workflow = StateGraph(State)

workflow.add_node("user_input", get_user_input)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("user_input")
workflow.add_edge("user_input", "generate_response")
workflow.set_finish_point("generate_response")


# Compile the graph with the checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# Run the graph in a loop
initial_state = {"messages": [], "current_input": "", "response": ""}
while True:
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "checkpoint_ns": "my_namespace"
        }
    }
    result = graph.invoke(initial_state, config=config)
    if result["current_input"].lower() == "exit":
        break
    initial_state = result  # Update the state for the next iteration
