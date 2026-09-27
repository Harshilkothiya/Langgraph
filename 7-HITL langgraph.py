from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.types import interrupt, Command
import random

#llm
from model import model
llm = model

#state of chatbot
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#toos

@tool
def get_stcok_price(name:str)->int:
    """
    Fetch latest stock price for a given name (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    prize = random.randint(10, 100)

    return {
        "name":name,
        "prize of one share":prize
    }


# use this tool to implement the HITL
@tool
def purchase_stock(name:str, quantity:int):
    """
    Simulate purchasing a given quantity of a stock name.
    """
    print("for me purchase tool calling.............")
    decision = interrupt({
        "question": f"Approve buying {quantity} shares of {name}? (yes/no)"
    })

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {name}.",
        "name": name,
        "quantity": quantity,
    }

    return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {name} was declined by human.",
            "name": name,
            "quantity": quantity,
        }



tools = [get_stcok_price, purchase_stock]
llm_tool = llm.bind_tools(tools)

# make the graph

def chat_node(state:ChatState):
    '''LLM node'''
    message = state['messages']
    response = llm_tool.invoke(message)

    return {"messages": [response]}

tool_node = ToolNode(tools)


# Checkpointer (in-memory)
memory = MemorySaver()


# Graph

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)



# run this chatbot 

if __name__ == "__main__":

    while True:
        user = input("You: ")

        if user.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break


        result = chatbot.invoke(
            {"messages":HumanMessage(user)}, 
            config={"configurable":{"thread_id":"harshil"}}
        )

        # Check for HITL interrupt from purchase_stock

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            return_user = interrupts[0].value
            print("pls conform: ", return_user)

            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable":{"thread_id":"harshil"}}
            )

        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content[0]['text']}\n")