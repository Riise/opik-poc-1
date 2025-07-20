"""
LangChain LLM interaction with Opik tracing.
This example demonstrates how to use the LangChain and OpenAI with Opik for tracing.
"""

import os
from typing import List, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
import opik
from opik.integrations.langchain import OpikTracer
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Configure Comet Opik
opik.configure(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)


class State(BaseModel):
    """State model for LangGraph."""

    messages: Annotated[List, add_messages]


def ask_llm(state: State) -> State:
    """Function to ask the OpenAI LLM a question and return the response."""

    # Create the LLM Chain using LangChain
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    response = model.invoke(state.messages)

    state.messages.append(
        HumanMessage(
            content=response.content,
            additional_kwargs={"role": "assistant"},
        )
    )

    return state


graph = StateGraph(State)
graph.add_node("chatbot", ask_llm)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

# Create the OpikTracer
opik_tracer = OpikTracer(graph=app.get_graph(xray=True))


# The Opik decorator `@opik.track` will automatically trace this function call
# (optional as the opik_tracer is tracing more detailed LangGraph interaction).
@opik.track(name="langgraph_poc")
def ask_agent(prompt: str) -> str:
    """Function to ask LangGraph a question and return the response."""

    config: RunnableConfig = {"callbacks": [opik_tracer]}
    messages = [HumanMessage(content=prompt)]
    initial_state = State(messages=messages)

    # Pass the OpikTracer callback to the Graph.invoke function
    result = app.invoke(initial_state, config)

    result = State(**result)
    return result.messages[-1].content


# Example interaction
if __name__ == "__main__":
    answer = ask_agent("What is the capital of Denmark?")
    print("Agent Response:", answer)
