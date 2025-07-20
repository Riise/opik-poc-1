"""
LangChain LLM interaction with Opik tracing.
This example demonstrates how to use the LangChain and OpenAI with Opik for tracing.
"""

import os
from dotenv import load_dotenv
import opik
from langchain.chat_models import init_chat_model
from opik.integrations.langchain import OpikTracer


# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Configure Comet Opik
opik.configure(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)

# That's the only additional like we need.
opik_tracer = OpikTracer()


# The Opik decorator `@opik.track` will automatically trace this function call
# (optional as the opik_tracer is tracing more detailed LLM interaction).
@opik.track(name="langchain_poc")
def ask_llm(prompt: str):
    """Function to ask the OpenAI LLM a question and return the response."""

    # Create the LLM Chain using LangChain
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    response = model.invoke(prompt, config={"callbacks": [opik_tracer]})

    return response.content


# Example interaction
if __name__ == "__main__":
    answer = ask_llm("What is the capital of Denmark?")
    print("LLM Response:", answer)
