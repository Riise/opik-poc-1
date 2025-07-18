"""
LLM interaction with Opik tracing.
This example demonstrates how to use the OpenAI LLM with Opik for tracing.
"""

import os
from dotenv import load_dotenv
import opik
from openai import OpenAI

# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE)

# Configure Opik (this attaches Opik's tracing to the LLM call)
opik.configure()


# The Opik decorator `@opik.track` will automatically trace this function call.
@opik.track(name="custom_poc")
def ask_llm(prompt):
    """Function to ask the OpenAI LLM a question and return the response."""

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# Example interaction
if __name__ == "__main__":
    answer = ask_llm("What is the capital of Denmark?")
    print("LLM Response:", answer)
