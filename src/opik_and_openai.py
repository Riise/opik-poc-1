"""
This module demonstrates how to use Opik to track an OpenAI LLM chain.
It includes functions to retrieve context and generate a response using the OpenAI API.
"""

import os
from dotenv import load_dotenv
import opik
from opik.integrations.openai import track_openai
from openai import OpenAI

# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Configure Comet Opik
opik.configure(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)

# Wrap your OpenAI client
client = OpenAI()
client = track_openai(client)


# Create your chain
@opik.track(name="openai_poc")
def llm_chain(input_text):
    """
    Function to handle the LLM chain interaction.
    It retrieves context and generates a response based on the input text.
    """
    context = retrieve_context(input_text)
    response = generate_response(input_text, context)  # type: ignore

    return response


@opik.track
def retrieve_context(_input_text):
    """
    Function to retrieve context based on the input text.
    This is a placeholder function that simulates context retrieval.
    """
    # For the purpose of this example, we are just returning a hardcoded list of strings
    context = [
        "What specific information are you looking for?",
        "How can I assist you with your interests today?",
        "Are there any topics you'd like to explore or learn more about?",
    ]
    return context


@opik.track
def generate_response(input_text, context):
    """
    Function to generate a response using the OpenAI API.
    """
    full_prompt = (
        f" If the user asks a question that is not specific, use the context to provide a relevant response.\n"
        f"Context: {', '.join(context)}\n"
        f"User: {input_text}\n"
        f"AI:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content


# Example interaction
if __name__ == "__main__":
    answer = llm_chain("What is the capital of Denmark?")
    print("LLM Response:", answer)
