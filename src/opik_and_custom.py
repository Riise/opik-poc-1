"""
LLM interaction with Opik tracing.
This example demonstrates how to use the OpenAI LLM with Opik for tracing.
"""

import os
from dotenv import load_dotenv
import opik
from opik import opik_context
from openai import OpenAI

MAX_TOKENS = 200
SAMPLES = [
    {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
    },
    {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
    },
    {
        "model": "gpt-3.5-turbo",
        "temperature": 1.0,
    },
    {
        "model": "gpt-3.5-turbo",
        "temperature": 2.0,
    },
    {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
    {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
    {
        "model": "gpt-4o-mini",
        "temperature": 1.0,
    },
    {
        "model": "gpt-4o-mini",
        "temperature": 2.0,
    },
]

# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Configure Comet Opik
opik.configure(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)


@opik.track
def ask_llm(prompt: str, temperature: float, model: str):
    """Function to ask the OpenAI LLM a question and return the response."""

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )

    opik_context.update_current_span(
        provider="openai",
        model=model,
        metadata={
            "temperature": temperature,
        },
        usage=response.usage.to_dict() if response.usage else None,
    )

    return response.choices[0].message.content


# The Opik decorator `@opik.track` will automatically trace this function call.
@opik.track(
    name="custom_poc",
    metadata={
        "custom": "This is custom metadata",
        "max_tokens": MAX_TOKENS,
    },
    tags=["custom", "example"],
    type="llm",
)
def multiple_llm_calls(prompt: str):
    """
    Function to demonstrate multiple calls to the LLM.
    """
    responses = []
    for sample in SAMPLES:
        response = ask_llm(
            prompt,
            sample["temperature"],  # type: ignore
            sample["model"],
        )
        responses.append(response)
    return responses


# Example interaction
if __name__ == "__main__":
    answer = multiple_llm_calls("Find a name for a dog.")
    print("LLM Response:", answer)
