"""
Low-level Opik API usage example.
"""

from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import opik
from openai import OpenAI

MODEL = "gpt-3.5-turbo"

# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Create Comet Opik client
# This is the low-level API usage, not using any specific integration.
# You can use this to create traces, spans, and log LLM calls manually.
opik_client = opik.Opik(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
    project_name=os.getenv("OPIK_PROJECT_NAME"),
)


def ask_llm(prompt: str, temperature: float, trace: opik.Trace):
    """Function to ask the OpenAI LLM a question and return the response."""

    openai_client = OpenAI()

    span = trace.span(
        name="llm_call",
        type="llm",
        input={"prompt": prompt},
        provider="openai",
        model=MODEL,
        metadata={
            "temperature": temperature,
        },
    )

    try:
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        span.update(
            output={"response": response.choices[0].message.content},
            usage=response.usage.to_dict() if response.usage else None,
        )

    finally:
        span.end()

    return response.choices[0].message.content


def multiple_calls(prompt: str):
    """
    Function to demonstrate multiple LLM calls.
    """

    # Create a trace
    trace = opik_client.trace(
        name="low_level_poc",
        input={"prompt": prompt},
        tags=["custom", "example"],
        start_time=datetime.now(timezone.utc),
    )

    responses = []

    try:
        temperatures = [0.0, 1.0, 2.0]  # Different temperatures for variety
        for temperature in temperatures:
            response = ask_llm(prompt, temperature, trace)  # type: ignore
            responses.append(response)

    finally:
        trace.update(
            output={"responses": responses},
            end_time=datetime.now(timezone.utc),
        )
        trace.end()

    return responses


# Example interaction
if __name__ == "__main__":
    answer = multiple_calls("Find a name for a dog.")
    print("LLM Response:", answer)

    opik_client.flush()  # Ensure all traces are sent before exiting
