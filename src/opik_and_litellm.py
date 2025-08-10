"""
LiteLLM with Opik tracing.
This example demonstrates how to use LiteLLM with Opik for tracing.
"""

import os
import asyncio
from dotenv import load_dotenv
import opik
import litellm
from litellm.integrations.opik.opik import OpikLogger


# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE, override=True)

# Configure Comet Opik
opik.configure(
    api_key=os.getenv("COMET_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
)


@opik.track(name="litellm")
async def call_litellm(user_prompt: str) -> str:
    """Stream input to LiteLLM and return the response.

    Args:
        user_prompt (str): The input string to be processed.

    Returns:
        str | Unknown | None: The response from LiteLLM.
    """

    messages = [{"role": "user", "content": user_prompt}]
    model_response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=False,
    )
    return model_response.choices[0].message.content  # type: ignore


async def main():
    """Main function with example interaction."""

    opik_logger = OpikLogger()
    litellm.callbacks = [opik_logger]

    answer = await call_litellm("How many letter R are in strawberry?")
    print("LLM Response:", answer)


if __name__ == "__main__":
    asyncio.run(main())
