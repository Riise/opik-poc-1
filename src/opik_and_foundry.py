"""
Azure AI Foundry LLM interaction with Opik tracing.
This example demonstrates how to use the Azure AI Foundry LLM with Opik for tracing.
"""

import os
from typing import List
from dotenv import load_dotenv
from opik import track
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    ChatRequestMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.ai.inference.models import TextContentItem
from azure.core.credentials import AzureKeyCredential


# Load dotenv from DOT_ENV_FILE if it exists
DOT_ENV_FILE = os.getenv("DOT_ENV_FILE", ".env")
if os.path.exists(DOT_ENV_FILE):
    load_dotenv(DOT_ENV_FILE)

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    api_version="2024-08-01-preview",
)


# Create your chain
@track(name="foundry_poc")
def llm_chain(input_text: str) -> str:
    """
    Function to handle the LLM chain interaction.
    It retrieves context and generates a response based on the input text.
    """
    context = retrieve_context(input_text)
    response = generate_response(input_text, context)

    return response


@track
def retrieve_context(input_text: str) -> List[str]:
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


@track
def generate_response(input_text: str, context: List[str]):
    """
    Function to generate a response calling API.
    """

    system_msg = f"""
    You are a helpful assistant. 
    If the user asks a question that is not specific, use the context to provide a relevant response.
    Context: {context}
    """

    messages: List[ChatRequestMessage] = [
        SystemMessage(content=system_msg),
        AssistantMessage(content="Hello, how can I assist you today?"),
        UserMessage(
            content=[
                TextContentItem(text=input_text),
            ]
        ),
    ]

    response = client.complete(
        messages=messages,
        model="openai/gpt-4o-mini",
        temperature=1,
        top_p=1,
    )

    return response.choices[0].message.content


# Example interaction
if __name__ == "__main__":
    answer = llm_chain("What is the capital of Denmark?")
    print("LLM Response:", answer)
