# Comet Opik POC 1

This is a proof of concept for using [Comet Opik](https://docs.comet.com/llm/opik) for tracking LLM interactions.

## Running POC Scripts

- `src/opik_and_openai.py`: Script for tracking LLM interactions with Opik's OpenAI integration.
- `src/opik_and_foundry.py`: Script for tracking LLM interactions with Azure AI Foundry. Currently no Azure AI Foundry integration is available for Opik.
- `src/opik_and_langchain.py`: Script for tracking LLM interactions using Opik's LangChain integration.
- `src/opik_and_langgraph.py`: Script for tracking AI agent interactions using Opik's LangGraph integration.
- `src/opik_low_level.py`: Demonstrates low-level Opik API usage.
- `src/opik_and_custom.py`: More advanced usage of Opik for custom LLM interactions.

Go to [comet.com](https://www.comet.com/opik/) to see the tracked interactions.

## Dependencies & Configuration

If not using the VS Code devcontainer, you need to install the required dependencies by running:

```bash
pip install -r requirements.txt
```

Copy the file `.env.example` to `.env` and fill in the required environment variables.

The following accounts are required:

- OpenAI account with API key
- Comet account with API key
- GitHub account with personal access token (PAT) for model access
