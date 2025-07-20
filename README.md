# Comet Opik POC 1

This is a proof of concept for using [Comet Opik](https://docs.comet.com/llm/opik) for tracking LLM interactions.

## Scripts

- `src/opik_and_openai.py`: Script for tracking LLM interactions with Opek's OpenAI integration.
- `src/opik_and_foundry.py`: Script for tracking LLM interactions with Azure AI Foundry. Currently no Azure AI Foundry integration is available for Opik.
- `src/opik_and_custom.py`: Contains functions to handle LLM interactions with Open AI without using any integration.
- `src/opik_and_langchain.py`: Script for tracking LLM interactions using Opik's LangChain integration.

Make sure to have the necessary dependencies installed and configured.

## Dependencies & Configuration

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

If you are using the devcontainer, the dependencies are already installed.

Copy the file `.env.example` to `.env` and fill in the required environment variables.

The following accounts are required:

- OpenAI account with API key
- Comet account with API key
- GitHub account with personal access token (PAT) for model access
