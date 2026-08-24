# LinkedIn AI Agent

Module 21 - AI Agent Project.

A professional AI agent built with LangChain that generates LinkedIn-style
posts. The user provides a **topic** and a **language**, and the agent uses a
Claude model through LangChain to write the post.

## Status

Project scaffold only. The agent is not implemented yet.

## Tech stack

- Python 3.11+
- LangChain
- LangChain Anthropic integration (`langchain-anthropic`)
- Claude API (Anthropic)
- Streamlit
- python-dotenv
- pytest

## Project structure

```
linkedin-ai-agent/
├── app/
│   ├── __init__.py      # Marks app as a Python package
│   ├── agent.py         # Builds the LangChain chain / agent entry point
│   ├── config.py        # Loads .env and exposes settings (python-dotenv)
│   ├── llm.py           # Creates the configured ChatAnthropic model
│   └── prompts.py       # Prompt templates for the LinkedIn post
├── tests/
│   └── __init__.py      # Marks tests as a Python package
├── .env.example         # Template for required environment variables
├── .gitignore           # Files excluded from version control
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── streamlit_app.py     # Streamlit UI (topic + language input)
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Running

```bash
streamlit run streamlit_app.py
```

## Tests

```bash
pytest
```
# linkedin-ai-agent
