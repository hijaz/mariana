# Mariana

Mariana is a local deep research CLI built with LangGraph, Ollama, SearXNG, and LangSmith.

## Install

1. Install dependencies in editable mode:

```bash
cd /home/hijaz/projects/mariana
python -m pip install -e .
```

2. Pull a local Ollama model, for example:

```bash
ollama pull gemma3:1b
```

3. Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

4. Start Ollama if it is not already running:

```bash
ollama serve
```

## Configuration

Mariana reads configuration from:

- `./.env` in the repository root (project defaults)
- `~/.mariana/.env` for runtime overrides
- environment variables at runtime

The repository defaults now use `gemma3:1b` for both `MARIANA_PLANNER_MODEL` and `MARIANA_SUMMARIZER_MODEL`.

### Block risk and delay

Because this tool uses local SearXNG to query search engines and scrape pages, it can still trigger rate limits or blocking if it requests many pages too fast. Mariana adds a default wait time between result fetches to reduce that risk.

You can tune the delay with:

```bash
MARIANA_SEARCH_DELAY_SECONDS=2.0
```

## Usage

```bash
python -m mariana research "What is LangGraph?"
```

To inspect current settings:

```bash
python -m mariana config
```

## Testing

Run tests with the package path configured:

```bash
cd /home/hijaz/projects/mariana
PYTHONPATH=src python -m pytest -q
```

## Notes

- The CLI performs a preflight check for Ollama connectivity and model availability.
- If the configured model is not available locally, it prints the available models and helps you pull the right one.
