# AlphaMind

AlphaMind is a lightweight research agent stack for earnings-call analysis. The repository is structured as follows:

```
data/              # Raw, processed, embedding, and metadata artifacts
agents/            # Reader, analyst, critic, and memory agents
prompts/           # Prompt templates for each role
rag/               # Loader, chunker, embedder, retriever utilities
config/            # Model and path configuration files
app.py             # CLI entry point for orchestrating the workflow
```

The code is intentionally skeletal so you can plug in your preferred LLM client, RAG backend, and evaluation criteria.
