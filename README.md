# StockMind Minimal Agent

Single-agent example that asks LM Studio for an arithmetic expression, evaluates it with the built-in calculator tool, and prints the result.

## Setup
1. Create a virtual environment.
2. `pip install -r requirements.txt`
3. Run LM Studio's local server (OpenAI API mode) at `http://127.0.0.1:1234` with the model `wen/Qwen2.5-7B-Instruct-GGUF`. Use `LLM_BASE_URL` or `LLM_MODEL` env vars if your settings differ.

## Usage
```bash
python app.py "Start with a0 = 2, a1 = 3.         
Define the sequence:
a(n) = 2*a(n-1) + a(n-2)

1) Compute a2, a3, a4.
2) Let b = a4^2 - a3 * a2.
3) Compute c = b / a1 + pi.

What is the value of c?""
```

Append `--mock` to bypass LM Studio and see deterministic placeholder responses while keeping the calculator call.
