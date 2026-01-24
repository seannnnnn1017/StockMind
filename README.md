# StockMind Agent

Single-agent CLI that routes user requests to:
- Calculator (pure arithmetic)
- Date lookup (local date)
- Taiwan stock lookup (TWSE daily data)

The LLM plans tool steps and the agent executes them.

## Setup
1. Create a virtual environment.
2. `pip install -r requirements.txt`
3. Run LM Studio's local server (OpenAI API mode) at `http://127.0.0.1:1234/v1`
   with model `qwen/qwen3-8b` (default). Override with env vars if needed. 

Environment variables:
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TIMEOUT`

## Usage

Calculator:
```bash
python app.py "If Q4 revenue was 18.5M and Q3 was 15.2M, what's the QoQ growth?"
```

Date:
```bash
python app.py "What is today's date?"
```

Taiwan stock:
```bash
python app.py "Check TW stock 2330 close on 2025-01-24"
python app.py "Compare TW stock 3443 price growth between 2025-01-01 and today"
```

If the exact date is not a trading day, the tool falls back to the nearest prior
trading day.

## Logging
Use `--log-mode off|normal|detail` or the shortcut `--detail`.
You can also set `AGENT_LOG_MODE` as the default; CLI flags override it.

```bash
python app.py "Check TW stock 2330 close on 2025-01-24" --detail
```

## Mock mode
```bash
python app.py "2 + 2" --mock
```

Mock mode returns placeholder LLM responses. It is useful for wiring tests, but
may not produce valid tool plans.

## Notes
- Stock data comes from the TWSE daily endpoint and requires network access.
- `stock_daily` searches backward up to 12 months for a prior trading day when
  the exact date is missing.
- Request timeout is 1 second in `tools/tw_stock.py`; adjust if needed.
- For better tool-plan stability, it is recommended to disable the model's "thinking"/"reasoning" mode (if available) and run in normal chat/instruction mode. This reduces verbose chain-of-thought and helps the LLM adhere strictly to the tool JSON schema.
