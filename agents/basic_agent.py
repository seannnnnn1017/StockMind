"""Router agent that answers directly or uses a calculator plan."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
import calendar
from typing import Any, Dict, List, Tuple

from llm_client import LLMClient, LLMMessage
from tools.calculator import CalculatorTool
from tools.date_tool import DateTool
from tools.helpers import LoopTool, MemoryTool
from tools.quote_field import QuoteFieldTool
from tools.tw_stock import TaiwanStockTool

RESET = "\033[0m"           #red
AGENT_COLOR = "\033[36m"    #blue
TOOL_COLOR = "\033[33m"     #yello


@dataclass(slots=True)
class PlanStep: # single calculation step
    description: str
    expression: str
    store: str | None = None


@dataclass(slots=True)
class AgentResponse:  # final response from the agent
    task: str
    answer: str | None = None
    expression: str | None = None
    value: float | None = None
    outputs: Tuple[Tuple[str, float], ...] = ()
    reasoning: str | None = None
    steps: Tuple[PlanStep | ToolStep, ...] = ()


@dataclass(slots=True)
class RouteDecision:  # chose between direct answer or calculation (using tool)
    mode: str
    answer: str | None = None
    reasoning: str | None = None
    symbol: str | None = None
    date: str | None = None


@dataclass(slots=True)
class ToolStep:
    description: str
    action: str
    args: Dict[str, Any] | None = None
    expression: str | None = None
    store: str | None = None


class PlanParseError(Exception):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class BasicAgent:
    """Routes between direct LLM answers and calculator plans."""

    def __init__(
        self,
        llm: LLMClient,
        calculator: CalculatorTool | None = None,
        log_mode: str = "normal",
    ) -> None:
        self.llm = llm
        self.calculator = calculator or CalculatorTool()
        self.date_tool = DateTool()
        self.quote_field = QuoteFieldTool()
        self.stock_tool = TaiwanStockTool()
        self.loop = LoopTool()
        self.log_mode = log_mode if log_mode in {"off", "normal", "detail"} else "normal"

    def solve(self, task: str) -> AgentResponse: 
        decision = self._route_task(task)
        if decision.mode == "respond":
            return AgentResponse(task=task, answer=decision.answer, reasoning=decision.reasoning)
        if decision.mode == "date":
            self._log_tool("date.run()")
            today = self.date_tool.run()
            return AgentResponse(
                task=task,
                answer=f"Current date: {today}",
                reasoning=decision.reasoning,
            )
        if decision.mode == "stock":
            steps, reasoning = self._plan_tool_steps(task)
            context = self._execute_tool_steps(steps)
            answer = self._finalize_tool_answer(task, context)
            return AgentResponse(task=task, answer=answer, reasoning=reasoning, steps=tuple(steps))
        self._log("normal", f"{AGENT_COLOR}[agent] planning solution for task: {task}{RESET}")
        steps, reasoning = self._plan_steps(task)  # get plan steps from LLM
        for idx, step in enumerate(steps, start=1):
            self._log("normal", f"{AGENT_COLOR}[agent] planned step {idx}: {step.description} -> {step.expression}{RESET}")
        memory = MemoryTool()
        self._initialize_memory_from_task(task, memory)
        value, final_expr, outputs = self._execute_steps(steps, memory)
        return AgentResponse(
            task=task,
            answer=str(value),
            expression=final_expr,
            value=value,
            outputs=outputs,
            reasoning=reasoning,
            steps=tuple(steps),
        )

    def _route_task(self, task: str) -> RouteDecision:
        system = LLMMessage(
            role="system",
            content=( 
                "Decide whether the user needs a calculation, a date lookup, "
                "a Taiwan stock lookup, or a direct answer. "
                "If calculation is needed, respond with strict JSON: "
                "{\"mode\": \"calculate\", \"reasoning\": str}. "
                "If a direct response is needed, respond with strict JSON: "
                "{\"mode\": \"respond\", \"answer\": str, \"reasoning\": str}. "
                "If a date lookup is needed, respond with strict JSON: "
                "{\"mode\": \"date\", \"reasoning\": str}. "
                "If a Taiwan stock lookup is needed, respond with strict JSON: "
                "{\"mode\": \"stock\", \"symbol\": str, \"date\": str, \"reasoning\": str}. "
                "The symbol is a Taiwan ticker such as 2330. "
                "Date is optional and should be YYYYMMDD or YYYYMM when provided. "
                "Only return JSON."
            ),
        )
        user = LLMMessage(role="user", content=f"Task: {task}")
        raw = self.llm.chat([system, user], temperature=0)
        self._log("normal", f"{AGENT_COLOR}[agent] routing response:\n{self._format_plan_text(raw)}{RESET}")
        try:
            payload = raw[raw.find("{") : raw.rfind("}") + 1]
            data = json.loads(payload)
            mode = str(data.get("mode", "")).strip().lower()
            reasoning = data.get("reasoning")
            if mode == "respond":
                answer = str(data.get("answer", "")).strip()
                if answer:
                    return RouteDecision(mode="respond", answer=answer, reasoning=reasoning)
            if mode == "date":
                return RouteDecision(mode="date", reasoning=reasoning)
            if mode == "stock":
                symbol = str(data.get("symbol", "")).strip()
                date = str(data.get("date", "")).strip() or None
                return RouteDecision(mode="stock", symbol=symbol or None, date=date, reasoning=reasoning)
            if mode == "calculate":
                return RouteDecision(mode="calculate", reasoning=reasoning)
        except json.JSONDecodeError:
            pass
        return RouteDecision(mode="calculate")

    def _plan_steps(self, task: str) -> Tuple[List[PlanStep], str | None]:
        system_base = (
            "You orchestrate a calculator tool. Each step must be a single arithmetic expression using numbers,"
            " + - * / ** %, parentheses, the literal constants pi/tau/e, and the math functions "
            f"{', '.join(sorted(CalculatorTool.allowed_functions.keys()))}."  
            " Use the `store` field to name each result; reference previous values via {name}."
            " Do not emit memory.* calls, code, loops, tuples, or assignments—fully unroll the math."
            " Do NOT use math.pi. Use the literal constant 'pi'."
            " All constants must be written as pi, e, tau (no module prefixes)."
            " Respond with strict JSON: {\"reasoning\": str, \"steps\": [{\"description\": str, \"expression\": str, \"store\": str}]}."
        )

        last_error: str | None = None
        raw: str = ""

        for attempt in range(3):  # one initial try + one repair
            system = LLMMessage(
                role="system",
                content=(
                    system_base
                    + ("\n\nThe previous plan failed to parse because: " + last_error
                       + "\nRewrite the plan to strictly follow the tool constraints."
                       if last_error else "")
                ),
            )
            user = LLMMessage(role="user", content=f"Task: {task}\nReturn only JSON.")
            raw = self.llm.chat([system, user])
            self._log(
                "normal",
                f"{AGENT_COLOR}[agent] LLM plan response (attempt {attempt+1}):\n"
                f"{self._format_plan_text(raw)}{RESET}",
            )

            try:
                return self._parse_plan(raw, task)
            except PlanParseError as e:
                last_error = str(e)
                self._log("normal", f"\033[31m[agent] plan rejected: {last_error}\033[0m")
                self._log("detail", f"\033[35m[agent] system prompt was:\n{system.content}\033[0m")
                self._log("detail", f"\033[35m[agent] user prompt was:\n{user.content}\033[0m")
                continue

        # Final fallback if repair also fails
        fallback = PlanStep(description="Compute value", expression=self._fallback_expression(task))
        return [fallback], None

    def _plan_tool_steps(self, task: str) -> Tuple[List[ToolStep], str | None]:
        system_base = (
            "You orchestrate tools to answer Taiwan stock questions. "
            "Return strict JSON: {\"reasoning\": str, \"steps\": [step...]}. "
            "Each step: {\"description\": str, \"action\": str, \"args\": obj, \"expression\": str, "
            "\"store\": str}. "
            "Actions:\n"
            "- date: args {} -> returns YYYY-MM-DD.\n"
            "- date_offset: args {\"base\": str, \"days\": int, \"months\": int, \"years\": int} "
            "-> returns YYYYMMDD.\n"
            "- date_year_start: args {\"base\": str} -> returns YYYYMMDD for Jan 1 of the base year.\n"
            "- stock_daily: args {\"symbol\": str, \"date\": str} -> returns daily quote dict. "
            "If the exact date is missing (holiday), it returns the nearest prior trading day.\n"
            "- stock_month: args {\"symbol\": str, \"month\": str} -> returns list of daily quotes.\n"
            "- stock_range: args {\"symbol\": str, \"start\": str, \"end\": str} -> returns list of daily quotes.\n"
            "- stock_recent: args {\"symbol\": str, \"count\": int} -> returns list of daily quotes.\n"
            "- quote_field: args {\"quote\": str, \"field\": str} -> returns numeric field.\n"
            "- calc: expression with placeholders like {name} -> returns scalar.\n"
            "Use store to name results and reference them with {name}. "
            "If the task asks for a specific date with fallback to the nearest date, "
            "a single stock_daily call is sufficient.\n"
            "Use quote_field to extract numeric fields before calc; do not access dict fields in calc.\n"
            "Calc expressions must be arithmetic using numbers/constants and {placeholders} only.\n"
            "Example: stock_daily -> quote_field close -> calc "
            "\"({today_close}-{start_close})/{start_close}*100\".\n"
            "For quote_field, set args.quote to a stored quote key (e.g., \"{stock_today}\").\n"
            "Always use date format YYYY-MM-DD (e.g., 2020-1-23) in args for date/month fields.\n"
            "If the task specifies a concrete date, do NOT use action=date; "
            "use that date directly in stock_* args.\n"
            "Use date tools to derive \"today\" or ranges when needed. "
            "Only return JSON."
        )
        last_error: str | None = None
        raw: str = ""
        for attempt in range(3):
            system = LLMMessage(
                role="system",
                content=(
                    system_base
                    + ("\n\nThe previous plan failed to parse because: " + last_error
                       + "\nRewrite the plan to strictly follow the tool constraints."
                       if last_error else "")
                ),
            )
            user = LLMMessage(role="user", content=f"Task: {task}\nReturn only JSON.")
            raw = self.llm.chat([system, user], temperature=0)
            self._log(
                "normal",
                f"{AGENT_COLOR}[agent] LLM tool plan response (attempt {attempt+1}):\n"
                f"{self._format_tool_plan_text(raw)}{RESET}",
            )
            try:
                return self._parse_tool_plan(raw, task)
            except PlanParseError as e:
                last_error = str(e)
                self._log("normal", f"\033[31m[agent] tool plan rejected: {last_error}\033[0m")
                continue
        fallback = self._fallback_tool_plan(task)
        return fallback, None

    def _parse_tool_plan(self, raw: str, task: str) -> Tuple[List[ToolStep], str | None]:
        try:
            payload = raw[raw.find("{") : raw.rfind("}") + 1]
            data = json.loads(payload)
            steps: List[ToolStep] = []
            task_has_date = bool(re.search(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", task))
            task_needs_today = bool(re.search(r"(今天|現在|目前|今|now|today)", task, re.IGNORECASE))
            allowed_actions = {
                "date",
                "date_offset",
                "date_year_start",
                "stock_daily",
                "stock_month",
                "stock_range",
                "stock_recent",
                "quote_field",
                "calc",
            }
            for idx, item in enumerate(data.get("steps", []), start=1):
                action = str(item.get("action", "")).strip()
                if action not in allowed_actions:
                    raise PlanParseError(f"Unsupported action: {action}", raw)
                description = str(item.get("description", f"Step {idx}")).strip() or f"Step {idx}"
                args = item.get("args") or None
                expression = str(item.get("expression", "")).strip() or None
                store = str(item.get("store", "")).strip() or None
                if action == "calc":
                    if not expression:
                        inferred = self._infer_calc_expression(store, steps)
                        if inferred:
                            expression = inferred
                        else:
                            raise PlanParseError("calc steps require expression", raw)
                    if any(token in expression for token in ["=", "def", "lambda", "import", ";"]):
                        raise PlanParseError("calc expressions cannot include statements or assignments", raw)
                    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\.", expression):
                        raise PlanParseError("calc expressions cannot access object fields; use quote_field", raw)
                    if any(ch in expression for ch in ["[", "]", "'", "\""]):
                        raise PlanParseError("calc expressions cannot index dicts; use quote_field", raw)
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression) and expression not in {"pi", "e", "tau"}:
                        inferred = self._infer_calc_expression(store, steps)
                        if inferred:
                            expression = inferred
                        else:
                            raise PlanParseError(
                                "calc expression must be arithmetic using numbers/constants or {placeholders}", raw
                            )
                    if re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_]*\}", expression):
                        inferred = self._infer_calc_expression(store, steps)
                        if inferred:
                            expression = inferred
                        else:
                            raise PlanParseError("calc expression must be arithmetic, not a single placeholder", raw)
                    if store and re.search(r"\{" + re.escape(store) + r"\}", expression):
                        raise PlanParseError("calc expression cannot reference its own store", raw)
                # Allow date tool usage even when a concrete date exists; let tests validate behavior.
                if action in {"date_offset", "date_year_start"} and not isinstance(args, dict):
                    raise PlanParseError(f"{action} requires args", raw)
                if action in {"stock_daily", "stock_month", "stock_range", "stock_recent", "quote_field"} and not isinstance(args, dict):
                    raise PlanParseError(f"{action} requires args", raw)
                if action == "quote_field":
                    if "quote" not in args or "field" not in args:
                        raise PlanParseError("quote_field requires args.quote and args.field", raw)
                steps.append(
                    ToolStep(
                        description=description,
                        action=action,
                        args=args if isinstance(args, dict) else None,
                        expression=expression,
                        store=store,
                    )
                )
            if not steps:
                raise PlanParseError("No valid steps were produced.", raw)
            return steps, data.get("reasoning")
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Invalid JSON returned by LLM: {exc}", raw) from exc

    def _parse_plan(self, raw: str, task: str) -> Tuple[List[PlanStep], str | None]:
        try:
            payload = raw[raw.find("{") : raw.rfind("}") + 1]
            data = json.loads(payload)
            steps: List[PlanStep] = []
            for idx, item in enumerate(data.get("steps", []), start=1):
                expression = str(item.get("expression", "")).strip()
                description = str(item.get("description", f"Step {idx}")).strip() or f"Step {idx}"
                store = (str(item.get("store")).strip() or None) if item.get("store") else None
                if not expression:
                    continue
                if "," in expression:
                    raise PlanParseError(f"Commas are not supported in expressions: {expression}", raw)
                if any(bad in expression for bad in ["=", "def", "lambda", "import", ";"]):
                    raise PlanParseError(
                        "Assignment, definitions, or statements are not allowed. "
                        f"Offending expression: {expression}", raw
                    )
                if "\n" in expression:
                    raise PlanParseError(
                        "Multi-line expressions are not supported; each step must be a single arithmetic expression.",
                        raw,
                    )
                placeholders = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", expression)
                for placeholder in placeholders:
                    if placeholder in {"pi", "tau", "e"}:
                        raise PlanParseError(
                            f"Use literal constants (e.g., pi) rather than {{{placeholder}}}.", raw
                        )
                stripped = expression
                for placeholder in placeholders:
                    stripped = stripped.replace(f"{{{placeholder}}}", "")
                allowed_words = {"pi", "tau", "e"} | set(CalculatorTool.allowed_functions.keys())
                for word in allowed_words:
                    stripped = re.sub(rf"\b{word}\b", "", stripped)
                if re.search(r"[A-Za-z_]", stripped):
                    raise PlanParseError(
                        "Variables other than placeholders or supported function names are not allowed.", raw
                    )
                steps.append(PlanStep(description=description, expression=expression, store=store))
            if not steps:
                raise PlanParseError("No valid steps were produced.", raw)
            return steps, data.get("reasoning")
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Invalid JSON returned by LLM: {exc}", raw) from exc

    def _fallback_expression(self, task: str) -> str:
        lower = task.lower()
        if "pi" in lower:
            return str(math.pi)
        if " tau" in lower or lower.startswith("tau"):
            return str(math.tau)
        if " e" in lower or lower.startswith("e"):
            return str(math.e)
        digits = [c for c in task if c.isdigit() or c in ".+-*/()"]
        expr = "".join(digits).strip()
        return expr or "0"

    def _execute_steps(self, steps: List[PlanStep], memory: MemoryTool) -> Tuple[float, str, Tuple[Tuple[str, float], ...]]:
        last_value = 0.0
        last_expr = "0"
        stored: List[Tuple[str, float]] = []
        for idx, step in enumerate(steps, start=1):
            substituted = self._substitute(step.expression, memory)
            normalized = self._normalize(substituted)
            self._log("normal", f"{AGENT_COLOR}[agent] executing step {idx}: {step.description} -> {substituted}{RESET}")
            self._log("detail", f"{AGENT_COLOR}[agent] normalized expr: {normalized}{RESET}")
            self._log_tool(f"calculator.run({normalized})")
            try:
                value = self.calculator.run(normalized)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed on step {idx} ({step.description}): {exc}") from exc
            target = step.store or f"step{idx}"
            memory.set(target, value)
            if step.store:
                stored.append((step.store, value))
            last_value = value
            last_expr = normalized
        return last_value, last_expr, tuple(stored)

    def _substitute(self, expression: str, memory: MemoryTool) -> str:
        pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            try:
                return str(memory.get(key))
            except KeyError as exc:
                raise ValueError(f"referenced value '{key}' is undefined") from exc

        return pattern.sub(replace, expression)

    def _normalize(self, expression: str) -> str:
        replacements = {"pi": str(math.pi), "tau": str(math.tau), "e": str(math.e)}
        pattern = re.compile(r"\b(pi|tau|e)\b")
        return pattern.sub(lambda m: replacements[m.group(0)], expression)

    def _execute_tool_steps(self, steps: List[ToolStep]) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for idx, step in enumerate(steps, start=1):
            action = step.action
            if action == "date":
                self._log_tool("date.run()")
                value = self.date_tool.run()
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'}={value}{RESET}")
                continue
            if action == "date_offset":
                args = self._resolve_args(step.args or {}, context)
                base = args.get("base") or self.date_tool.run()
                days = int(args.get("days", 0) or 0)
                months = int(args.get("months", 0) or 0)
                years = int(args.get("years", 0) or 0)
                value = self._offset_date(str(base), days=days, months=months, years=years)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'}={value}{RESET}")
                continue
            if action == "date_year_start":
                args = self._resolve_args(step.args or {}, context)
                base = args.get("base") or self.date_tool.run()
                value = self._year_start(str(base))
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'}={value}{RESET}")
                continue
            if action == "stock_daily":
                args = self._resolve_args(step.args or {}, context)
                symbol = str(args.get("symbol", "")).strip()
                date = args.get("date")
                suffix = f", {date}" if date else ""
                self._log_tool(f"tw_stock.fetch_daily({symbol}{suffix})")
                value = self.stock_tool.fetch_daily(symbol, str(date) if date else None)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'} (quote){RESET}")
                continue
            if action == "stock_month":
                args = self._resolve_args(step.args or {}, context)
                symbol = str(args.get("symbol", "")).strip()
                month = str(args.get("month", "")).strip()
                self._log_tool(f"tw_stock.fetch_month({symbol}, {month})")
                value = self.stock_tool.fetch_month(symbol, month)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'} (list){RESET}")
                continue
            if action == "stock_range":
                args = self._resolve_args(step.args or {}, context)
                symbol = str(args.get("symbol", "")).strip()
                start = str(args.get("start", "")).strip()
                end = str(args.get("end", "")).strip()
                self._log_tool(f"tw_stock.fetch_range({symbol}, {start}, {end})")
                value = self.stock_tool.fetch_range(symbol, start, end)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'} (list){RESET}")
                continue
            if action == "stock_recent":
                args = self._resolve_args(step.args or {}, context)
                symbol = str(args.get("symbol", "")).strip()
                count = int(args.get("count", 0))
                self._log_tool(f"tw_stock.fetch_recent({symbol}, {count})")
                value = self.stock_tool.fetch_recent(symbol, count)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'} (list){RESET}")
                continue
            if action == "quote_field":
                args = self._resolve_args(step.args or {}, context)
                quote_arg = args.get("quote")
                field = str(args.get("field", "")).strip()
                if quote_arg is None:
                    raise ValueError(f"quote_field step {idx} missing quote")
                if isinstance(quote_arg, dict):
                    quote = quote_arg
                    label = "quote"
                else:
                    quote_key = str(quote_arg).strip()
                    if not quote_key:
                        raise ValueError(f"quote_field step {idx} missing quote key")
                    quote = context.get(quote_key)
                    label = quote_key
                self._log_tool(f"quote_field.run({label}, {field})")
                value = self.quote_field.run(quote, field)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'}={value}{RESET}")
                continue
            if action == "calc":
                expression = step.expression or ""
                self._log("detail", f"{AGENT_COLOR}[agent] raw calc expr: {expression}{RESET}")
                substituted = self._substitute_context(expression, context)
                normalized = self._normalize(substituted)
                self._log("detail", f"{AGENT_COLOR}[agent] calc expr: {normalized}{RESET}")
                self._log_tool(f"calculator.run({normalized})")
                value = self.calculator.run(normalized)
                self._store_context(context, step.store or f"step{idx}", value)
                self._log("detail", f"{AGENT_COLOR}[agent] stored {step.store or f'step{idx}'}={value}{RESET}")
                continue
            raise ValueError(f"Unsupported action in execution: {action}")
        return context

    def _initialize_memory_from_task(self, task: str, memory: MemoryTool) -> None:
        # Patterns like: x=2, x = 2, or 'x is 2'
        assignment_pattern = re.compile(
            r"([a-zA-Z_][a-zA-Z0-9_]*)(?:_\{?(\d+)\}?)?\s*(?:=|is)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        for base, subscript, value in assignment_pattern.findall(task):
            name = f"{base}_{subscript}" if subscript else base
            try:
                memory.set(name, float(value))
                if subscript:
                    memory.set(f"{base}{subscript}", float(value))
            except ValueError:
                continue

    def _format_plan_text(self, raw: str) -> str:
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                reasoning = data.get("reasoning", "")
                steps = data.get("steps", [])
                lines = []
                if reasoning:
                    lines.append(f"Reasoning:\n  {reasoning}")
                if steps:
                    lines.append("Steps:")
                    for idx, step in enumerate(steps, start=1):
                        desc = step.get("description", "")
                        expr = step.get("expression", "")
                        store = step.get("store", "")
                        lines.append(f"  {idx}. desc={desc} | expr={expr} | store={store}")
                return "\n".join(lines) or raw
            except json.JSONDecodeError:
                return raw
        return raw

    def _format_tool_plan_text(self, raw: str) -> str:
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                reasoning = data.get("reasoning", "")
                steps = data.get("steps", [])
                lines = []
                if reasoning:
                    lines.append(f"Reasoning:\n  {reasoning}")
                if steps:
                    lines.append("Steps:")
                    for idx, step in enumerate(steps, start=1):
                        desc = step.get("description", "")
                        action = step.get("action", "")
                        store = step.get("store", "")
                        lines.append(f"  {idx}. {desc} | action={action} | store={store}")
                return "\n".join(lines) or raw
            except json.JSONDecodeError:
                return raw
        return raw

    def _substitute_context(self, expression: str, context: Dict[str, Any]) -> str:
        pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                raise ValueError(f"referenced value '{key}' is undefined")
            value = context[key]
            if not isinstance(value, (int, float)):
                raise ValueError(f"referenced value '{key}' is not numeric")
            return str(value)

        return pattern.sub(replace, expression)

    def _resolve_args(self, value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())
            if match:
                return context.get(match.group(1))
            pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
            return pattern.sub(lambda m: str(context.get(m.group(1), "")), value)
        if isinstance(value, list):
            return [self._resolve_args(item, context) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_args(val, context) for key, val in value.items()}
        return value

    def _store_context(self, context: Dict[str, Any], name: str, value: Any) -> None:
        if not name:
            return
        context[name] = value

    def _offset_date(self, base: str, days: int = 0, months: int = 0, years: int = 0) -> str:
        dt = self._parse_date_value(base)
        if years or months:
            dt = self._add_months(dt, years * 12 + months)
        if days:
            dt = dt + timedelta(days=days)
        return dt.strftime("%Y%m%d")

    def _year_start(self, base: str) -> str:
        dt = self._parse_date_value(base)
        return f"{dt.year:04d}0101"

    def _parse_date_value(self, value: str) -> datetime:
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 8:
            year = int(digits[:4])
            month = int(digits[4:6])
            day = int(digits[6:8])
            return datetime(year, month, day)
        if len(digits) == 6:
            year = int(digits[:4])
            month = int(digits[4:6])
            return datetime(year, month, 1)
        if len(digits) == 4:
            year = int(digits)
            return datetime(year, 1, 1)
        raise ValueError(f"Invalid date value: {value}")

    def _add_months(self, dt: datetime, months: int) -> datetime:
        total_month = dt.month - 1 + months
        year = dt.year + total_month // 12
        month = total_month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return datetime(year, month, day)

    def _extract_symbol(self, task: str) -> str:
        match = re.search(r"\b(\d{4,6})\b", task)
        return match.group(1) if match else ""

    def _fallback_tool_plan(self, task: str) -> List[ToolStep]:
        symbols = re.findall(r"\b(\d{4,6})\b", task)
        symbol = symbols[0] if symbols else ""
        date_match = re.findall(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", task)
        if date_match and symbol:
            return [
                ToolStep(
                    description="Fetch daily quote",
                    action="stock_daily",
                    args={"symbol": symbol, "date": date_match[0]},
                    store="quote",
                ),
            ]
        return [
            ToolStep(
                description="Fetch latest quote",
                action="stock_daily",
                args={"symbol": symbol},
                store="quote",
            ),
        ]

    def _finalize_tool_answer(self, task: str, context: Dict[str, Any]) -> str:
        system = LLMMessage(
            role="system",
            content=(
                "Use the provided context to answer the user's question about Taiwan stocks. "
                "Do not invent numbers. If data is missing, say so. "
                "If a returned quote date differs from the requested date, "
                "explicitly mention the actual date used and explain it is the nearest prior trading day. "
                "Answer in Traditional Chinese, concise and clear."
            ),
        )
        user = LLMMessage(
            role="user",
            content=(
                f"Task: {task}\n"
                f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            ),
        )
        return self.llm.chat([system, user], temperature=0)

    def _infer_calc_expression(self, store: str | None, steps: List[ToolStep]) -> str | None:
        if not store:
            return None
        store_lower = store.lower()
        current, start = self._infer_close_pair(steps)
        if not current or not start:
            return None
        percent_keys = ["percent", "percentage", "rate", "pct"]
        amount_keys = ["amount", "delta", "diff", "growth", "increase"]
        if any(key in store_lower for key in percent_keys):
            return f"({{{current}}}-{{{start}}})/{{{start}}}*100"
        if any(key in store_lower for key in amount_keys):
            return f"{{{current}}}-{{{start}}}"
        return None

    def _infer_close_pair(self, steps: List[ToolStep]) -> Tuple[str | None, str | None]:
        candidates: List[str] = []
        for step in steps:
            if not step.store:
                continue
            name = step.store
            lowered = name.lower()
            if "close" in lowered or "price" in lowered:
                candidates.append(name)
        if not candidates:
            return None, None

        def pick(keys: List[str]) -> str | None:
            for name in candidates:
                lowered = name.lower()
                if any(key in lowered for key in keys):
                    return name
            return None

        current = pick(["today", "current", "now", "latest"])
        start = pick(["start", "year_start", "begin", "base", "initial"])
        if current and start and current != start:
            return current, start
        if len(candidates) >= 2:
            return candidates[-1], candidates[0]
        return None, None

    def _log_tool(self, detail: str) -> None:
        self._log("normal", f"{TOOL_COLOR}[tool] {detail}{RESET}")

    def _log(self, level: str, message: str) -> None:
        if self.log_mode == "off":
            return
        if self.log_mode == "normal" and level == "detail":
            return
        print(message, flush=True)
