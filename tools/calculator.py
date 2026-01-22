"""Deterministic calculator tool."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Iterable


class CalculatorTool:
    """
    Tool: calculator

    Purpose:
      Evaluate a pure arithmetic expression and return a number.

    Input:
      expression: string

    Grammar:
      - numbers (int, float)
      - operators: + - * / ** %
      - parentheses
      - constants: pi, e, tau
      - math functions: sin, cos, tan, log, exp, sqrt, abs, floor, ceil
      - NO variables outside of stored placeholders
      - NO '='
      - NO user-defined function calls
      - NO memory access

    Semantics:
      - Deterministic
      - Side-effect free
      - Single-step evaluation

    Usage rules for agents:
      - One tool call per step
      - Expression must be self-contained
      - Results must be stored and referenced via {name}
    """
    _ops: Dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }
    allowed_functions: Dict[str, Any] = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    def run(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression: {expression}") from exc
        return float(self._eval_node(tree.body))

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return float(node.n)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return float(self._ops[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
            return float(self._ops[type(node.op)](self._eval_node(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.allowed_functions:
                args = [self._eval_node(arg) for arg in node.args]
                if len(args) != 1:
                    raise ValueError(f"{func_name} expects exactly 1 argument")
                return float(self.allowed_functions[func_name](args[0]))
        raise ValueError(f"Unsupported expression segment: {ast.dump(node)}")


if __name__ == "__main__":
    calc = CalculatorTool()
    print(calc.run("3 + 4 * 2"))
    