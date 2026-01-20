"""Deterministic calculator tool."""
from __future__ import annotations

import ast
import operator
from typing import Any, Dict


class CalculatorTool:
    """Evaluates basic arithmetic expressions via AST interpretation."""

    _ops: Dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
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
        raise ValueError(f"Unsupported expression segment: {ast.dump(node)}")


if __name__ == "__main__":
    calculator = CalculatorTool()
    expr = '3 + 5 * (2 - 8) / 4 ** pi'
    try:
        result = calculator.run(expr)
        print(f"Result: {result}")
    except ValueError as e:
        print(e)