"""Tools: plain Python functions the agent can call.

    @tool
    def get_order(order_id: str) -> str:
        "Look up an order by its id."
        return db[order_id]

The function's signature and docstring become the JSON schema the model sees.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def __call__(self, **kwargs) -> str:
        try:
            out = self.fn(**kwargs)
        except Exception as e:  # errors go back to the model, not up the stack
            return f"Error: {type(e).__name__}: {e}"
        return out if isinstance(out, str) else json.dumps(out, default=str)


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator turning a function into a `Tool`. Works bare or with arguments."""

    def wrap(f: Callable) -> Tool:
        hints = get_type_hints(f)
        props, required = {}, []
        for pname, p in inspect.signature(f).parameters.items():
            props[pname] = {"type": _JSON_TYPES.get(hints.get(pname, str), "string")}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return Tool(
            name=name or f.__name__,
            description=description or inspect.getdoc(f) or f.__name__,
            parameters={"type": "object", "properties": props, "required": required},
            fn=f,
        )

    return wrap(fn) if fn is not None else wrap
