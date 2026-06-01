"""Tool-schema helpers shared across providers."""

import inspect


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _annotation_to_jsonschema(annotation) -> str:
    """Map a Python type annotation to a JSON-Schema primitive name."""
    # Handle PEP-604 union types (e.g. `str | None`): pick the first
    # non-None member as the declared type. Optional parameters are
    # represented by their default value, not a JSON-Schema "null".
    args = getattr(annotation, "__args__", None)
    if args:
        for a in args:
            if a is type(None):
                continue
            return _TYPE_MAP.get(a, "string")
    return _TYPE_MAP.get(annotation, "string")


def _function_parameters_schema(func) -> dict:
    """Build a JSON-Schema `parameters` object for a callable's args.

    Skips `self` and `**kwargs`-style catch-alls. Required parameters
    are those without defaults.
    """
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    properties: dict = {}
    required: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name == "self":
            continue
        properties[param.name] = {"type": _annotation_to_jsonschema(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def function_to_json(func) -> dict:
    """OpenAI Chat Completions tool spec (`{type:function, function:{...}}`).

    Kept for backwards compatibility with code that emitted tools as
    JSON in the SYSTEM_PROMPT during the legacy ReAct harness; new
    code should build a `ToolSpec` and let each provider wrapper
    render its native form.
    """
    parameters = _function_parameters_schema(func)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": parameters,
        },
    }


if __name__ == "__main__":
    import json

    def check_balance(account_id: int) -> str:
        """Checks the balance of the given account ID."""
        return f"Balance for account {account_id} is $1000."

    print(json.dumps(function_to_json(check_balance), indent=2))
