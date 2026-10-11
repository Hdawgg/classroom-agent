"""Cleans tool names on a model reply before the tool is dispatched.

gpt-oss models sometimes leak a harmony channel token into the tool name, e.g.
"generate_questions<|channel|>commentary" instead of "generate_questions", which then fails
as an unknown tool. Used by both the solve agent (agent.py) and the study agent (study_agent.py).
"""
from typing import Iterable, List, Tuple


def clean_tool_name(name) -> str:
    """Everything before the first "<|", without surrounding whitespace."""
    return str(name or "").split("<|", 1)[0].strip()


def normalize_tool_calls(message, known: Iterable[str]) -> List[Tuple[str, str, str]]:
    """Renames tool calls on `message` (an AIMessage) in place, only when the cleaned name is a
    known tool; unknown names are left untouched so the caller reports them as before.
    Also fixes additional_kwargs["tool_calls"], which is what gets sent back to the API next turn.
    Returns (call_id, raw_name, clean_name) for every call that was renamed."""
    known = set(known)

    def fixed(name):
        clean = clean_tool_name(name)
        return clean if isinstance(name, str) and clean != name and clean in known else None

    renamed = []
    for tc in list(getattr(message, "tool_calls", None) or []) + list(getattr(message, "invalid_tool_calls", None) or []):
        clean = fixed(tc.get("name"))
        if clean:
            renamed.append((tc.get("id"), tc["name"], clean))
            tc["name"] = clean
    for tc in (getattr(message, "additional_kwargs", None) or {}).get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        clean = fixed(fn.get("name")) if isinstance(fn, dict) else None
        if clean:
            fn["name"] = clean
    return renamed
