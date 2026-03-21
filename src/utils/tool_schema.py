from typing import Any, Dict


def _render_langchain_tool(tool_name: str, tool: Any) -> list[str]:
    lines = [f"{tool_name}:"]

    if desc := getattr(tool, "description", None):
        lines.append(f"  description: {desc.strip()}")

    if output_key := getattr(tool, "output_key", None):
        lines.append(f"  writes to state: ${output_key}")

    args_schema = getattr(tool, "args_schema", None)
    if args_schema:
        lines.append("  args:")
        for field_name, field_info in args_schema.model_fields.items():
            field_type   = getattr(field_info.annotation, "__name__", "any")
            field_desc   = field_info.description or ""
            is_required  = field_info.is_required()
            default      = field_info.default if not is_required else None
            optional_tag = f" (optional, default={default!r})" if not is_required else ""
            lines.append(f"    - {field_name} ({field_type}{optional_tag}): {field_desc}")

    return lines

"""
the prompt the LLM receives look like this in the tools section:

paper_search:
  description: Fetches papers from both Arxiv and HuggingFace
  writes to state: $papers
  args:
    - search_query (str): Search query from user
    - limit (int, optional, default=10): Max papers per source
"""


def _render_plain_tool(tool_name: str, tool: Any) -> list[str]:
    """Renders a plain tool using a hand-written .schema attribute."""
    lines = [f"{tool_name}:"]
    schema = tool.schema

    if desc := schema.get("description"):
        lines.append(f"  description: {desc}")

    if output_key := schema.get("output_key"):
        lines.append(f"  writes to state: ${output_key}")

    for arg_name, meta in schema.get("args", {}).items():
        arg_type   = meta.get("type", "any")
        arg_desc   = meta.get("description", "")
        source     = meta.get("source", "")
        optional   = meta.get("optional", False)

        parts = []
        if source:
            parts.append(f"use {source!r}")
        if optional:
            parts.append("optional")

        suffix = f" → {', '.join(parts)}" if parts else ""
        lines.append(f"    - {arg_name} ({arg_type}){': ' + arg_desc if arg_desc else ''}{suffix}")

    return lines


def render_tool_schemas(tools: Dict[str, Any]) -> str:
    """
    Auto-renders tool descriptions for prompt injection.
    - LangChain BaseTools  → introspected from args_schema (Pydantic)
    - Plain tools          → read from .schema attribute
    """
    try:
        from langchain.tools import BaseTool as LangChainBaseTool
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False

    blocks = ["TOOLS AND REQUIRED ARGUMENTS:\n"]

    for tool_name, tool in tools.items():
        if HAS_LANGCHAIN and isinstance(tool, LangChainBaseTool):
            lines = _render_langchain_tool(tool_name, tool)
        elif hasattr(tool, "schema"):
            lines = _render_plain_tool(tool_name, tool)
        else:
            lines = [f"{tool_name}:\n  (no schema or args_schema found)"]

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
