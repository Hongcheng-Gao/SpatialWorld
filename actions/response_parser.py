"""Parse VLM responses with THINK and ACTION blocks."""

from typing import Any, Dict, Optional

from actions.parser import parse_action_string

THINK_PLACEHOLDER_IF_MISSING = "(No <THINK> block; action will still be executed)"
THINK_PLACEHOLDER_IF_EMPTY = "Model did not provide valid thinking (THINK content is empty)"


def extract_tag_block(text: str, tag: str) -> Optional[str]:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = text.find(start_token)
    end = text.find(end_token)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(start_token) : end]


def parse_vlm_response(
    response_text: str,
    enable_summary: bool = False,
    env_type: str = "ai2thor",
    executor_type: str | None = None,
) -> Dict[str, Any]:
    think_block = extract_tag_block(response_text, "THINK")
    action_block = extract_tag_block(response_text, "ACTION")
    if not action_block:
        raise ValueError("Missing <ACTION> tag")
    action_string = action_block.strip()
    if not action_string:
        raise ValueError("ACTION tag content is empty")
    if not think_block:
        thinking_text = THINK_PLACEHOLDER_IF_MISSING
    else:
        thinking_text = think_block.strip() or THINK_PLACEHOLDER_IF_EMPTY
    updated_summary = ""
    if enable_summary:
        summary_block = extract_tag_block(response_text, "SUMMARY")
        if summary_block:
            updated_summary = summary_block.strip()
    return {
        "thinking_text": thinking_text,
        "action_string": action_string,
        "parsed_action": parse_action_string(
            action_string,
            env_type=env_type,
            executor_type=executor_type,
        ),
        "updated_summary": updated_summary,
    }
