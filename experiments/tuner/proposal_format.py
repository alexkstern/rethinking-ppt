"""Custom ``format_proposal`` callables for ``lt.Tuner``.

``make_log_and_format`` appends the full response to a JSONL transcript and
returns a context entry with the THINKING block dropped by default.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable


def make_log_and_format(
    transcript_path: str | Path,
    *,
    strip_thinking: bool = True,
    strip_response: bool = False,
) -> Callable[[dict], str]:
    transcript_path = Path(transcript_path)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(response: dict) -> str:
        with transcript_path.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "reasoning": response.get("reasoning"),
                "content": response.get("content"),
                "tool_args": response.get("tool_args"),
            }) + "\n")
        parts: list[str] = []
        if not strip_thinking and response.get("reasoning"):
            parts.append(f"THINKING:\n{response['reasoning']}")
        # strip_response leaves only the PROPOSED config; with strip_thinking
        # too, context grows by just the config + its RESULT per trial.
        if not strip_response and response.get("content"):
            parts.append(f"RESPONSE:\n{response['content']}")
        parts.append(f"PROPOSED: {response.get('tool_args', {})}")
        return "\n".join(parts)

    return fmt
