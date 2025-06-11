"""
Map agent events to UI messages.
"""

import logging

from oai_coding_agent.agent.events import (
    AgentEvent,
    MessageOutputEvent,
    ReasoningEvent,
    ToolCallEvent,
)
from oai_coding_agent.console.rendering import UIMessage

logger = logging.getLogger(__name__)


def map_event_to_ui_message(event: AgentEvent) -> UIMessage | None:
    """Map an agent event to a UI message."""
    logger.debug("Internal event received: %r", event)

    match event:
        case ToolCallEvent(name=name, arguments=arguments):
            return UIMessage(role="tool", content=f"{name}({arguments})")
        case ReasoningEvent(text=text):
            return UIMessage(role="thought", content=f"💭 {text}")
        case MessageOutputEvent(text=text):
            return UIMessage(role="assistant", content=text)
        case _:
            return None
