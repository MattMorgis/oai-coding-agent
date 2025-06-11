from typing import List, Optional

from oai_coding_agent.agent import AgentProtocol
from oai_coding_agent.console.callbacks import ChatCallback, ChatEvent
from oai_coding_agent.console.ui_event_mapper import map_event_to_ui_message


class AgentDialog:
    """Dialog that handles agent interactions with event callbacks."""

    def __init__(self, agent: Optional[AgentProtocol] = None) -> None:
        self.agent = agent
        self.callbacks: List[ChatCallback] = []

    def add_callback(self, callback: ChatCallback) -> None:
        """Add a callback to the agent dialog."""
        self.callbacks.append(callback)

    def notify_callbacks(self, event: ChatEvent, message: str) -> None:
        """Notify all callbacks of an event."""
        for callback in self.callbacks:
            callback.on_event(event, message)

    async def process_message(self, message: str) -> str:
        """Process the user message and return a response with event notifications."""
        try:
            self.notify_callbacks(
                ChatEvent.START_PROCESSING, "Starting to process message"
            )

            if self.agent:
                # Use real agent
                event_stream, result = await self.agent.run(message)

                # Consume the event stream and convert to callback events
                async for event in event_stream:
                    ui_msg = map_event_to_ui_message(event)
                    if ui_msg:
                        # Convert UI message to callback event
                        if ui_msg["role"] == "tool":
                            self.notify_callbacks(ChatEvent.THINKING, ui_msg["content"])
                        elif ui_msg["role"] == "thought":
                            self.notify_callbacks(ChatEvent.THINKING, ui_msg["content"])
                        elif ui_msg["role"] == "assistant":
                            self.notify_callbacks(
                                ChatEvent.PROCESSING_COMPLETE, ui_msg["content"]
                            )

                return result.final_output or "Agent completed processing."
            else:
                return "The agent is not configured."

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.notify_callbacks(ChatEvent.ERROR, error_message)
            return (
                "I apologize, but I encountered an error while processing your message."
            )
