"""
Agent for streaming OAI agent interactions with a local codebase.
"""

__all__ = ["Agent", "AgentProtocol"]

import logging
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from agents import (
    Agent as SDKAgent,
)
from agents import (
    ModelSettings,
    Runner,
    gen_trace_id,
    trace,
)
from openai.types.shared.reasoning import Reasoning

from oai_coding_agent.agent.instruction_builder import build_instructions
from oai_coding_agent.agent.mcp_servers import start_mcp_servers
from oai_coding_agent.agent.mcp_tool_selector import get_filtered_function_tools
from oai_coding_agent.runtime_config import RuntimeConfig

from .events import (
    MessageOutputEvent,
    ReasoningEvent,
    ToolCallEvent,
    map_sdk_event_to_agent_event,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the interface for agents."""

    config: RuntimeConfig

    async def __aenter__(self) -> "AgentProtocol": ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    async def run(
        self,
        user_input: str,
    ) -> AsyncIterator[ToolCallEvent | ReasoningEvent | MessageOutputEvent]: ...


class Agent:
    """Agent that manages MCP servers, tracing, and SDK agent interactions."""

    def __init__(self, config: RuntimeConfig, max_turns: int = 100):
        self.config = config
        self.max_turns = max_turns
        # track the last response ID internally
        self._previous_response_id: Optional[str] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._sdk_agent: Optional[SDKAgent] = None

    async def __aenter__(self) -> "Agent":
        # Initialize exit stack for async contexts and callbacks
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Start MCP servers (filesystem, CLI, Git, GitHub) and register cleanup
        mcp_servers = await start_mcp_servers(
            self.config,
            self._exit_stack,
        )

        # Begin tracing
        trace_id = gen_trace_id()
        trace_ctx = trace(workflow_name="OAI Coding Agent", trace_id=trace_id)
        trace_ctx.__enter__()
        self._exit_stack.callback(trace_ctx.__exit__, None, None, None)

        # Build instructions and fetch filtered MCP function-tools
        dynamic_instructions = build_instructions(self.config)
        function_tools = await get_filtered_function_tools(mcp_servers, self.config)

        # Instantiate the SDK Agent with the filtered function-tools
        self._sdk_agent = SDKAgent(
            name="Coding Agent",
            instructions=dynamic_instructions,
            model=self.config.model.value,
            model_settings=ModelSettings(
                reasoning=Reasoning(summary="auto", effort="high"),
                parallel_tool_calls=True,
            ),
            tools=function_tools,
        )

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        user_input: str,
    ) -> AsyncIterator[ToolCallEvent | ReasoningEvent | MessageOutputEvent]:
        """
        Send one user message to the agent and return an async iterator of agent events.
        """
        if self._sdk_agent is None:
            raise RuntimeError("Agent not initialized. Use async with context manager.")

        result = Runner.run_streamed(
            self._sdk_agent,
            user_input,
            previous_response_id=self._previous_response_id,
            max_turns=self.max_turns,
        )

        # Automatically resume from the last_response_id set on previous runs
        async def _map_events() -> AsyncIterator[
            ToolCallEvent | ReasoningEvent | MessageOutputEvent
        ]:
            """Map SDK events to agent events, filtering out None values."""
            async for sdk_event in result.stream_events():
                # Store the last-response ID so subsequent calls continue the dialogue
                self._previous_response_id = result.last_response_id
                agent_event = map_sdk_event_to_agent_event(sdk_event)
                if agent_event is not None:
                    yield agent_event

        return _map_events()
