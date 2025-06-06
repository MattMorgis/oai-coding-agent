import os
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, cast
from unittest.mock import Mock

import pytest
from agents import Agent as SDKAgent
from agents import RunItemStreamEvent, Runner
from agents.items import (  # type: ignore[attr-defined]
    MessageOutputItem,
    ReasoningItem,
    ResponseFunctionToolCall,
    ToolCallItem,
)
from agents.mcp import MCPServerStdioParams

import oai_coding_agent.mcp.mcp_servers as mcp_servers_module
from oai_coding_agent.agent.agent import Agent
from oai_coding_agent.agent.events import (
    MessageOutputEvent,
    ReasoningEvent,
    ToolCallEvent,
)
from oai_coding_agent.mcp.mcp_servers import (
    ALLOWED_CLI_COMMANDS,
    ALLOWED_CLI_FLAGS,
    QuietMCPServerStdio,
)
from oai_coding_agent.runtime_config import ModeChoice, ModelChoice, RuntimeConfig


def test_allowed_cli_vars() -> None:
    # Ensure allowed CLI commands and flags are defined as lists and contain expected values
    assert isinstance(ALLOWED_CLI_COMMANDS, list)
    assert "grep" in ALLOWED_CLI_COMMANDS
    assert isinstance(ALLOWED_CLI_FLAGS, list)
    assert "all" in ALLOWED_CLI_FLAGS


def test_quiet_mcp_server_stdio_create_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch stdio_client to capture params and errlog
    captured = {}

    def fake_stdio_client(params_arg: Any, errlog: Any | None = None) -> str:
        captured["params"] = params_arg
        captured["errlog"] = errlog
        return "fake_streams"

    monkeypatch.setattr(mcp_servers_module, "stdio_client", fake_stdio_client)

    params = {"command": "test", "args": []}
    quiet = QuietMCPServerStdio(
        name="test",
        params=cast(MCPServerStdioParams, params),
        client_session_timeout_seconds=10,
        cache_tools_list=False,
    )
    streams = quiet.create_streams()
    assert streams == "fake_streams"
    # Params should be passed through unchanged
    # Params should contain the correct command and args attributes
    param_obj = captured["params"]
    assert hasattr(param_obj, "command") and param_obj.command == "test"
    assert hasattr(param_obj, "args") and param_obj.args == []
    # errlog should be a file to os.devnull
    errlog = captured["errlog"]
    assert hasattr(errlog, "write")
    assert getattr(errlog, "name", None) == os.devnull
    # Close the errlog file
    errlog.close()


@pytest.mark.asyncio
async def test_run_streams_and_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create mock SDK events with proper structure
    # Mock tool call event
    tool_call_item = Mock(spec=ToolCallItem)
    tool_call_raw = ResponseFunctionToolCall(
        name="test_tool",
        arguments='{"arg": "value"}',
        call_id="test_call_id",
        type="function_call",
    )
    tool_call_item.raw_item = tool_call_raw
    tool_call_event = Mock(spec=RunItemStreamEvent)
    tool_call_event.item = tool_call_item

    # Mock reasoning event
    reasoning_item = Mock(spec=ReasoningItem)
    reasoning_raw = Mock()
    reasoning_raw.summary = [Mock(text="Test reasoning")]
    reasoning_item.raw_item = reasoning_raw
    reasoning_event = Mock(spec=RunItemStreamEvent)
    reasoning_event.item = reasoning_item

    # Mock message output event
    message_item = Mock(spec=MessageOutputItem)
    message_raw = Mock()
    message_raw.content = [Mock(text="Test message")]
    message_item.raw_item = message_raw
    message_event = Mock(spec=RunItemStreamEvent)
    message_event.item = message_item

    events = [tool_call_event, reasoning_event, message_event]

    class FakeResult:
        def __init__(self, evts: list[Any]) -> None:
            self._events = evts
            # expose last_response_id so Agent.run can store it
            self.last_response_id: Optional[str] = None

        def stream_events(self) -> AsyncGenerator[Any, None]:
            async def gen() -> AsyncGenerator[Any, None]:
                for e in self._events:
                    yield e

            return gen()

    fake_result = FakeResult(events)

    # Monkeypatch Runner.run_streamed
    monkeypatch.setattr(
        Runner,
        "run_streamed",
        lambda *args, **kwargs: fake_result,
    )
    # Initialize agent and set dummy SDK agent
    config = RuntimeConfig(
        openai_api_key="k",
        github_token="TOK",
        model=ModelChoice.codex_mini_latest,
        repo_path=Path("."),
        mode=ModeChoice.async_,
    )
    agent = Agent(config, max_turns=1)
    agent._sdk_agent = cast(SDKAgent, object())

    event_stream = await agent.run("input text")
    # Verify we can iterate the mapped events from the stream
    collected = []
    async for event in event_stream:
        collected.append(event)
    assert len(collected) == 3

    # Check that events were properly mapped
    assert isinstance(collected[0], ToolCallEvent)
    assert collected[0].name == "test_tool"
    assert collected[0].arguments == '{"arg": "value"}'

    assert isinstance(collected[1], ReasoningEvent)
    assert collected[1].text == "Test reasoning"

    assert isinstance(collected[2], MessageOutputEvent)
    assert collected[2].text == "Test message"
