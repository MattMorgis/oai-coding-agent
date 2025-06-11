import pytest
from rich.console import Console

import oai_coding_agent.console.rendering as rendering
from oai_coding_agent.console.rendering import UIMessage


@pytest.fixture(autouse=True)
def record_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Replace rendering.console with a record-capable Console."""
    recorder = Console(record=True, width=80)
    monkeypatch.setattr(rendering, "console", recorder)
    # Prevent actual clear
    monkeypatch.setattr(rendering, "clear_terminal", lambda: None)
    return recorder


def test_clear_terminal_callable() -> None:
    # Just ensure clear_terminal exists and is callable
    assert callable(rendering.clear_terminal)


def test_render_message_user(record_console: Console) -> None:
    msg: UIMessage = {"role": "user", "content": "hello user"}
    rendering.render_message(msg)
    out = record_console.export_text()
    assert "You: hello user" in out


def test_render_message_assistant(record_console: Console) -> None:
    msg: UIMessage = {"role": "assistant", "content": "**bold** and `code`"}
    rendering.render_message(msg)
    out = record_console.export_text()
    assert "oai:" in out
    assert "bold" in out
    assert "code" in out


def test_render_message_system(record_console: Console) -> None:
    msg: UIMessage = {"role": "system", "content": "system info"}
    rendering.render_message(msg)
    out = record_console.export_text()
    assert "System:" in out
    assert "system info" in out


def test_render_message_thought(record_console: Console) -> None:
    msg: UIMessage = {"role": "thought", "content": "thinking..."}
    rendering.render_message(msg)
    out = record_console.export_text()
    assert "thinking..." in out


def test_render_message_tool(record_console: Console) -> None:
    msg: UIMessage = {"role": "tool", "content": "tool output"}
    rendering.render_message(msg)
    out = record_console.export_text()
    assert "Tool: tool output" in out
