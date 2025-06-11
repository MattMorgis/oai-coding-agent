import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import MockAgent

import oai_coding_agent.console.console as console_module
import oai_coding_agent.console.rendering as rendering
from oai_coding_agent.runtime_config import (
    ModeChoice,
    ModelChoice,
    RuntimeConfig,
)


class DummyPromptSession:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def prompt(self, prompt_str: str) -> str:
        # Immediately exit on slash command
        return "/exit"


@pytest.fixture(autouse=True)
def setup_repl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Console:
    # Redirect console output to recorder and disable clear
    recorder = Console(record=True, width=80)
    monkeypatch.setattr(rendering, "console", recorder)
    monkeypatch.setattr(console_module, "console", recorder)
    monkeypatch.setattr(rendering, "clear_terminal", lambda: None)
    monkeypatch.setattr(console_module, "clear_terminal", lambda: None)

    # Force history path into tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    return recorder


@pytest.mark.asyncio
async def test_repl_console_exits_on_exit_and_prints_header(
    setup_repl: Console, tmp_path: Path
) -> None:
    recorder = setup_repl
    config = RuntimeConfig(
        openai_api_key="APIKEY",
        github_token="GHTOKEN",
        model=ModelChoice.codex_mini_latest,
        repo_path=tmp_path,
        mode=ModeChoice.default,
        prompt="Test prompt",
    )

    # Create agent and console directly
    agent = MockAgent(config)
    console = console_module.HeadlessConsole(agent)
    await console.run()

    output = recorder.export_text()
    # HeadlessConsole prints the prompt, not a header
    assert "Prompt: Test prompt" in output
