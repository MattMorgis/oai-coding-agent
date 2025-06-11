from pathlib import Path
from typing import Any, Callable, Optional

import pytest
from conftest import MockAgent, MockConsole
from typer.testing import CliRunner

import oai_coding_agent.cli as cli_module
from oai_coding_agent.agent import AgentProtocol
from oai_coding_agent.cli import create_app
from oai_coding_agent.console.console import Console
from oai_coding_agent.preflight import PreflightCheckError
from oai_coding_agent.runtime_config import ModeChoice, ModelChoice, RuntimeConfig


@pytest.fixture
def mock_agent_factory() -> Any:
    """Create a factory that returns a MockAgent."""
    created_agent: Optional[MockAgent] = None

    def factory(config: RuntimeConfig) -> AgentProtocol:
        nonlocal created_agent
        created_agent = MockAgent(config)
        return created_agent

    class FactoryResult:
        @property
        def agent(self) -> Optional[MockAgent]:
            return created_agent

        @property
        def factory(self) -> Callable[[RuntimeConfig], AgentProtocol]:
            return factory

    return FactoryResult()


@pytest.fixture
def mock_console_factory() -> Any:
    """Create a factory that returns a MockConsole."""
    created_console: Optional[MockConsole] = None

    def factory(agent: AgentProtocol) -> Console:
        nonlocal created_console
        created_console = MockConsole(agent)
        return created_console

    class FactoryResult:
        @property
        def console(self) -> Optional[MockConsole]:
            return created_console

        @property
        def factory(self) -> Callable[[AgentProtocol], Console]:
            return factory

    return FactoryResult()


@pytest.fixture(autouse=True)
def stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub preflight checks for CLI tests to not block execution
    monkeypatch.setattr(
        cli_module, "run_preflight_checks", lambda repo_path: (None, None)
    )


def test_cli_invokes_console_with_explicit_flags(
    mock_agent_factory: Any,
    mock_console_factory: Any,
    tmp_path: Path,
) -> None:
    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--openai-api-key",
            "TESTKEY",
            "--openai-base-url",
            "https://api.custom",
            "--github-token",
            "GHKEY",
            "--model",
            "o3",
            "--repo-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert mock_agent_factory.agent is not None
    assert mock_console_factory.console is not None

    agent = mock_agent_factory.agent
    assert agent.config.repo_path == tmp_path
    assert agent.config.model == ModelChoice.o3
    assert agent.config.openai_api_key == "TESTKEY"
    assert agent.config.openai_base_url == "https://api.custom"
    assert agent.config.mode == ModeChoice.default

    assert mock_console_factory.console.run_called


def test_cli_uses_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent_factory: Any,
    mock_console_factory: Any,
    tmp_path: Path,
) -> None:
    # Set environment variables for API keys and base URL
    monkeypatch.setenv("OPENAI_API_KEY", "ENVKEY")
    monkeypatch.setenv("OPENAI_BASE_URL", "ENVURL")
    monkeypatch.setenv("GITHUB_TOKEN", "ENVGH")

    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    result = runner.invoke(app, ["--repo-path", str(tmp_path)])
    assert result.exit_code == 0
    assert mock_agent_factory.agent is not None

    agent = mock_agent_factory.agent
    assert agent.config.repo_path == tmp_path
    assert agent.config.model == ModelChoice.codex_mini_latest
    assert agent.config.openai_api_key == "ENVKEY"
    assert agent.config.openai_base_url == "ENVURL"
    assert agent.config.mode == ModeChoice.default


def test_cli_uses_cwd_as_default_repo_path(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent_factory: Any,
    mock_console_factory: Any,
) -> None:
    # Set environment variables for API keys
    monkeypatch.setenv("OPENAI_API_KEY", "ENVKEY")
    monkeypatch.setenv("GITHUB_TOKEN", "ENVGH")

    # Get the actual current working directory
    expected_cwd = Path.cwd()

    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    result = runner.invoke(app, [])  # No --repo-path specified
    assert result.exit_code == 0
    assert mock_agent_factory.agent is not None

    agent = mock_agent_factory.agent
    assert agent.config.repo_path == expected_cwd
    assert agent.config.model == ModelChoice.codex_mini_latest
    assert agent.config.openai_api_key == "ENVKEY"
    assert agent.config.openai_base_url == "http://127.0.0.1:1234/v1"
    assert agent.config.mode == ModeChoice.default


def test_cli_prompt_invokes_headless_main(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent_factory: Any,
    mock_console_factory: Any,
    tmp_path: Path,
) -> None:
    # Set environment variables for API keys
    monkeypatch.setenv("OPENAI_API_KEY", "ENVKEY")
    monkeypatch.setenv("GITHUB_TOKEN", "ENVGH")

    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    result = runner.invoke(
        app, ["--repo-path", str(tmp_path), "--prompt", "Do awesome things"]
    )
    assert result.exit_code == 0
    assert mock_agent_factory.agent is not None

    agent = mock_agent_factory.agent
    assert agent.config.repo_path == tmp_path
    assert agent.config.model == ModelChoice.codex_mini_latest
    assert agent.config.openai_api_key == "ENVKEY"
    assert agent.config.openai_base_url == "http://127.0.0.1:1234/v1"
    assert agent.config.mode == ModeChoice.async_
    assert agent.config.prompt == "Do awesome things"


def test_cli_prompt_stdin_invokes_headless_main(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent_factory: Any,
    mock_console_factory: Any,
    tmp_path: Path,
) -> None:
    # Set environment variables for API keys
    monkeypatch.setenv("OPENAI_API_KEY", "ENVKEY")
    monkeypatch.setenv("GITHUB_TOKEN", "ENVGH")

    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    prompt_str = "Huge prompt content that exceeds usual limits"
    result = runner.invoke(
        app, ["--repo-path", str(tmp_path), "--prompt", "-"], input=prompt_str
    )
    assert result.exit_code == 0
    assert mock_agent_factory.agent is not None

    agent = mock_agent_factory.agent
    assert agent.config.repo_path == tmp_path
    assert agent.config.model == ModelChoice.codex_mini_latest
    assert agent.config.openai_api_key == "ENVKEY"
    assert agent.config.openai_base_url == "http://127.0.0.1:1234/v1"
    assert agent.config.mode == ModeChoice.async_
    assert agent.config.prompt == prompt_str


def test_cli_exits_on_preflight_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent_factory: Any,
    mock_console_factory: Any,
    tmp_path: Path,
) -> None:
    # Set environment variables for API keys
    monkeypatch.setenv("OPENAI_API_KEY", "ENVKEY")
    monkeypatch.setenv("GITHUB_TOKEN", "ENVGH")

    # Mock run_preflight_checks to raise PreflightCheckError
    def mock_preflight_failure(repo_path: Path) -> tuple[None, None]:
        raise PreflightCheckError(
            ["Node.js binary not found on PATH", "Docker daemon is not running"]
        )

    monkeypatch.setattr(cli_module, "run_preflight_checks", mock_preflight_failure)

    app = create_app(mock_agent_factory.factory, mock_console_factory.factory)
    runner = CliRunner()
    result = runner.invoke(app, ["--repo-path", str(tmp_path)])

    # Should exit with code 1
    assert result.exit_code == 1

    # Should print both error messages to stderr
    assert "Error: Node.js binary not found on PATH" in result.output
    assert "Error: Docker daemon is not running" in result.output

    # Agent and console should NOT be created
    assert mock_agent_factory.agent is None
    assert mock_console_factory.console is None
