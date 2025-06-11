import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer

from oai_coding_agent.agent import Agent, AgentProtocol
from oai_coding_agent.auth.github_browser_auth import authenticate_github_browser
from oai_coding_agent.auth.token_storage import delete_github_token, get_github_token
from oai_coding_agent.console.console import Console, HeadlessConsole
from oai_coding_agent.console.ui import ChatInterface
from oai_coding_agent.logger import setup_logging
from oai_coding_agent.preflight import PreflightCheckError, run_preflight_checks
from oai_coding_agent.runtime_config import (
    GITHUB_TOKEN,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    ModeChoice,
    ModelChoice,
    RuntimeConfig,
    load_envs,
)

# Global factory functions - set by create_app()
_agent_factory: Optional[Callable[[RuntimeConfig], AgentProtocol]] = None
_console_factory: Optional[Callable[[AgentProtocol], Console | ChatInterface]] = None


def default_agent_factory(config: RuntimeConfig) -> AgentProtocol:
    """Default factory for creating Agent instances."""
    return Agent(config)


def default_console_factory(agent: AgentProtocol) -> Console | ChatInterface:
    """Default factory for creating Console instances."""
    if agent.config.prompt:
        return HeadlessConsole(agent)
    else:
        return ChatInterface(agent)


def create_github_app() -> typer.Typer:
    # Create github subcommand group
    github_app = typer.Typer(rich_markup_mode=None)
    github_app.command("auth")(github_auth)
    github_app.command("logout")(github_logout)
    return github_app


def github_auth() -> None:
    """Authenticate with GitHub using browser-based flow."""
    typer.echo("🔐 Starting GitHub authentication...")

    # Check if already authenticated
    existing_token = get_github_token()
    if existing_token:
        typer.echo("⚠️  You already have a stored GitHub token.")
        if not typer.confirm("Do you want to re-authenticate?"):
            typer.echo("Authentication cancelled.")
            return

    # Perform authentication
    token = authenticate_github_browser()
    if token:
        typer.echo("\n✅ Authentication successful!")
        typer.echo("You can now use the agent with full GitHub integration.")
    else:
        typer.echo("\n❌ Authentication failed.")
        typer.echo("Please try again or set GITHUB_TOKEN manually.")
        raise typer.Exit(code=1)


def github_logout() -> None:
    """Remove stored GitHub authentication token."""
    if not get_github_token():
        typer.echo("No stored GitHub token found.")
        return

    if typer.confirm("Are you sure you want to remove your GitHub token?"):
        if delete_github_token():
            typer.echo("✅ Successfully logged out from GitHub.")
            typer.echo("You'll need to authenticate again to use GitHub features.")
        else:
            typer.echo("❌ Failed to remove token.")
            raise typer.Exit(code=1)
    else:
        typer.echo("Logout cancelled.")


def main(
    ctx: typer.Context,
    openai_api_key: Annotated[
        str,
        typer.Option(envvar=OPENAI_API_KEY_ENV, help="OpenAI API key"),
    ],
    github_token: Annotated[
        Optional[str],
        typer.Option(
            envvar=GITHUB_TOKEN,
            help="GitHub Token",
        ),
    ] = None,
    model: Annotated[
        ModelChoice, typer.Option("--model", "-m", help="OpenAI model to use")
    ] = ModelChoice.codex_mini_latest,
    mode: Annotated[
        ModeChoice,
        typer.Option("--mode", help="Agent mode: default, async, or plan"),
    ] = ModeChoice.default,
    repo_path: Path = typer.Option(
        Path.cwd(),
        "--repo-path",
        help=(
            "Path to the repository. This path (and its subdirectories) "
            "are the only files the agent has permission to access"
        ),
    ),
    openai_base_url: Annotated[
        Optional[str],
        typer.Option(envvar=OPENAI_BASE_URL_ENV, help="OpenAI base URL"),
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt text for non-interactive async mode; use '-' to read from stdin",
        ),
    ] = None,
    atlassian: Annotated[
        bool,
        typer.Option(
            "--atlassian",
            help="Enable Atlassian MCP server (only available in plan mode)",
        ),
    ] = False,
) -> None:
    """OAI CODING AGENT - starts an interactive session"""
    # If no subcommand, run default action
    if ctx.invoked_subcommand is None:
        logger = logging.getLogger(__name__)
        # Run preflight checks and get git info
        try:
            github_repo, branch_name = run_preflight_checks(repo_path)
        except PreflightCheckError as e:
            for error in e.errors:
                typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(code=1)

        # Read prompt text if provided
        prompt_text = None
        if prompt:
            if prompt == "-":
                prompt_text = sys.stdin.read()
            else:
                prompt_text = prompt

        # Handle GitHub authentication
        if not github_token and mode == ModeChoice.default and not prompt:
            # Only prompt for browser auth in interactive Default mode
            typer.echo("\n⚠️  No GitHub Personal Access Token found.")
            typer.echo("Would you like to authenticate with GitHub using your browser?")
            if typer.confirm("Authenticate now?"):
                token = authenticate_github_browser()
                if token:
                    github_token = token
                else:
                    typer.echo("\n❌ Browser authentication failed.")
                    typer.echo("Please set GITHUB_TOKEN manually.")
                    raise typer.Exit(code=1)
            else:
                typer.echo("\nAlternatively, you can:")
                typer.echo(
                    "  • Set environment variable: export GITHUB_TOKEN=your_token"
                )
                typer.echo(
                    "  • Use command line option: --github-personal-access-token your_token"
                )
                typer.echo("  • The agent will continue without GitHub integration")

        # Note: github_token can be None - the agent will handle this gracefully

        cfg = RuntimeConfig(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            github_token=github_token,
            model=model,
            repo_path=repo_path,
            mode=ModeChoice.async_ if prompt else mode,  # run in async mode if prompt
            github_repo=github_repo,
            branch_name=branch_name,
            prompt=prompt_text,
            atlassian=atlassian,
        )

        if not prompt:
            logger.info(
                f"Starting chat with model {cfg.model.value} on repo {cfg.repo_path}"
            )
        else:
            logger.info(f"Running prompt in headless (async): {cfg.prompt}")

        try:
            factory = _agent_factory or default_agent_factory
            console_fact = _console_factory or default_console_factory
            agent = factory(cfg)
            console = console_fact(agent)
            if isinstance(console, HeadlessConsole):
                asyncio.run(console.run())
            elif isinstance(console, ChatInterface):
                console.run()
        except KeyboardInterrupt:
            print("\nExiting...")


def create_app(
    agent_factory: Optional[Callable[[RuntimeConfig], AgentProtocol]] = None,
    console_factory: Optional[
        Callable[[AgentProtocol], Console | ChatInterface]
    ] = None,
) -> typer.Typer:
    """
    Create and configure the Typer application.

    Args:
        agent_factory: Factory function to create Agent instances
        console_factory: Factory function to create Console instances

    Returns:
        Typer application
    """
    setup_logging()

    # Load API keys and related settings from .env if not already set in the environment
    load_envs()

    if agent_factory is None:
        agent_factory = default_agent_factory
    if console_factory is None:
        console_factory = default_console_factory

    # Set global factory functions
    global _agent_factory, _console_factory
    _agent_factory = agent_factory
    _console_factory = console_factory

    app = typer.Typer(rich_markup_mode=None)
    github_app = create_github_app()
    app.add_typer(github_app, name="github")

    app.callback(invoke_without_command=True)(main)

    return app


# Create default app instance for backward compatibility
app = create_app()


if __name__ == "__main__":
    app()
