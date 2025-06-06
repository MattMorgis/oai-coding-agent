import asyncio
from typing import Any, Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.panel import Panel

from oai_coding_agent.runtime_config import get_data_dir

from ..agent import AgentProtocol
from ..interrupt_handler import InterruptedError
from .key_bindings import get_key_bindings
from .rendering import clear_terminal, console, render_message
from .slash_commands import handle_slash_command, register_slash_commands
from .state import UIState
from .ui_event_mapper import map_event_to_ui_message


class Console(Protocol):
    """Common interface for console interactions."""

    agent: AgentProtocol

    async def run(self) -> None:
        pass


class HeadlessConsole:
    """Console that runs headless (single prompt) mode."""

    def __init__(self, agent: AgentProtocol) -> None:
        self.agent = agent

    async def run(self) -> None:
        """
        Execute one prompt in async 'headless' mode and render streamed output.
        """
        if not self.agent.config.prompt:
            raise ValueError("Prompt is required for headless mode")

        console.print(f"[bold cyan]Prompt:[/bold cyan] {self.agent.config.prompt}")
        async with self.agent:
            with self.agent.interrupt_handler:
                try:
                    event_stream = await self.agent.run(self.agent.config.prompt)
                    async for event in event_stream:
                        ui_msg = map_event_to_ui_message(event)
                        if ui_msg:
                            render_message(ui_msg)
                except InterruptedError:
                    console.print("\n[yellow]Response interrupted by user.[/yellow]")
                except KeyboardInterrupt:
                    console.print("\n[red]Exiting...[/red]")


class ReplConsole:
    """Console that runs interactive REPL mode."""

    def __init__(self, agent: AgentProtocol) -> None:
        self.agent = agent

    async def _process_agent_response(self, event_stream: Any) -> None:
        """Process and render agent response events."""
        async for event in event_stream:
            ui_msg = map_event_to_ui_message(event)
            render_message(ui_msg)

    async def run(self) -> None:
        """Interactive REPL loop for the console interface."""
        state = UIState()
        clear_terminal()

        register_slash_commands(state)

        console.print(
            Panel(
                f"[bold cyan]╭─ OAI CODING AGENT ─╮[/bold cyan]\n\n"
                f"[dim]Current Directory:[/dim] [dim cyan]{self.agent.config.repo_path}[/dim cyan]\n"
                f"[dim]Model:[/dim] [dim cyan]{self.agent.config.model.value}[/dim cyan]\n"
                f"[dim]Mode:[/dim] [dim cyan]{self.agent.config.mode.value}[/dim cyan]",
                expand=False,
            )
        )

        kb = get_key_bindings()

        # Store prompt history under the XDG data directory
        history_dir = get_data_dir()
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "prompt_history"

        prompt_session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
            complete_while_typing=True,
            completer=WordCompleter([f"/{c}" for c in state.slash_commands]),
            complete_in_thread=True,
            key_bindings=kb,
            style=Style.from_dict(
                {"prompt": "ansicyan bold", "auto-suggestion": "#888888"}
            ),
            erase_when_done=True,
        )

        async with self.agent:
            continue_loop = True
            while continue_loop:
                try:
                    user_input = await asyncio.to_thread(
                        lambda: prompt_session.prompt("› ")
                    )
                    if not user_input.strip():
                        continue

                    if user_input.strip().lower() in ["exit", "quit", "/exit", "/quit"]:
                        continue_loop = state.slash_commands["exit"]("")
                        continue

                    if user_input.startswith("/"):
                        continue_loop = handle_slash_command(state, user_input)
                        continue

                    console.print(f"[dim]› {user_input}[/dim]\n")

                    # Install interrupt handler and run the agent
                    with self.agent.interrupt_handler:
                        try:
                            # Get the event stream first to ensure we have an async iterator
                            event_stream = await self.agent.run(user_input)
                            # Create task for cancellable execution
                            async with self.agent.interrupt_handler.cancellable_task(
                                self._process_agent_response(event_stream)
                            ) as task:
                                await task
                        except InterruptedError:
                            # Handle interruption gracefully
                            console.print(
                                "\n[yellow]Response interrupted by user.[/yellow]"
                            )

                            # Prompt user for how to proceed
                            proceed_prompt = await asyncio.to_thread(
                                lambda: prompt_session.prompt(
                                    "How would you like to proceed? (continue with new instructions or press Enter to skip): "
                                )
                            )

                            if proceed_prompt.strip():
                                # Continue with new instructions
                                user_input = proceed_prompt
                                console.print(f"[dim]› {user_input}[/dim]\n")

                                # Reset interrupt state and run with new input
                                self.agent.interrupt_handler.reset()
                                event_stream = await self.agent.run(user_input)
                                async for event in event_stream:
                                    ui_msg = map_event_to_ui_message(event)
                                    render_message(ui_msg)
                            else:
                                console.print(
                                    "[dim]Skipping interrupted response.[/dim]\n"
                                )
                        except KeyboardInterrupt:
                            # Second interrupt means exit
                            console.print("\n[red]Exiting...[/red]")
                            continue_loop = False

                except (KeyboardInterrupt, EOFError):
                    continue_loop = False
