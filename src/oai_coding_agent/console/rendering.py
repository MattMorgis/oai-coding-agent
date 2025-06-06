import os
from typing import Optional

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.text import Text

from .state import UIMessage


# Classes to override the default Markdown renderer
class PlainHeading(Heading):
    """Left-aligned, no panel."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        self.text.justify = "left"
        yield self.text


class PlainMarkdown(Markdown):
    elements = Markdown.elements.copy()
    elements["heading_open"] = PlainHeading


# Apply override globally for Markdown
Markdown.elements["heading_open"] = PlainHeading


console = Console()

# Global variable to track interrupt indicator
_interrupt_live: Optional[Live] = None


def clear_terminal() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def show_interrupt_indicator() -> None:
    """Show the 'ESC to interrupt' indicator."""
    global _interrupt_live
    if _interrupt_live is None:
        indicator_text = Text("(esc to interrupt)", style="dim")
        _interrupt_live = Live(indicator_text, console=console, refresh_per_second=1)
        _interrupt_live.start()


def hide_interrupt_indicator() -> None:
    """Hide the 'ESC to interrupt' indicator."""
    global _interrupt_live
    if _interrupt_live is not None:
        _interrupt_live.stop()
        _interrupt_live = None


def render_message(msg: UIMessage) -> None:
    """Render a single message via Rich."""
    role = msg.get("role")
    content = msg.get("content", "")
    if role == "user":
        console.print(f"[bold blue]You:[/bold blue] {content}")
    elif role == "assistant":
        console.print("[bold cyan]oai:[/bold cyan]", end=" ")
        md = Markdown(content, code_theme="nord", hyperlinks=True)
        console.print(md)
    elif role == "system":
        console.print(f"[dim yellow]System:[/dim yellow] [yellow]{content}[/yellow]")
    elif role == "thought":
        console.print(f"[italic dim]{content}[/italic dim]")
    elif role == "tool":
        console.print(f"[dim green]Tool: {content}[/dim green]")

    console.print()
