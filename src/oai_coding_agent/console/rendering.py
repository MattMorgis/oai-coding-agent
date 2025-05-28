import os
from typing import Any, Iterable

from rich.console import Console
from rich.markdown import Heading, Markdown

from .state import Message


# Classes to override the default Markdown renderer
class PlainHeading(Heading):  # type: ignore[misc]
    """Left-aligned, no panel."""

    def __rich_console__(self, console: Console, options: Any) -> Iterable[Any]:
        self.text.justify = "left"
        yield self.text


class PlainMarkdown(Markdown):  # type: ignore[misc]
    elements = Markdown.elements.copy()
    elements["heading_open"] = PlainHeading


# Apply override globally for Markdown
Markdown.elements["heading_open"] = PlainHeading


console = Console()


def clear_terminal() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def render_message(msg: Message) -> None:
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
