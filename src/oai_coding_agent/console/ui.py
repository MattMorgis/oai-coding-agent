"""
A Text User Interface (TUI) chat application built with the Textual framework.
This module implements a chat interface that allows users to interact with a chatbot
in a terminal-based environment. It features real-time message display, background
processing, and event logging.

Key components:
- ChatMessage: Custom widget for displaying formatted chat messages
- ChatInterface: Main application class handling the UI and agent interactions
- Background processing: Asynchronous message handling to keep UI responsive
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.styles import Style
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Header, Input, Label, Static
from textual.worker import Worker, WorkerState

from oai_coding_agent.agent import AgentProtocol
from oai_coding_agent.console.agent_dialog import AgentDialog
from oai_coding_agent.console.callbacks import FileLogCallback, TuiCallback
from oai_coding_agent.runtime_config import get_data_dir


class ChatMessage(Label):
    """
    A custom widget for displaying chat messages with formatting.

    Each message shows:
    - The sender's name in bold
    - A timestamp
    - The message content

    Args:
        sender (str): Name of the message sender
        message (str): Content of the message
    """

    def __init__(self, sender: str, message: str) -> None:
        # Format timestamp as HH:MM:SS
        current_time = datetime.now().strftime("%H:%M:%S")
        # Create rich text with markup for styling
        formatted_message = Text.from_markup(
            f"[bold]{sender}[/bold] ({current_time})\n{message}"
        )
        super().__init__(formatted_message)


class LiveMessage(Widget):
    """Updates a message in real-time."""

    message = reactive("")

    def render(self) -> str:
        return self.message

    def show(self, msg: str) -> None:
        self.message = msg
        self.remove_class("hidden")

    def hide(self) -> None:
        self.message = ""
        self.add_class("hidden")


class PromptInput(Input):
    """A widget for displaying a prompt."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Store prompt history under the XDG data directory
        history_dir = get_data_dir()
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = history_dir / "prompt_history"

    def on_mount(self) -> None:
        self.session = PromptSession(
            placeholder="Type your message here and press Enter...",
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
            complete_while_typing=True,
            complete_in_thread=True,
            style=Style.from_dict(
                {"prompt": "ansicyan bold", "auto-suggestion": "#888888"}
            ),
            erase_when_done=True,
        )

    async def on_key(self, event: KeyPressEvent) -> None:
        if event.key == "enter":
            self.session.append_to_history(self.value)
            on_input_submitted

        if event.key == "tab":
            if self.session.default_buffer.suggestion:
                self.session.default_buffer.insert_text(
                    self.session.default_buffer.suggestion.text
                )
            else:
                self.session.default_buffer.complete_next()
        elif event.key == "up":
            self.session.default_buffer.history_backward()
        elif event.key == "down":
            self.session.default_buffer.history_forward()


class ChatInterface(App[Any]):
    """
    Main Terminal UI application class implementing the chat interface.

    Features:
    - Real-time message display
    - Background message processing
    - Keyboard shortcuts
    - Event logging
    - Scrollable message history
    """

    # Path to CSS file for styling the interface
    CSS_PATH = Path(__file__).parent / "static" / "styles.css"

    # Define keyboard shortcuts
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        # Binding("escape", "quit", "Quit", show=True),
    ]

    def __init__(self, agent: AgentProtocol) -> None:
        """Initialize the chat interface with a AgentDialog instance."""
        super().__init__()
        self.agent = agent
        self.agent_dialog = AgentDialog(agent)
        self.title = "OAI Coding Agent"

    async def on_mount(self) -> None:
        """
        Set up callbacks after the app is mounted.

        Initializes:
        1. Agent async context (MCP servers, tracing, SDK agent)
        2. TUI callback for updating the interface
        3. File logging callback for event tracking


        Timing:
        - on_mount is called after all widgets are created and mounted
        - It's safe to query and reference widgets here
        - Perfect place for post-initialization setup

        Widget Access:
        - query_one("#message-container") works here because widgets exist
        - Would fail if called in __init__ (widgets don't exist yet)

        Lifecycle Order:
        - Textual App Lifecycle
            1. __init__()           # Initial setup
            2. compose()           # Create widgets
            3. on_mount()         # Post-mount setup
            4. on_ready()        # App ready for user interaction

        Best Practices:
        - Use __init__ for basic initialization
        - Use compose for widget creation
        - Use on_mount for:
            - Widget queries
            - Event handlers setup
            - Callback registration
            - Post-initialization configuration

        This method is crucial for proper initialization timing in Textual applications, ensuring all components are properly set up after the UI is ready.
        """
        # Initialize agent async context (starts MCP servers, tracing, SDK agent)
        await self.agent.__aenter__()

        message_container = self.query_one("#message-container", ScrollableContainer)
        live_message = self.query_one("#live-message", LiveMessage)

        # Add TUI callback to handle agent dialog events and update the interface
        tui_callback = TuiCallback(self, message_container, ChatMessage, live_message)
        self.agent_dialog.add_callback(tui_callback)

        # Set up file logging for events
        log_dir = get_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_callback = FileLogCallback(str(log_dir / "console.log"))
        self.agent_dialog.add_callback(file_callback)

    async def on_unmount(self) -> None:
        """Clean up agent async context when app is unmounted."""
        await self.agent.__aexit__(None, None, None)

    def compose(self) -> ComposeResult:
        """
        Create and arrange the UI widgets.

        Layout:
        - Header at the top
        - Chat container with:
            - Scrollable message history
            - Input field at the bottom
        """
        yield Header()
        with Container(id="chat-container"):
            with ScrollableContainer(id="message-container"):
                yield Static(
                    Panel(
                        f"[bold cyan]╭─ OAI CODING AGENT ─╮[/bold cyan]\n\n"
                        f"[dim]Current Directory:[/dim] [dim cyan]{self.agent.config.repo_path}[/dim cyan]\n"
                        f"[dim]Model:[/dim] [dim cyan]{self.agent.config.model.value}[/dim cyan]\n"
                        f"[dim]Mode:[/dim] [dim cyan]{self.agent.config.mode.value}[/dim cyan]",
                        expand=False,
                    )
                )
                yield ChatMessage("Agent", "Hello! How can I help you today?")
            with Container(id="live-container"):
                yield LiveMessage(id="live-message", classes="hidden")
            with Container(id="input-container"):
                # yield PromptInput(
                #     placeholder="Type something...",
                #     id="user-input",
                # )
                yield Input(
                    placeholder="Type your message here and press Enter...",
                    id="user-input",
                )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes.

        Args:
            event (Worker.StateChanged): The worker state change event

        Notes:
        1. on_worker_state_changed is a Textual event handler (main thread)
        2. Direct UI updates are safe in the main thread
        3. Worker state changes provide a reliable way to handle completion

        This pattern follows Textual's thread safety model:
        - Event handlers → Direct UI updates
        - Background operations → Use call_from_thread
        - Worker state changes → Already in main thread

        """
        message_container = self.query_one("#message-container")

        if event.worker.name == "agent_processing":  # Only handle our agent worker
            if event.worker.state == WorkerState.SUCCESS:
                # We're already in the main thread, so update UI directly
                result = event.worker.result or ""
                message_container.mount(ChatMessage("Agent", str(result)))
                message_container.scroll_end(animate=False)

    def process_message_in_background(self, user_input: str) -> Worker[Any]:
        """Process user messages in a background thread to keep UI responsive.

        Args:
            user_input (str): The message from the user

        Returns:
            Worker: A background worker processing the message
        """
        message_container = self.query_one("#message-container")
        message_container.mount(ChatMessage("You", user_input))
        message_container.scroll_end(animate=False)

        # Create worker for background processing (async)
        worker = self.run_worker(
            self.agent_dialog.process_message(user_input),
            name="agent_processing",  # Important for identifying our worker
        )
        return worker

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handle user input submission when Enter is pressed.

        Args:
            event (Input.Submitted): The input submission event
        """
        user_input = event.value
        if not user_input.strip():
            return

        # Clear input field after submission
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        # Process message asynchronously
        self.process_message_in_background(user_input)
