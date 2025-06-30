# ruff: noqa: I001
from typing import Optional
import webbrowser

import pyperclip
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt

from oai_coding_agent.auth.github_browser_auth import poll_for_token, start_device_flow
from oai_coding_agent.auth.token_storage import (
    delete_token as delete_github_token,
    get_token as get_github_token,
    save_token as save_github_token,
)


class GitHubConsole:
    def __init__(self) -> None:
        pass

    def _copy_to_clipboard(self, text: str) -> bool:
        """Try to copy text to clipboard. Returns True if successful."""
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    def authenticate(self) -> Optional[str]:
        """Core GitHub authentication flow."""
        print("\n[bold]Starting GitHub login[/bold]")

        # Start device flow
        device_flow = start_device_flow()
        if not device_flow:
            print("[red]✗ Failed to start GitHub login[/red]")
            return None

        # Display the code
        print(f"\nYour authentication code: [bold]`{device_flow.user_code}`[/bold]")

        # Try to copy to clipboard
        if self._copy_to_clipboard(device_flow.user_code):
            print("[dim green]✓ Code copied to clipboard[/dim green]")
        else:
            print("[dim]Copy this code - GitHub will ask for it[/dim]")

        # Prompt to open browser
        print(
            f"\nPress [bold]Enter[/bold] to open {device_flow.verification_uri} in your browser",
            end="",
        )
        Prompt.ask()

        # Open browser
        try:
            webbrowser.open(device_flow.verification_uri)
            print("[dim green]✓ Browser opened[/dim green]")
        except Exception:
            print(f"[dim]Please visit: {device_flow.verification_uri}[/dim]")

        print("\nNext steps:")
        print("  1. Log in to GitHub")
        print("  2. Enter the code shown above")
        print("  3. Authorize the application")
        print()

        # Poll for completion with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Waiting for authentication...", total=None)

            token = poll_for_token(
                device_flow.device_code,
                device_flow.interval,
                timeout=300,
            )

            progress.stop()

        if token:
            print("[dim green]✓ Successfully logged in to GitHub![/dim green]")
            save_github_token("GITHUB_TOKEN", token)

            return token
        else:
            print("[red]✗ Login failed or timed out[/red]")
            return None

        print()

    def prompt_auth(self) -> Optional[str]:
        """Prompt user to authenticate if no token is found."""
        token = get_github_token("GITHUB_TOKEN")
        if token:
            return token

        print()
        print("[bold]GitHub integration available[/bold]")
        print()
        print(
            "[bold dim]Quick two second setup:[/bold dim] [dim]We'll open your browser to connect with GitHub[/dim]"
        )
        print("[dim]This enables creating PRs, issues and more[/dim]")
        print()

        if Confirm.ask("\nWould you like to log in to GitHub?"):
            return self.authenticate()
        else:
            print("\n[dim]Continuing without GitHub integration[/dim]")
            print(
                "[dim]You can log in later by running:[/dim] [dim bold]oai github login[/dim bold]"
            )
            print()
            return None

    def check_or_authenticate(self) -> Optional[str]:
        """Check for existing token or authenticate. Used by the auth subcommand."""
        token = get_github_token("GITHUB_TOKEN")
        if token:
            if not Confirm.ask("\nWould you like to log in to GitHub?"):
                print("[green]Using existing GitHub login.[/green]")
                return token

        return self.authenticate()

    def logout(self) -> bool:
        """Log out from GitHub by removing stored token."""
        if not get_github_token("GITHUB_TOKEN"):
            print("No stored GitHub token found.")
            return True

        if Confirm.ask("\nAre you sure you want to remove your GitHub token?"):
            if delete_github_token("GITHUB_TOKEN"):
                print("[dim green]✓ Successfully logged out from GitHub.[/dim green]")
                print("You'll need to log in again to use GitHub features.")
                return True
            else:
                print("[red]✗ Failed to remove token.[/red]")
                return False
        else:
            print("Logout cancelled.")
            return True
