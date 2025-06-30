from unittest.mock import patch

import pytest

from oai_coding_agent.auth.github_browser_auth import DeviceFlowData
from oai_coding_agent.console.github_console import GitHubConsole


@pytest.fixture
def github_console() -> GitHubConsole:
    return GitHubConsole()


def test_copy_to_clipboard_success(github_console: GitHubConsole) -> None:
    """Test successful clipboard copy."""
    with patch("oai_coding_agent.console.github_console.pyperclip.copy") as mock_copy:
        result = github_console._copy_to_clipboard("test_code")
        assert result is True
        mock_copy.assert_called_once_with("test_code")


def test_copy_to_clipboard_failure(github_console: GitHubConsole) -> None:
    """Test clipboard copy failure."""
    with patch(
        "oai_coding_agent.console.github_console.pyperclip.copy",
        side_effect=Exception("Clipboard error"),
    ):
        result = github_console._copy_to_clipboard("test_code")
        assert result is False


def test_prompt_auth_with_existing_token(github_console: GitHubConsole) -> None:
    """Test prompt_auth when token already exists."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token",
        return_value="existing_token",
    ):
        token = github_console.prompt_auth()
        assert token == "existing_token"


def test_prompt_auth_user_says_no(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test prompt_auth when user declines."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value=None
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=False
        ):
            token = github_console.prompt_auth()
            assert token is None
            captured = capsys.readouterr()
            assert "Continuing without GitHub integration" in captured.out


def test_prompt_auth_user_says_yes(github_console: GitHubConsole) -> None:
    """Test prompt_auth when user agrees to authenticate."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value=None
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=True
        ):
            with patch.object(
                github_console, "authenticate", return_value="new_token"
            ) as mock_auth:
                token = github_console.prompt_auth()
                assert token == "new_token"
                mock_auth.assert_called_once()


def test_authenticate_device_flow_failure(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test authenticate when device flow fails to start."""
    with patch(
        "oai_coding_agent.console.github_console.start_device_flow", return_value=None
    ):
        token = github_console.authenticate()
        assert token is None
        captured = capsys.readouterr()
        assert "Failed to start GitHub login" in captured.out


def test_authenticate_success(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test successful authentication flow."""
    device_flow = DeviceFlowData(
        device_code="DEV123",
        user_code="USER123",
        verification_uri="https://github.com/login/device",
        interval=5,
        expires_in=900,
    )

    with patch(
        "oai_coding_agent.console.github_console.start_device_flow",
        return_value=device_flow,
    ):
        with patch(
            "oai_coding_agent.console.github_console.Prompt.ask", return_value=""
        ):
            with patch("oai_coding_agent.console.github_console.webbrowser.open"):
                with patch(
                    "oai_coding_agent.console.github_console.poll_for_token",
                    return_value="token123",
                ):
                    with patch(
                        "oai_coding_agent.console.github_console.save_github_token",
                        return_value=True,
                    ):
                        with patch("oai_coding_agent.console.github_console.Progress"):
                            token = github_console.authenticate()
                            assert token == "token123"
                            captured = capsys.readouterr()
                            assert "USER123" in captured.out
                            assert "Successfully logged in to GitHub" in captured.out


def test_authenticate_timeout(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test authentication timeout."""
    device_flow = DeviceFlowData(
        device_code="DEV123",
        user_code="USER123",
        verification_uri="https://github.com/login/device",
        interval=5,
        expires_in=900,
    )

    with patch(
        "oai_coding_agent.console.github_console.start_device_flow",
        return_value=device_flow,
    ):
        with patch(
            "oai_coding_agent.console.github_console.Prompt.ask", return_value=""
        ):
            with patch("oai_coding_agent.console.github_console.webbrowser.open"):
                with patch(
                    "oai_coding_agent.console.github_console.poll_for_token",
                    return_value=None,
                ):
                    with patch("oai_coding_agent.console.github_console.Progress"):
                        token = github_console.authenticate()
                        assert token is None
                        captured = capsys.readouterr()
                        assert "Login failed or timed out" in captured.out


def test_check_or_authenticate_no_existing_token(github_console: GitHubConsole) -> None:
    """Test check_or_authenticate with no existing token."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value=None
    ):
        with patch.object(
            github_console, "authenticate", return_value="new_token"
        ) as mock_auth:
            token = github_console.check_or_authenticate()
            assert token == "new_token"
            mock_auth.assert_called_once()


def test_check_or_authenticate_existing_token_keep(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test check_or_authenticate when user keeps existing token."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token",
        return_value="existing_token",
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=False
        ):
            token = github_console.check_or_authenticate()
            assert token == "existing_token"
            captured = capsys.readouterr()
            assert "Using existing GitHub login" in captured.out


def test_check_or_authenticate_existing_token_reauth(
    github_console: GitHubConsole,
) -> None:
    """Test check_or_authenticate when user wants to re-authenticate."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token",
        return_value="existing_token",
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=True
        ):
            with patch.object(
                github_console, "authenticate", return_value="new_token"
            ) as mock_auth:
                token = github_console.check_or_authenticate()
                assert token == "new_token"
                mock_auth.assert_called_once()


def test_logout_no_token(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test logout when no token exists."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value=None
    ):
        result = github_console.logout()
        assert result is True
        captured = capsys.readouterr()
        assert "No stored GitHub token found" in captured.out


def test_logout_user_cancels(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test logout when user cancels."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value="token"
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=False
        ):
            result = github_console.logout()
            assert result is True
            captured = capsys.readouterr()
            assert "Logout cancelled" in captured.out


def test_logout_success(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test successful logout."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value="token"
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=True
        ):
            with patch(
                "oai_coding_agent.console.github_console.delete_github_token",
                return_value=True,
            ):
                result = github_console.logout()
                assert result is True
                captured = capsys.readouterr()
                assert "Successfully logged out from GitHub" in captured.out


def test_logout_failure(
    github_console: GitHubConsole, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test logout failure."""
    with patch(
        "oai_coding_agent.console.github_console.get_github_token", return_value="token"
    ):
        with patch(
            "oai_coding_agent.console.github_console.Confirm.ask", return_value=True
        ):
            with patch(
                "oai_coding_agent.console.github_console.delete_github_token",
                return_value=False,
            ):
                result = github_console.logout()
                assert result is False
                captured = capsys.readouterr()
                assert "Failed to remove token" in captured.out
