import time
from typing import Any, Dict, Optional

import pytest
import requests

import oai_coding_agent.auth.github_browser_auth as gba


class DummyResponse:
    """Minimal dummy response for requests.post stubbing."""

    def __init__(self, status_code: int, json_data: Dict[str, Any]):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> Dict[str, Any]:
        return self._json_data


DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"


def test_start_device_flow_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: stub device flow response
    device_data = {
        "device_code": "DEV_CODE",
        "user_code": "USER_CODE_123",
        "verification_uri": "https://github.com/verify",
        "interval": 5,
        "expires_in": 900,
    }

    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        if url == DEVICE_CODE_URL:
            # Verify request payload
            assert data and data.get("client_id") == gba.GITHUB_APP_CLIENT_ID
            assert data and "repo" in data.get("scope", "")
            return DummyResponse(200, device_data)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(requests, "post", fake_post)

    # Act
    result = gba.start_device_flow()

    # Assert
    assert result is not None
    assert result.device_code == "DEV_CODE"
    assert result.user_code == "USER_CODE_123"
    assert result.verification_uri == "https://github.com/verify"
    assert result.interval == 5
    assert result.expires_in == 900


def test_start_device_flow_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: stub failed response
    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        return DummyResponse(400, {"error": "invalid_request"})

    monkeypatch.setattr(requests, "post", fake_post)

    # Act
    result = gba.start_device_flow()

    # Assert
    assert result is None


def test_poll_for_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: stub successful token response
    token_data = {"access_token": "TOKEN_ABC"}

    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        if url == TOKEN_URL:
            assert data and data.get("device_code") == "DEV_CODE"
            return DummyResponse(200, token_data)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", lambda: 0)

    # Act
    token = gba.poll_for_token("DEV_CODE", interval=1, timeout=10)

    # Assert
    assert token == "TOKEN_ABC"


def test_poll_for_token_authorization_pending_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: stub pending then success
    token_sequence = [
        {"error": "authorization_pending"},
        {"access_token": "FINAL_TOKEN"},
    ]
    call: Dict[str, int] = {"count": 0}

    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        if url == TOKEN_URL:
            resp = token_sequence[call["count"]]
            call["count"] += 1
            return DummyResponse(200, resp)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", lambda: 0)

    # Act
    token = gba.poll_for_token("DEV_CODE", interval=1, timeout=10)

    # Assert
    assert token == "FINAL_TOKEN"
    assert call["count"] == 2


def test_poll_for_token_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: simulate timeout
    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        return DummyResponse(200, {"error": "authorization_pending"})

    # Simulate time progression
    times = [0, 301]

    def fake_time() -> float:
        return times.pop(0)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", fake_time)

    # Act
    token = gba.poll_for_token("DEV_CODE", interval=1, timeout=300)

    # Assert
    assert token is None


def test_poll_for_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: simulate error response
    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        return DummyResponse(
            200, {"error": "access_denied", "error_description": "Denied"}
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", lambda: 0)

    # Act
    token = gba.poll_for_token("DEV_CODE", interval=1, timeout=10)

    # Assert
    assert token is None


def test_poll_for_token_with_progress_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: test progress callback
    callback_calls: list[float] = []

    def progress_callback(elapsed: float) -> None:
        callback_calls.append(elapsed)

    token_data = {"access_token": "TOKEN_WITH_PROGRESS"}

    def fake_post(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> DummyResponse:
        return DummyResponse(200, token_data)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(time, "time", lambda: 0)

    # Act
    token = gba.poll_for_token(
        "DEV_CODE", interval=1, timeout=10, progress_callback=progress_callback
    )

    # Assert
    assert token == "TOKEN_WITH_PROGRESS"
    assert len(callback_calls) == 1
    assert callback_calls[0] == 0
