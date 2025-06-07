import pytest

import oai_coding_agent.auth.github_browser_auth as gba


class DummyResponse:
    """Minimal dummy response for requests.post stubbing."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> dict:
        return self._json_data


DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"


def test_successful_flow(monkeypatch, capsys):
    # Arrange: stub device flow and token responses
    device_data = {
        "device_code": "DEV_CODE",
        "user_code": "USER_CODE_123",
        "verification_uri": "https://github.com/verify",
        "interval": 1,
    }
    token_data = {"access_token": "TOKEN_ABC"}

    def fake_post(url, data=None, headers=None):
        if url == DEVICE_CODE_URL:
            # Verify initial request payload
            assert data.get("client_id") == gba.GITHUB_APP_CLIENT_ID
            assert "repo" in data.get("scope", "")
            return DummyResponse(200, device_data)
        if url == TOKEN_URL:
            return DummyResponse(200, token_data)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(gba.requests, "post", fake_post)
    monkeypatch.setattr(gba.webbrowser, "open", lambda uri: True)
    # Stub saving token to avoid filesystem I/O
    saved = {"token": None}

    def fake_save(token: str) -> bool:
        saved["token"] = token
        return True

    monkeypatch.setattr(gba, "save_github_token", fake_save)
    # Speed up polling logic
    monkeypatch.setattr(gba.time, "sleep", lambda _: None)
    monkeypatch.setattr(gba.time, "time", lambda: 0)

    # Act
    token = gba.authenticate_github_browser()

    # Assert
    assert token == "TOKEN_ABC"
    assert saved["token"] == "TOKEN_ABC"
    out = capsys.readouterr().out
    assert "USER_CODE_123" in out
    assert "https://github.com/verify" in out


def test_authorization_pending_then_success(monkeypatch, capsys):
    # Arrange: stub device flow and token polling sequence
    device_data = {
        "device_code": "DEV_CD",
        "user_code": "USER_CD",
        "verification_uri": "https://github.com/verify2",
        "interval": 1,
    }
    # First call: pending, then success
    token_sequence = [
        {"error": "authorization_pending"},
        {"access_token": "FINAL_TOKEN"},
    ]
    call = {"count": 0}

    def fake_post(url, data=None, headers=None):
        if url == DEVICE_CODE_URL:
            return DummyResponse(200, device_data)
        if url == TOKEN_URL:
            resp = token_sequence[call["count"]]
            call["count"] += 1
            return DummyResponse(200, resp)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(gba.requests, "post", fake_post)
    monkeypatch.setattr(gba.webbrowser, "open", lambda uri: True)
    monkeypatch.setattr(gba, "save_github_token", lambda token: True)
    monkeypatch.setattr(gba.time, "sleep", lambda _: None)
    monkeypatch.setattr(gba.time, "time", lambda: 0)

    # Act
    token = gba.authenticate_github_browser()

    # Assert
    assert token == "FINAL_TOKEN"
    out = capsys.readouterr().out
    assert "USER_CD" in out
    assert call["count"] == 2


def test_timeout_path(monkeypatch, capsys):
    # Arrange: stub device flow; simulate timeout by advancing time
    device_data = {
        "device_code": "DEV_TIMEOUT",
        "user_code": "USER_TIMEOUT",
        "verification_uri": "https://github.com/timeout",
        "interval": 1,
    }

    def fake_post(url, data=None, headers=None):
        if url == DEVICE_CODE_URL:
            return DummyResponse(200, device_data)
        if url == TOKEN_URL:
            # Should not reach polling due to immediate timeout
            return DummyResponse(200, {"error": "authorization_pending"})
        raise AssertionError(f"Unexpected URL called: {url}")

    # Simulate time progression: start_time=0, next check returns > timeout
    times = [0, 301]

    def fake_time():
        return times.pop(0)

    monkeypatch.setattr(gba.requests, "post", fake_post)
    monkeypatch.setattr(gba.webbrowser, "open", lambda uri: True)
    monkeypatch.setattr(gba, "save_github_token", lambda token: True)
    monkeypatch.setattr(gba.time, "sleep", lambda _: None)
    monkeypatch.setattr(gba.time, "time", fake_time)

    # Act
    token = gba.authenticate_github_browser()

    # Assert
    assert token is None
    out = capsys.readouterr().out
    assert "Authentication timeout" in out


def test_error_path(monkeypatch, capsys):
    # Arrange: stub device flow; simulate non-recoverable error
    device_data = {
        "device_code": "DEV_ERR",
        "user_code": "USER_ERR",
        "verification_uri": "https://github.com/err",
        "interval": 1,
    }

    def fake_post(url, data=None, headers=None):
        if url == DEVICE_CODE_URL:
            return DummyResponse(200, device_data)
        if url == TOKEN_URL:
            return DummyResponse(200, {"error": "access_denied", "error_description": "Denied"})
        raise AssertionError(f"Unexpected URL called: {url}")

    saved = {"called": False}

    def fake_save(token: str) -> bool:
        saved["called"] = True
        return True

    monkeypatch.setattr(gba.requests, "post", fake_post)
    monkeypatch.setattr(gba.webbrowser, "open", lambda uri: True)
    monkeypatch.setattr(gba, "save_github_token", fake_save)
    monkeypatch.setattr(gba.time, "sleep", lambda _: None)
    monkeypatch.setattr(gba.time, "time", lambda: 0)

    # Act
    token = gba.authenticate_github_browser()

    # Assert
    assert token is None
    assert not saved["called"]
    out = capsys.readouterr().out
    assert "Authentication failed: Denied" in out
