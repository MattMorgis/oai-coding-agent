import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

# Using GitHub CLI's client ID for device flow
# This is a public client ID the GitHub Coding Agent application
GITHUB_APP_CLIENT_ID = "Ov23liCVY3S4HY5FMODo"

# Default scopes for repository access
DEFAULT_SCOPES = [
    "repo",  # Full access to repositories (public and private)
    "read:user",  # Read basic user profile information
]


@dataclass
class DeviceFlowData:
    """Data returned from GitHub device flow initiation."""

    device_code: str
    user_code: str
    verification_uri: str
    interval: int
    expires_in: int


def start_device_flow(scopes: Optional[list[str]] = None) -> Optional[DeviceFlowData]:
    """
    Start GitHub device flow authentication.

    Args:
        scopes: List of GitHub OAuth scopes. Defaults to DEFAULT_SCOPES.

    Returns:
        DeviceFlowData if successful, None if request fails.
    """
    if scopes is None:
        scopes = DEFAULT_SCOPES

    response = requests.post(
        "https://github.com/login/device/code",
        data={"client_id": GITHUB_APP_CLIENT_ID, "scope": " ".join(scopes)},
        headers={"Accept": "application/json"},
    )

    if response.status_code != 200:
        return None

    data = response.json()
    return DeviceFlowData(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        interval=data.get("interval", 5),
        expires_in=data.get("expires_in", 900),
    )


def poll_for_token(
    device_code: str,
    interval: int = 5,
    timeout: int = 300,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Optional[str]:
    """
    Poll GitHub for authentication completion.

    Args:
        device_code: The device code from start_device_flow
        interval: Initial polling interval in seconds
        timeout: Maximum time to wait in seconds
        progress_callback: Optional callback called on each poll iteration

    Returns:
        Access token if authentication succeeds, None otherwise.
    """
    start_time = time.time()
    current_interval = interval

    while time.time() - start_time < timeout:
        time.sleep(current_interval)

        if progress_callback:
            progress_callback(elapsed=time.time() - start_time)

        response = requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_APP_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            continue

        result = response.json()

        if "access_token" in result:
            access_token = result["access_token"]
            if isinstance(access_token, str):
                return access_token
        elif result.get("error") == "authorization_pending":
            # User hasn't completed auth yet
            continue
        elif result.get("error") == "slow_down":
            # Increase polling interval
            current_interval = result.get("interval", current_interval + 5)
        else:
            # Authentication failed or was denied
            return None

    # Timeout reached
    return None
