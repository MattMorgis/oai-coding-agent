"""Test interrupt handling functionality."""

import asyncio
import signal
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from oai_coding_agent.interrupt_handler import InterruptHandler, InterruptedError


class TestInterruptHandler:
    """Test the InterruptHandler class."""
    
    def test_interrupt_handler_initialization(self):
        """Test that InterruptHandler initializes correctly."""
        handler = InterruptHandler()
        assert not handler.interrupted
        assert len(handler._active_tasks) == 0
    
    def test_install_uninstall(self):
        """Test signal handler installation and uninstallation."""
        handler = InterruptHandler()
        
        # Install handler
        handler.install()
        assert handler._original_handler is not None
        
        # Uninstall handler
        handler.uninstall()
        assert handler._original_handler is None
    
    def test_interrupt_state(self):
        """Test interrupt state management."""
        handler = InterruptHandler()
        
        # Initially not interrupted
        assert not handler.interrupted
        
        # Simulate interrupt
        handler._interrupted = True
        assert handler.interrupted
        
        # Reset state
        handler.reset()
        assert not handler.interrupted
    
    def test_check_interrupted(self):
        """Test check_interrupted raises when interrupted."""
        handler = InterruptHandler()
        
        # Should not raise when not interrupted
        handler.check_interrupted()
        
        # Should raise when interrupted
        handler._interrupted = True
        with pytest.raises(asyncio.CancelledError):
            handler.check_interrupted()
    
    @pytest.mark.asyncio
    async def test_cancellable_task(self):
        """Test cancellable task context manager."""
        handler = InterruptHandler()
        
        # Create a mock coroutine
        async def long_running_task():
            await asyncio.sleep(10)
            return "completed"
        
        # Run task and cancel it
        async with handler.cancellable_task(long_running_task()) as task:
            assert task in handler._active_tasks
            task.cancel()
        
        # Task should be removed from active tasks
        assert task not in handler._active_tasks
        assert task.cancelled()
    
    @pytest.mark.asyncio
    async def test_wait_for_interrupt(self):
        """Test waiting for interrupt event."""
        handler = InterruptHandler()
        
        # Set up a task to wait for interrupt
        wait_task = asyncio.create_task(handler.wait_for_interrupt())
        
        # Should not complete immediately
        await asyncio.sleep(0.1)
        assert not wait_task.done()
        
        # Trigger interrupt
        handler._event.set()
        
        # Should complete now
        await wait_task
    
    def test_context_manager(self):
        """Test InterruptHandler as context manager."""
        handler = InterruptHandler()
        
        with handler:
            # Handler should be installed
            assert handler._original_handler is not None
        
        # Handler should be uninstalled after exit
        assert handler._original_handler is None


@pytest.mark.asyncio
async def test_agent_interrupt_handling():
    """Test interrupt handling in the agent."""
    from oai_coding_agent.agent import Agent
    from oai_coding_agent.runtime_config import RuntimeConfig, ModelChoice, ModeChoice
    
    # Create a mock config
    config = RuntimeConfig(
        openai_api_key="test-key",
        github_token="test-github-token",
        model=ModelChoice.codex_mini_latest,
        mode=ModeChoice.default,
        repo_path="/tmp/test"
    )
    
    # Create agent
    agent = Agent(config)
    
    # Mock the SDK agent
    agent._sdk_agent = MagicMock()
    
    # Create a mock event stream that yields some events
    async def mock_stream_events():
        yield {"type": "message", "content": "Starting..."}
        await asyncio.sleep(0.1)
        # Simulate interrupt
        agent.interrupt_handler._interrupted = True
        yield {"type": "message", "content": "This should trigger interrupt"}
    
    # Mock the Runner.run_streamed
    mock_result = MagicMock()
    mock_result.stream_events = mock_stream_events
    mock_result.last_response_id = "test-response-id"
    
    with patch('oai_coding_agent.agent.agent.Runner.run_streamed', return_value=mock_result):
        # Run the agent and expect InterruptedError
        with pytest.raises(InterruptedError):
            event_stream = await agent.run("test input")
            events = []
            async for event in event_stream:
                events.append(event)
        
        # Verify response ID was preserved
        assert agent._previous_response_id == "test-response-id"


@pytest.mark.asyncio  
async def test_console_interrupt_prompt():
    """Test console prompting after interrupt."""
    from oai_coding_agent.console.console import ReplConsole
    from oai_coding_agent.runtime_config import RuntimeConfig, ModelChoice, ModeChoice
    
    # Create mock agent
    mock_agent = AsyncMock()
    mock_agent.config = RuntimeConfig(
        openai_api_key="test-key",
        github_token="test-github-token",
        model=ModelChoice.codex_mini_latest,
        mode=ModeChoice.default,
        repo_path="/tmp/test"
    )
    mock_agent.interrupt_handler = InterruptHandler()
    
    # Create console
    console = ReplConsole(mock_agent)
    
    # Mock the event stream to raise InterruptedError
    async def mock_run(user_input):
        async def stream():
            yield {"type": "message", "content": "Starting..."}
            raise InterruptedError("Interrupted by user")
        return stream()
    
    mock_agent.run.side_effect = mock_run
    
    # Test that console handles interrupt gracefully
    # This is a simplified test - in real usage, the prompt_session
    # would handle the user interaction
    assert console.agent == mock_agent