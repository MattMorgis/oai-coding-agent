import asyncio
import contextlib
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, TypeVar
import types

T = TypeVar('T')


@dataclass
class InterruptHandler:
    """Manages interrupt handling for graceful cancellation of running tasks."""
    
    _interrupted: bool = field(default=False, init=False)
    _active_tasks: Set[asyncio.Task[Any]] = field(default_factory=set, init=False)
    _original_handler: Optional[Callable[[int, Optional[types.FrameType]], Any]] = field(default=None, init=False)
    _event: Optional[asyncio.Event] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        self._event = asyncio.Event()
    
    def install(self) -> None:
        """Install the interrupt handler."""
        handler = signal.signal(signal.SIGINT, self._handle_interrupt)
        if callable(handler):
            self._original_handler = handler
    
    def uninstall(self) -> None:
        """Restore the original signal handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None
    
    def _handle_interrupt(self, signum: int, frame: Optional[types.FrameType]) -> None:
        """Handle SIGINT signal."""
        self._interrupted = True
        if self._event:
            self._event.set()
        # Cancel all active tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
    
    @property
    def interrupted(self) -> bool:
        """Check if an interrupt has occurred."""
        return self._interrupted
    
    def reset(self) -> None:
        """Reset the interrupt state."""
        self._interrupted = False
        if self._event:
            self._event.clear()
    
    def check_interrupted(self) -> None:
        """Check if interrupted and raise CancelledError if so."""
        if self._interrupted:
            raise asyncio.CancelledError("Interrupted by user")
    
    async def wait_for_interrupt(self) -> None:
        """Wait for an interrupt to occur."""
        if self._event:
            await self._event.wait()
    
    @contextlib.asynccontextmanager
    async def cancellable_task(self, coro: Any) -> Any:
        """Run a coroutine as a cancellable task."""
        task = asyncio.create_task(coro)
        self._active_tasks.add(task)
        try:
            yield task
        finally:
            self._active_tasks.discard(task)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def __enter__(self) -> "InterruptHandler":
        """Context manager entry."""
        self.install()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.uninstall()
        pass


class InterruptedError(Exception):
    """Raised when an operation is interrupted by the user."""
    pass