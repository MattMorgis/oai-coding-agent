import asyncio
import contextlib
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from typing import Any, Optional, Set, TypeVar

T = TypeVar('T')


@dataclass
class InterruptHandler:
    """Manages interrupt handling for graceful cancellation of running tasks using ESC key."""
    
    _interrupted: bool = field(default=False, init=False)
    _active_tasks: Set[asyncio.Task[Any]] = field(default_factory=set, init=False)
    _event: Optional[asyncio.Event] = field(default=None, init=False)
    _key_monitor_task: Optional[asyncio.Task[Any]] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        self._event = asyncio.Event()
    
    def start_monitoring(self) -> None:
        """Start monitoring for ESC key press."""
        try:
            if self._key_monitor_task is None or self._key_monitor_task.done():
                self._key_monitor_task = asyncio.create_task(self._monitor_esc_key())
        except RuntimeError:
            # No event loop running, skip monitoring
            pass
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for ESC key press."""
        if self._key_monitor_task and not self._key_monitor_task.done():
            self._key_monitor_task.cancel()
            self._key_monitor_task = None
    
    async def _monitor_esc_key(self) -> None:
        """Monitor for ESC key press in a separate thread."""
        try:
            # Only work on Unix-like systems with a TTY
            if not sys.stdin.isatty():
                return
            
            old_settings = termios.tcgetattr(sys.stdin)
            
            def read_key() -> Optional[str]:
                """Read a single key press."""
                try:
                    tty.setraw(sys.stdin.fileno())
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # ESC key
                            return 'ESC'
                    return None
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            # Run key monitoring in executor to avoid blocking
            while not self._interrupted:
                try:
                    key = await asyncio.get_event_loop().run_in_executor(
                        None, read_key
                    )
                    if key == 'ESC':
                        self._handle_interrupt()
                        break
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Fallback if terminal handling fails
                    await asyncio.sleep(0.1)
                    
        except ImportError:
            # Fallback for systems without termios/tty
            pass
        except Exception:
            # Silent fail for any other issues
            pass
    
    def _handle_interrupt(self) -> None:
        """Handle interrupt (ESC key press)."""
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
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_monitoring()
        pass


class InterruptedError(Exception):
    """Raised when an operation is interrupted by the user."""
    pass