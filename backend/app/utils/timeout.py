"""
Timeout utilities for long-running operations
"""
import asyncio
import functools
from typing import Callable, Any
from app.utils import get_logger

logger = get_logger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def run_with_timeout(timeout_seconds: int):
    """
    Decorator to run a synchronous function with a timeout in an async context

    Args:
        timeout_seconds: Maximum execution time in seconds

    Raises:
        TimeoutError: If execution exceeds timeout
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            loop = asyncio.get_event_loop()

            try:
                logger.debug("Running %s with timeout of %d seconds",
                           func.__name__, timeout_seconds)

                # Run the synchronous function in a thread pool with timeout
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(func, *args, **kwargs)),
                    timeout=timeout_seconds
                )

                logger.debug("Function %s completed successfully", func.__name__)
                return result

            except asyncio.TimeoutError:
                error_msg = (
                    f"Operation timed out after {timeout_seconds} seconds. "
                    f"The analysis is taking longer than expected. "
                    f"Please try with a smaller area or different date range."
                )
                logger.error("Timeout in %s: %s", func.__name__, error_msg)
                raise TimeoutError(error_msg)
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, str(e), exc_info=True)
                raise

        return wrapper
    return decorator
