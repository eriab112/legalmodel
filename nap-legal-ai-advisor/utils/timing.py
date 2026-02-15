"""
Performance timing for NAP Legal AI Advisor.
Logs timing for key operations to identify bottlenecks.
"""

import time
import functools
from typing import Optional


_timings: dict = {}


def timed(label: Optional[str] = None):
    """Decorator that logs execution time of a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = label or f"{func.__module__}.{func.__qualname__}"
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _timings[name] = elapsed
            print(f"\u23f1 {name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


def get_timings() -> dict:
    """Return all recorded timings."""
    return dict(_timings)


def clear_timings():
    """Clear all recorded timings."""
    _timings.clear()
