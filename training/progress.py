from __future__ import annotations

import time
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress(iterable, *, total: int | None = None, desc: str = "", leave: bool = True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
        return
    print(message)


@contextmanager
def timed_stage(name: str):
    start_time = time.perf_counter()
    log(f"[start] {name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log(f"[done] {name} | elapsed={format_duration(elapsed)}")


def remaining_time(start_time: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= completed:
        return "00:00"
    elapsed = time.perf_counter() - start_time
    avg = elapsed / completed
    return format_duration(avg * (total - completed))
