import contextlib
import time
from dataclasses import dataclass
from typing import Generator, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class RunStats:
    seconds: float
    max_rss_mb: Optional[float] = None

    def as_dict(self) -> dict:
        return {"seconds": self.seconds, "max_rss_mb": self.max_rss_mb}


@contextlib.contextmanager
def track_stats() -> Generator[RunStats, None, None]:
    """Context manager to track wall time and peak RSS (if psutil available)."""
    proc = psutil.Process() if psutil else None
    start = time.perf_counter()
    max_rss = proc.memory_info().rss if proc else None
    stats = RunStats(seconds=0.0, max_rss_mb=None)
    try:
        yield stats
    finally:
        end = time.perf_counter()
        if proc:
            try:
                cur = proc.memory_info().rss
                max_rss = max(max_rss, cur) if max_rss is not None else cur
            except Exception:
                pass
        stats.seconds = end - start
        stats.max_rss_mb = (max_rss or 0) / (1024 * 1024) if max_rss is not None else None
