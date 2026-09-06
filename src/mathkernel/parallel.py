# =============================================================================
# MathKernel - parallel
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Persistent process pools keyed by worker count. On Windows (spawn start
# method) creating a ProcessPoolExecutor costs ~1-2s of interpreter startup
# per worker set; pooling amortizes that across every batch in the server
# process lifetime. concurrent.futures registers its own interpreter-exit
# shutdown; the explicit atexit below is belt and braces.
#
# The spawn context is pinned explicitly: fork() from the multi-threaded
# server process can deadlock (POSIX default), and Python 3.13+ warns about
# it. Spawn is already the only option on Windows/macOS, so this also makes
# behavior identical across platforms.
_SPAWN = mp.get_context("spawn")
_POOLS: dict[int, ProcessPoolExecutor] = {}


def _get_pool(workers: int) -> ProcessPoolExecutor:
    pool = _POOLS.get(workers)
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=_SPAWN)
        _POOLS[workers] = pool
    return pool


@atexit.register
def _shutdown_pools() -> None:
    for pool in _POOLS.values():
        pool.shutdown(wait=False, cancel_futures=True)
    _POOLS.clear()


def resolve_workers(requested: int | None = None, *, cap: int | None = None) -> int:
    """Resolve a worker count: explicit request wins, then CPU count, capped by settings."""
    cpus = os.cpu_count() or 4
    workers = requested if requested and requested > 0 else cpus
    if cap is not None and cap > 0:
        workers = min(workers, cap)
    return max(1, workers)


def thread_map(fn: Callable[[T], R], items: Iterable[T], *, workers: int,
               min_parallel: int = 2) -> list[R]:
    """Order-preserving map for latency-bound work (Z3 C calls, Lean subprocesses).

    Threads are sufficient there because those engines release the GIL; for pure
    Python CPU-bound work use process_map instead.
    """
    batch = list(items)
    if workers <= 1 or len(batch) < min_parallel:
        return [fn(x) for x in batch]
    with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
        return list(pool.map(fn, batch))


def process_map(fn: Callable[[T], R], items: Iterable[T], *, workers: int,
                chunksize: int | None = None, min_parallel: int = 4) -> list[R]:
    """Order-preserving map for CPU-bound pure-Python work across processes.

    `fn` must be a module-level importable function (Windows spawn start method).
    Falls back to serial execution for small batches or a single worker.
    Pools are persistent (spawn cost is paid once per process, not per call)
    and chunksize adapts to batch size so IPC round-trips stay amortized.
    """
    batch = list(items)
    if workers <= 1 or len(batch) < min_parallel:
        return [fn(x) for x in batch]
    pool = _get_pool(min(workers, len(batch)))
    if chunksize is None:
        chunksize = max(1, len(batch) // (workers * 8))
    return list(pool.map(fn, batch, chunksize=chunksize))
