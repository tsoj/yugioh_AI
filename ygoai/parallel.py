"""Crash tolerant parallel map for the command line tools.

multiprocessing.Pool silently loses a task (and then waits forever) when a
worker process dies abruptly, e.g. from a crash inside the native core.
`run_parallel` uses a ProcessPoolExecutor instead; when the pool breaks, the
tasks that were in flight are re-run one at a time in fresh pools to find the
crashing one, which is reported as a Crash result instead of hanging.
"""

from __future__ import annotations

import concurrent.futures as cf
import faulthandler
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator


@dataclass
class Crash:
    task: Any
    error: str


def _init_worker(initializer, initargs):
    faulthandler.enable(file=sys.stderr, all_threads=False)
    if initializer is not None:
        initializer(*initargs)


def run_parallel(fn: Callable, tasks: Iterable, jobs: int, initializer=None,
                 initargs: tuple = ()) -> Iterator[tuple[Any, Any]]:
    """Yields (task, result) pairs as they complete; result is a Crash if the
    worker running the task died."""
    pending = list(tasks)
    suspects: list = []
    while pending or suspects:
        if suspects:
            # isolate: run each suspect alone
            task = suspects.pop(0)
            with cf.ProcessPoolExecutor(1, initializer=_init_worker,
                                        initargs=(initializer, initargs)) as ex:
                try:
                    yield task, ex.submit(fn, task).result()
                except cf.process.BrokenProcessPool as e:
                    yield task, Crash(task, f"worker died: {e}")
            continue
        ex = cf.ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(initializer, initargs))
        futures = {ex.submit(fn, t): t for t in pending}
        pending = []
        try:
            for fut in cf.as_completed(futures):
                task = futures.pop(fut)
                try:
                    yield task, fut.result()
                except cf.process.BrokenProcessPool:
                    suspects.append(task)
                    break
        finally:
            # whatever did not complete is re-run: the ones that were running
            # are suspects, the rest simply go to a new pool
            for fut, task in futures.items():
                if fut.running():
                    suspects.append(task)
                else:
                    pending.append(task)
            ex.shutdown(wait=True, cancel_futures=True)
