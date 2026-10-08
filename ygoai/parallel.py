"""Crash tolerant parallel map for the command line tools.

multiprocessing.Pool silently loses a task (and then waits forever) when a
worker process dies abruptly, e.g. from a crash inside the native core.
`run_parallel` uses a ProcessPoolExecutor instead; when the pool breaks, the
tasks that were in flight are re-run one at a time in fresh pools to find the
crashing one, which is reported as a Crash result instead of hanging.
"""

from __future__ import annotations

import concurrent.futures as cf
import ctypes
import faulthandler
import gc
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator


@dataclass
class Crash:
    task: Any
    error: str


def _init_worker(initializer, initargs, parent: int | None = None):
    faulthandler.enable(file=sys.stderr, all_threads=False)
    # (end with the parent, however it ends: the workers are started by a forkserver,
    # outside its process group, and would otherwise live on, holding their memory)
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    if parent is not None and os.getppid() != parent:
        os._exit(0)  # (it died before the line above)
    if initializer is not None:
        initializer(*initargs)


def _run_task(fn, task):
    """A task, then the memory of its garbage back to the system (workers run
    many tasks, and the heap would otherwise stay at the largest one's peak)."""
    try:
        return fn(task)
    finally:
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


# tasks a worker process runs before it is replaced: per-process caches (e.g.
# of state views, decks, compiled scripts) only grow, which adds up over the
# hour of a whole regression suite
MAX_TASKS_PER_CHILD = 100


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
                                        initargs=(initializer, initargs, os.getpid())) as ex:
                try:
                    yield task, ex.submit(_run_task, fn, task).result()
                except cf.process.BrokenProcessPool as e:
                    yield task, Crash(task, f"worker died: {e}")
            continue
        ex = cf.ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(initializer, initargs, os.getpid()),
                                    max_tasks_per_child=MAX_TASKS_PER_CHILD)
        futures = {ex.submit(_run_task, fn, t): t for t in pending}
        pending = []
        try:
            for fut in cf.as_completed(futures):
                task = futures.pop(fut)
                try:
                    yield task, fut.result()
                except cf.process.BrokenProcessPool:
                    suspects.append(task)
                    break
        except GeneratorExit:
            # (the caller stopped early, e.g. a finished SPRT: the running tasks are
            # abandoned, not waited for - a long game can take many minutes)
            for p in list((ex._processes or {}).values()):
                p.terminate()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            # whatever did not complete is re-run: the ones that were running
            # are suspects, the rest simply go to a new pool
            for fut, task in futures.items():
                if fut.running():
                    suspects.append(task)
                else:
                    pending.append(task)
            ex.shutdown(wait=True, cancel_futures=True)
