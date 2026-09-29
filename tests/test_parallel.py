"""The parallel runner must survive worker crashes."""

import os
import unittest

from ygoai.parallel import Crash, run_parallel


def work(x):
    if x == 3:
        os._exit(1)  # simulates a segfault in native code
    return x * x


class ParallelTest(unittest.TestCase):
    def test_crash_is_reported(self):
        results = dict(run_parallel(work, range(8), jobs=3))
        self.assertEqual(sorted(results), list(range(8)))
        self.assertIsInstance(results[3], Crash)
        for x in range(8):
            if x != 3:
                self.assertEqual(results[x], x * x)


if __name__ == "__main__":
    unittest.main()
