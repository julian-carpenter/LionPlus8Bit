import pytest

import unittest
import torch
from unittest.mock import patch
from lionplus8bit import _numerical_guard as guard


pytestmark = pytest.mark.cuda


class Guard(unittest.TestCase):
    def test_fused_matches_eager(self):
        for n in [1, 255, 256, 257, 8192]:
            for tiny in [False, True]:
                p = torch.nn.Parameter(torch.zeros(n, device="cuda"))
                p.grad = torch.ones_like(p)
                if tiny:
                    p.grad[0] = -1e-39
                state = {p: {"state1": torch.zeros_like(p)}}
                actual = guard.has_tiny_values([p], state, 1e-20)
                with patch.object(guard, "triton", None):
                    expected = guard.has_tiny_values([p], state, 1e-20)
                self.assertEqual(actual, expected)
                self.assertEqual(actual, tiny)


if __name__ == "__main__":
    unittest.main()
