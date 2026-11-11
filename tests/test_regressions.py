import pytest

import copy
import unittest
import torch
from lionplus8bit import LionPlus8bit


pytestmark = pytest.mark.cuda


class Regressions(unittest.TestCase):
    def test_ratio(self):
        p = torch.nn.Parameter(torch.ones(2, device="cuda"))
        o = LionPlus8bit([p], lr=0.01, betas=(0, 0), weight_decay=0, max_grad_norm=1e-30)
        p.grad = torch.full_like(p, 1e20)
        o.step()
        torch.testing.assert_close(
            o.state[p]["state1"], torch.full_like(p, 1e-30 / 2**0.5), rtol=1e-6, atol=0
        )

    def test_layout_rejected_before_update(self):
        p = torch.nn.Parameter(torch.ones(3, 4, device="cuda").t())
        o = LionPlus8bit([p])
        p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            o.step()
        self.assertFalse(o.state)

    def test_stream(self):
        for size in [2, 8192]:
            p = torch.nn.Parameter(torch.zeros(size, device="cuda"))
            o = LionPlus8bit([p], lr=1, betas=(0, 0), weight_decay=0)
            p.grad = torch.ones_like(p)
            o.step()
            p.data.zero_()
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            update = o.update_step

            def delay(*args):
                torch.cuda._sleep(20000000)
                return update(*args)

            o.update_step = delay
            with torch.cuda.stream(s):
                p.grad.fill_(-1)
                o.step()
            torch.cuda.synchronize()
            torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)

    def test_policy_resume(self):
        ps = [torch.nn.Parameter(torch.ones(8192, device="cuda")) for _ in range(2)]
        a = LionPlus8bit(ps, min_8bit_size=8193)
        ps[0].grad = torch.ones_like(ps[0])
        a.step()
        b = LionPlus8bit([torch.nn.Parameter(p.clone()) for p in ps])
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        self.assertEqual(b.args.min_8bit_size, 8193)

    def test_beta_representability(self):
        with self.assertRaises(ValueError):
            LionPlus8bit([torch.nn.Parameter(torch.ones(1, device="cuda"))], betas=(1 - 1e-9, 0.98))


if __name__ == "__main__":
    unittest.main(verbosity=2)
