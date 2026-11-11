import pytest

import copy
import unittest
import torch
from lionplus8bit import LionPlus8bit


pytestmark = pytest.mark.cuda


class Policies(unittest.TestCase):
    def test_master_accumulates_and_restores(self):
        for dtype in [torch.float16, torch.bfloat16]:
            p = torch.nn.Parameter(torch.ones(8192, device="cuda", dtype=dtype))
            o = LionPlus8bit([p], lr=5e-5, weight_decay=0, betas=(0, 0))
            for _ in range(100):
                p.grad = torch.ones_like(p)
                o.step()
            self.assertLess(p[0].item(), 1)
            self.assertEqual(o.state[p]["master_weight"].dtype, torch.float32)
            q = torch.nn.Parameter(p.detach().clone())
            b = LionPlus8bit([q])
            b.load_state_dict(copy.deepcopy(o.state_dict()))
            p.grad = torch.ones_like(p)
            q.grad = torch.ones_like(q)
            o.step()
            b.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_optimized_defaults_against_compatibility(self):
        p = torch.nn.Parameter(torch.ones(8192, device="cuda"))
        q = torch.nn.Parameter(p.detach().clone())
        a = LionPlus8bit([p], max_grad_norm=0.5, native_gradient_scale=False, batched_norm=False)
        b = LionPlus8bit([q], max_grad_norm=0.5)
        for i in range(10):
            p.grad = (torch.arange(p.numel(), device=p.device).float() % 17 + 1) * (i + 1) / 17
            q.grad = p.grad.clone()
            a.step()
            b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(
            a.state[p]["absmax1"], b.state[q]["absmax1"], rtol=3e-6, atol=1e-9
        )

    def test_quantized_subnormal(self):
        p = torch.nn.Parameter(torch.ones(8192, device="cuda"))
        o = LionPlus8bit([p], min_8bit_size=0, betas=(0, 0), weight_decay=0, max_grad_norm=1e-39)
        p.grad = torch.ones_like(p)
        o.step()
        self.assertTrue((o.state[p]["absmax1"] > 0).all())
        self.assertTrue((p < 1).all())


class MorePolicies(unittest.TestCase):
    def test_direct_noncontiguous_grad(self):
        for dtype in [torch.float32, torch.float16]:
            p = torch.nn.Parameter(torch.zeros((64, 128), device="cuda", dtype=dtype))
            o = LionPlus8bit([p], lr=0.1, betas=(0, 0), weight_decay=0, native_gradient_scale=True)
            g = (torch.arange(8192, device="cuda").reshape(128, 64).t() - 4096).to(dtype)
            p.grad = g
            o.step()
            torch.testing.assert_close(p, (-0.1 * g.sign()).to(dtype), rtol=0.001, atol=0)
            self.assertIs(p.grad, g)

    def test_strict_custom_stream_and_later_clip(self):
        p = torch.nn.Parameter(torch.ones(8192, device="cuda"))
        o = LionPlus8bit(
            [p],
            min_8bit_size=0,
            betas=(0, 0),
            weight_decay=0,
            max_grad_norm=1e-39,
            native_gradient_scale=True,
        )
        p.grad = torch.ones_like(p)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            torch.cuda._sleep(20000000)
            o.step()
        torch.cuda.synchronize()
        self.assertTrue((o.state[p]["absmax1"] > 0).all())
        o.param_groups[0]["max_grad_norm"] = 1.0
        p.grad = torch.ones_like(p) * 100
        o.step()
        torch.testing.assert_close(
            o.state[p]["absmax1"],
            torch.full_like(o.state[p]["absmax1"], 1 / 8192**0.5),
            rtol=1e-5,
            atol=0,
        )

    def test_master_tiny_clipping(self):
        p = torch.nn.Parameter(torch.ones(8192, device="cuda", dtype=torch.float16))
        o = LionPlus8bit([p], max_grad_norm=1e-10)
        p.grad = torch.ones_like(p)
        o.step()
        self.assertTrue((o.state[p]["absmax1"] > 0).all())


class Overrides(unittest.TestCase):
    def test_lazy_override_resume(self):
        import bitsandbytes as bnb

        ps = [torch.nn.Parameter(torch.ones(8192, device="cuda")) for _ in range(2)]
        bnb.optim.GlobalOptimManager.get_instance().override_config(ps[1], "optim_bits", 32)
        a = LionPlus8bit(ps)
        ps[0].grad = torch.ones_like(ps[0])
        a.step()
        qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
        b = LionPlus8bit(qs)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        qs[1].grad = torch.ones_like(qs[1])
        b.step()
        self.assertEqual(b.state[qs[1]]["state1"].dtype, torch.float32)


class MixedScale(unittest.TestCase):
    def test_unrepresentable_threshold(self):
        p = torch.nn.Parameter(torch.ones(1, device="cuda"))
        o = LionPlus8bit([p], max_grad_norm=1e-300)
        p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(ValueError, "representability"):
            o.step()
        self.assertFalse(o.state)

    def test_tiny_coordinate_among_normal_gradients(self):
        p = torch.nn.Parameter(torch.zeros(8192, device="cuda"))
        o = LionPlus8bit([p], lr=0.1, betas=(0, 0), weight_decay=0)
        p.grad = torch.ones_like(p)
        p.grad[0] = 1e-39
        o.step()
        self.assertLess(p[0].item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
