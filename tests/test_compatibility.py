"""Integration tests using real bitsandbytes kernels.

Run: python -m pytest tests/test_compatibility.py
Uses CUDA when available, otherwise the bitsandbytes CPU backend.
"""

import io
import math
import unittest
from unittest.mock import patch

import bitsandbytes as bnb
import torch

from lionplus8bit import LionPlus8bit as PackagedOptimizer


def LionPlus8bit(*args, **kwargs):
    """Explicit compatibility mode for the original bitwise replay contract."""
    kwargs.setdefault("native_gradient_scale", False)
    kwargs.setdefault("batched_norm", False)
    return PackagedOptimizer(*args, **kwargs)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter(n, dtype=torch.float32):
    return torch.nn.Parameter(torch.linspace(-0.5, 0.5, n, device=DEVICE, dtype=dtype))


def assert_native_states_equal(test, a, pa, b, pb):
    sa, sb = a.state[pa], b.state[pb]
    test.assertEqual(sa.keys(), sb.keys())
    for key in sa:
        if isinstance(sa[key], torch.Tensor):
            torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)
        else:
            test.assertEqual(sa[key], sb[key])


class LionPlus8bitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_matches_native_lion8bit_with_manually_clipped_gradients(self):
        actual = [parameter(8192), parameter(7)]
        expected = [torch.nn.Parameter(p.detach().clone()) for p in actual]

        def groups(ps):
            return [
                {"params": [ps[0]], "lr": 0.01, "betas": (0.7, 0.91), "weight_decay": 0.13},
                {"params": [ps[1]], "lr": 0.02, "betas": (0.8, 0.94), "weight_decay": 0.0},
            ]

        a = LionPlus8bit(groups(actual), max_grad_norm=0.4)
        b = bnb.optim.Lion8bit(groups(expected))
        for t in range(20):
            gradients = [
                torch.sin(torch.arange(p.numel(), device=DEVICE) * 0.13 + t)
                * (100.0 if t % 4 == 0 else 0.1)
                for p in actual
            ]
            if t == 5:
                gradients = [torch.zeros_like(g) for g in gradients]
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
            # Direct tensor division avoids overflow in scalar/tensor's reciprocal path.
            coefficient = (torch.tensor(0.4, device=DEVICE) / norm).clamp(max=1.0)
            originals = []
            for pa, pb, grad in zip(actual, expected, gradients):
                pa.grad = grad.clone()
                originals.append(pa.grad)
                pb.grad = grad * coefficient
            a.step()
            b.step()
            for pa, pb, original, grad in zip(actual, expected, originals, gradients):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
                self.assertIs(pa.grad, original)
                torch.testing.assert_close(pa.grad, grad, rtol=0, atol=0)
                assert_native_states_equal(self, a, pa, b, pb)

    def test_real_uint8_state_and_small_tensor_float32_fallback(self):
        large, small = parameter(8192), parameter(8)
        opt = LionPlus8bit([large, small])
        large.grad, small.grad = torch.ones_like(large), torch.ones_like(small)
        opt.step()
        self.assertEqual(opt.state[large]["state1"].dtype, torch.uint8)
        self.assertEqual(opt.state[small]["state1"].dtype, torch.float32)
        self.assertEqual(opt.state[large]["absmax1"].numel(), 32)
        self.assertNotIn("exp_avg", opt.state[large])
        bytes_used = sum(
            t.numel() * t.element_size()
            for t in opt.state[large].values()
            if isinstance(t, torch.Tensor)
        )
        self.assertLess(bytes_used, large.numel() * 2)

    def test_min_8bit_size_can_quantize_small_tensors(self):
        p = parameter(16)
        opt = LionPlus8bit([p], min_8bit_size=0)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertEqual(opt.state[p]["state1"].dtype, torch.uint8)

    def test_small_tensor_path_matches_paper_equations(self):
        p, q = parameter(2), parameter(1)
        opt = LionPlus8bit(
            [{"params": [p]}, {"params": [q], "weight_decay": 0.0}],
            lr=0.1,
            betas=(0.6, 0.8),
            weight_decay=0.2,
            max_grad_norm=2.5,
        )
        x = [p.detach().clone(), q.detach().clone()]
        momentum = [torch.zeros_like(p), torch.zeros_like(q)]
        for raw in [[[3.0, 0.0], [4.0]], [[-2.0, 1.0], [0.0]], [[10.0, -10.0], [-10.0]]]:
            norm = math.hypot(*(v for row in raw for v in row))
            scale = min(1.0, 2.5 / norm)
            for i, param in enumerate([p, q]):
                param.grad = torch.tensor(raw[i], device=DEVICE)
                g = torch.tensor([scale * v for v in raw[i]], device=DEVICE)
                direction = torch.sign(0.6 * momentum[i] + 0.4 * g)
                x[i] = (1.0 - 0.1 * (0.2 if i == 0 else 0.0)) * x[i] - 0.1 * direction
                momentum[i] = 0.8 * momentum[i] + 0.2 * g
            opt.step()
            for i, param in enumerate([p, q]):
                torch.testing.assert_close(param, x[i], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(
                    opt.state[param]["state1"], momentum[i], rtol=1e-6, atol=1e-7
                )

    def test_infinite_threshold_matches_native_lion8bit(self):
        p = parameter(8192)
        q = torch.nn.Parameter(p.detach().clone())
        a = LionPlus8bit(
            [p], lr=0.01, betas=(0.95, 0.98), weight_decay=0.01, max_grad_norm=math.inf
        )
        b = bnb.optim.Lion8bit([q], lr=0.01, betas=(0.95, 0.98), weight_decay=0.01)
        for t in range(5):
            p.grad = torch.cos(torch.arange(p.numel(), device=DEVICE) + t)
            q.grad = p.grad.clone()
            a.step()
            b.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            assert_native_states_equal(self, a, p, b, q)

    def test_clipping_affects_future_direction(self):
        a_param = torch.nn.Parameter(torch.zeros(8192, device=DEVICE))
        b_param = torch.nn.Parameter(torch.zeros_like(a_param))
        a = LionPlus8bit([a_param], lr=1, betas=(0.5, 0.5), weight_decay=0, max_grad_norm=1)
        b = LionPlus8bit([b_param], lr=1, betas=(0.5, 0.5), weight_decay=0, max_grad_norm=math.inf)
        for value in [100.0, -1.0]:
            for p, opt in [(a_param, a), (b_param, b)]:
                p.grad = torch.full_like(p, value)
                opt.step()
        torch.testing.assert_close(a_param, torch.zeros_like(a_param), rtol=0, atol=0)
        torch.testing.assert_close(b_param, torch.full_like(b_param, -2.0), rtol=0, atol=0)

    def test_checkpoint_preserves_quantized_state_and_continuation(self):
        p = parameter(8192)
        a = LionPlus8bit([p], lr=0.01, max_grad_norm=0.2)
        p.grad = torch.sin(p.detach() * 10)
        a.step()
        buffer = io.BytesIO()
        torch.save(a.state_dict(), buffer)
        buffer.seek(0)
        q = torch.nn.Parameter(p.detach().clone())
        b = LionPlus8bit([q])
        b.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(b.param_groups[0]["max_grad_norm"], 0.2)
        self.assertEqual(b.state[q]["state1"].dtype, torch.uint8)
        for t in range(5):
            p.grad = torch.cos(torch.arange(p.numel(), device=DEVICE) + t)
            q.grad = p.grad.clone()
            a.step()
            b.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            assert_native_states_equal(self, a, p, b, q)

    def test_added_group_inherits_restored_global_threshold(self):
        p = parameter(1)
        source = LionPlus8bit([p], max_grad_norm=0.2)
        q, r = parameter(1), parameter(1)
        opt = LionPlus8bit([q])
        opt.load_state_dict(source.state_dict())
        opt.add_param_group({"params": [r]})
        self.assertEqual(opt.param_groups[1]["max_grad_norm"], 0.2)
        q.grad = torch.tensor([3.0], device=DEVICE)
        r.grad = torch.tensor([4.0], device=DEVICE)
        opt.step()
        torch.testing.assert_close(opt.state[q]["state1"], torch.tensor([0.0024], device=DEVICE))
        torch.testing.assert_close(opt.state[r]["state1"], torch.tensor([0.0032], device=DEVICE))

    def test_closure_runs_once_with_grad_enabled(self):
        p = torch.nn.Parameter(torch.tensor([2.0], device=DEVICE))
        opt = LionPlus8bit([p], lr=0.1, weight_decay=0)
        calls = []

        def closure():
            self.assertTrue(torch.is_grad_enabled())
            calls.append(1)
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            return loss

        with torch.no_grad():
            loss = opt.step(closure)
        self.assertEqual(len(calls), 1)
        self.assertEqual(loss.item(), 4)
        torch.testing.assert_close(p, torch.tensor([1.9], device=DEVICE))

    def test_none_and_zero_gradient_behavior(self):
        p, q = parameter(1), parameter(1)
        opt = LionPlus8bit([p, q], lr=0.1, weight_decay=0.5)
        p.grad = torch.zeros_like(p)
        before_q = q.detach().clone()
        opt.step()
        torch.testing.assert_close(p, torch.tensor([-0.475], device=DEVICE))
        torch.testing.assert_close(q, before_q, rtol=0, atol=0)
        self.assertNotIn(q, opt.state)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(opt.step())

    def test_backend_exception_restores_gradient_reference(self):
        p = parameter(8192)
        opt = LionPlus8bit([p])
        p.grad = torch.ones_like(p)
        original = p.grad
        with patch.object(
            bnb.optim.Lion8bit, "update_step", side_effect=RuntimeError("test failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                opt.step()
        self.assertIs(p.grad, original)
        torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_optimizer_hooks_run_once_even_after_native_optimizer_exists(self):
        # PyTorch wraps each concrete optimizer class's step lazily.
        native = bnb.optim.Lion8bit([parameter(1)])
        self.assertIsNotNone(native)
        p = parameter(8192)
        opt = LionPlus8bit([p], max_grad_norm=1)
        p.grad = torch.ones_like(p)
        events = []

        def before(optimizer, args, kwargs):
            events.append("pre")
            p.grad.mul_(2)

        def after(optimizer, args, kwargs):
            events.append("post")

        opt.register_step_pre_hook(before)
        opt.register_step_post_hook(after)
        opt.step()
        self.assertEqual(events, ["pre", "post"])
        torch.testing.assert_close(p.grad, torch.full_like(p, 2))
        torch.testing.assert_close(
            opt.state[p]["absmax1"],
            torch.full_like(opt.state[p]["absmax1"], 0.02 / math.sqrt(p.numel())),
        )

    def test_low_precision_clipping_coefficient_keeps_float32_precision(self):
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                p = torch.nn.Parameter(torch.zeros(10000, device=DEVICE, dtype=dtype))
                opt = LionPlus8bit([p], lr=0.1, betas=(0, 0), weight_decay=0, max_grad_norm=0.01)
                p.grad = torch.full_like(p, 60000)
                opt.step()
                torch.testing.assert_close(p, torch.full_like(p, -0.1))
                self.assertEqual(opt.state[p]["state1"].dtype, torch.uint8)
                torch.testing.assert_close(
                    opt.state[p]["absmax1"],
                    torch.full_like(opt.state[p]["absmax1"], 0.0001),
                    rtol=0.01,
                    atol=1e-7,
                )

    def test_large_finite_gradient_norm_is_stable(self):
        p = parameter(2)
        opt = LionPlus8bit([p], betas=(0, 0), weight_decay=0, max_grad_norm=1)
        p.grad = torch.full_like(p, 1e20)
        opt.step()
        torch.testing.assert_close(opt.state[p]["state1"], torch.full_like(p, 1 / math.sqrt(2)))

    def test_extreme_thresholds_with_representable_gradient_norms(self):
        cases = [
            (1e-39, 0.0, 0.0),
            (1e-300, 0.0, 0.0),
            (1e-39, 1e-38, 1e-39 / math.sqrt(2)),
            (1e50, 1.0, 1.0),
        ]
        for threshold, gradient, expected in cases:
            with self.subTest(threshold=threshold, gradient=gradient):
                p = parameter(2)
                opt = LionPlus8bit([p], betas=(0, 0), weight_decay=0, max_grad_norm=threshold)
                p.grad = torch.full_like(p, gradient)
                opt.step()
                torch.testing.assert_close(
                    opt.state[p]["state1"], torch.full_like(p, expected), rtol=1e-5, atol=0
                )

    def test_nonfinite_gradients_rejected_before_state_or_parameter_update(self):
        for bad in [math.nan, math.inf, -math.inf]:
            with self.subTest(bad=bad):
                p, q = parameter(1), parameter(1)
                before = p.detach().clone()
                opt = LionPlus8bit([p, q])
                p.grad, q.grad = torch.ones_like(p), torch.full_like(q, bad)
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    opt.step()
                torch.testing.assert_close(p, before, rtol=0, atol=0)
                self.assertEqual(len(opt.state), 0)

    def test_unsupported_gradient_types_rejected(self):
        for dtype in [torch.float64, torch.complex64]:
            p = parameter(2, dtype)
            opt = LionPlus8bit([p])
            p.grad = torch.ones_like(p)
            with self.assertRaisesRegex(RuntimeError, "float32|dtype"):
                opt.step()
        p = parameter(2)
        opt = LionPlus8bit([p])
        p.grad = torch.sparse_coo_tensor([[0]], [1.0], (2,), device=DEVICE, check_invariants=True)
        with self.assertRaisesRegex(RuntimeError, "dense"):
            opt.step()

    def test_invalid_hyperparameters_and_conflicting_thresholds(self):
        for kw in [
            dict(lr=math.nan),
            dict(weight_decay=-1),
            dict(betas=(0.9, 1)),
            dict(max_grad_norm=0),
            dict(max_grad_norm=math.nan),
            dict(min_8bit_size=-1),
        ]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                LionPlus8bit([parameter(1)], **kw)
        with self.assertRaisesRegex(ValueError, "same"):
            LionPlus8bit(
                [
                    {"params": [parameter(1)], "max_grad_norm": 1},
                    {"params": [parameter(1)], "max_grad_norm": 2},
                ]
            )

    def test_noncontiguous_gradient_is_restored(self):
        p = torch.nn.Parameter(torch.zeros((128, 64), device=DEVICE))
        opt = LionPlus8bit([p], lr=0.1, weight_decay=0)
        p.grad = torch.ones((64, 128), device=DEVICE).t()
        original = p.grad
        self.assertFalse(original.is_contiguous())
        opt.step()
        self.assertIs(p.grad, original)
        torch.testing.assert_close(p, torch.full_like(p, -0.1))

    def test_amp_training_and_scheduler_with_actual_quantized_momentum(self):
        torch.manual_seed(4)
        x = torch.randn(32, 128, device=DEVICE)
        y = x @ (0.1 * torch.randn(128, 64, device=DEVICE))
        model = torch.nn.Linear(128, 64).to(DEVICE)
        opt = LionPlus8bit(model.parameters(), lr=0.01, weight_decay=0, max_grad_norm=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-4)
        scaler = torch.amp.GradScaler(DEVICE.type)
        initial = torch.nn.functional.mse_loss(model(x), y).item()
        for _ in range(150):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        final = torch.nn.functional.mse_loss(model(x), y).item()
        self.assertLess(final, initial * 0.05)
        self.assertEqual(opt.state[model.weight]["state1"].dtype, torch.uint8)


if __name__ == "__main__":
    print(f"torch={torch.__version__}; bitsandbytes={bnb.__version__}; device={DEVICE}")
    unittest.main()
