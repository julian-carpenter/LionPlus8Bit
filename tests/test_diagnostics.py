import types
import unittest
import torch
from lionplus8bit.diagnostics import compare_signs


class Diagnostics(unittest.TestCase):
    def test_identical_and_changed_signs(self):
        p = torch.ones(4)
        q = torch.ones(4)

        def opt(parameter, momentum):
            return types.SimpleNamespace(
                param_groups=[dict(params=[parameter], betas=(0.9, 0.98))],
                state={parameter: dict(state1=momentum)},
            )

        a = opt(p, torch.ones(4))
        b = opt(q, torch.ones(4))
        self.assertEqual(compare_signs(a, b, [torch.zeros(4)])[0]["sign_disagreements"], 0)
        b.state[q]["state1"].neg_()
        self.assertEqual(compare_signs(a, b, [torch.zeros(4)])[0]["sign_disagreements"], 4)


if __name__ == "__main__":
    unittest.main()
