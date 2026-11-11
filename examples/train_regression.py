"""Small runnable example; it is a smoke test, not a convergence benchmark."""

import argparse
import torch
from lionplus8bit import LionPlus8bit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--clipping", choices=["global", "adagc"], default="global")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing CPU fallback")
    torch.manual_seed(4)
    torch.set_num_threads(1)
    model = torch.nn.Linear(128, 64).to(args.device)
    inputs = torch.randn(32, 128, device=args.device)
    targets = torch.randn(32, 64, device=args.device)
    optimizer = LionPlus8bit(
        model.parameters(), lr=0.01, weight_decay=0, max_grad_norm=1.0, clipping=args.clipping
    )
    initial = torch.nn.functional.mse_loss(model(inputs), targets).item()
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(inputs), targets).item()
    print(f"device={args.device}, initial_mse={initial:.8f}, final_mse={final:.8f}")
    print("weight momentum:", optimizer.state[model.weight]["state1"].dtype)
    print("bias momentum:", optimizer.state[model.bias]["state1"].dtype)
    if final >= 0.05 * initial:
        raise SystemExit("Smoke-test loss did not fall sufficiently")


if __name__ == "__main__":
    main()
