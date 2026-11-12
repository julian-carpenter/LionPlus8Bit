"""FP32 weights, FP16 autocast and AdaGC, one process/optimizer per GPU.

Launch: torchrun --standalone --nproc-per-node=4 examples/train_ddp.py
Use NCCL_P2P_DISABLE=1 only if needed for your node's transport configuration.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lionplus8bit import LionPlus8bit


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    model = DDP(torch.nn.Linear(64, 128).cuda(), device_ids=[local_rank])
    optimizer = LionPlus8bit(model.parameters(), clipping="adagc", max_grad_norm=1)
    scaler = torch.amp.GradScaler("cuda", init_scale=32)
    for step in range(120):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(8, 64, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x).float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if dist.get_rank() == 0:
        print("Completed 120 DDP/AMP/AdaGC attempts")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
