import json
import modules
import time
import torch as th
import torchvision as tv
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Optional
import utils
import os


def main(rank: int,
         world_size: int,
         batch_size: int,
         epochs: int,
         load_path: Optional[str],
         config_path: str,
         save_path: str,
         data_path: str,
         num_workers: int,
         port: str,
         save_every: int) -> None:

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    th.cuda.set_device(rank)

    with open(config_path, "r") as f:
        config = json.load(f)
        num_channels = config["num_channels"]
        scale = config["scale"]
        config = config["flow"]

    transform = []
    if ps := config.get("patch_size", None):
        transform += [tv.transforms.RandomCrop(ps)]
    if config.get("horizontal_flip", False):
        transform += [tv.transforms.RandomHorizontalFlip()]
    if config.get("vertical_flip", False):
        transform += [tv.transforms.RandomVerticalFlip()]
    transform += [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    transform = tv.transforms.Compose(transform)
    dataset = tv.datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            sampler=DistributedSampler(dataset),
                            num_workers=num_workers)
    if rank == 0:
        print(f"{len(dataset)} images in {data_path}, {len(dataloader)} batches of size {batch_size}, {world_size} gpus.")  # noqa: E501
    model = modules.DegradationFlow(num_channels=num_channels,
                                    transform_cfg=config["transform"],
                                    scale=scale).to(rank)

    if rank == 0:
        print(f"Loaded {config_path} model with {utils.count_parameters(model)} parameters.")
    model = DDP(model, device_ids=[rank])
    optim = th.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sched = th.optim.lr_scheduler.StepLR(optim, step_size=round(1e5 / len(dataloader)), gamma=0.5)
    if load_path is not None:
        utils.load_state(checkpoint_path=load_path,
                         model=model.module,
                         scheduler=sched,
                         optimizer=optim)
        if rank == 0:
            print(f"Resuming from {load_path}.")

    max_epoch = epochs + sched.last_epoch
    dist.barrier()
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        print(f"Starting training, {epochs} epochs.", flush=True)
    for epoch in range(sched.last_epoch + 1, max_epoch + 1):
        dataloader.sampler.set_epoch(epoch)
        avg_coarse_loss, avg_details_loss, avg_loss = 0, 0, 0
        th.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for x, _ in dataloader:
            x = x.to(rank)
            x = x + th.rand_like(x) / 127.5
            optim.zero_grad()

            coarse_loss, details_loss, logdet = model(x)
            loss = 8128.125 * coarse_loss + 0.5 * details_loss - logdet

            if not th.isfinite(loss):
                raise RuntimeError("Loss is not finite.")

            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optim.step()

            avg_coarse_loss += coarse_loss.detach() / len(dataloader)
            avg_details_loss += details_loss.detach() / len(dataloader)
            avg_loss += loss.detach() / len(dataloader)
        sched.step()
        dist.barrier()
        if rank == 0:
            if epoch % save_every == 0:
                utils.save_state(checkpoint_path=os.path.join(save_path, str(epoch * len(dataloader)).zfill(8) + ".pt"),
                                 model=model.module,
                                 scheduler=sched,
                                 optimizer=optim)
            print(f"Epoch {str(epoch).zfill(len(str(max_epoch)))}/{epochs}, Avg Loss: {avg_loss.item():.6e}, \
                    Avg Coarse Loss: {avg_coarse_loss.item():.6e}, Avg Details Loss: {avg_details_loss.item():.6e}, \
                    Time: {time.perf_counter() - start:.2f} s, Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size (per gpu)")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers (per gpu)")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to load")
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    parser.add_argument("--save_path", type=str, required=True, help="path to save")
    parser.add_argument("--data_path", type=str, required=True, help="path to data")
    parser.add_argument("--port", type=str, default="12355", help="port")
    parser.add_argument("--save_every", type=int, default=1, help="save every n epochs")
    args = parser.parse_args()

    world_size = th.cuda.device_count()
    mp.spawn(main,
             args=(world_size,
                   args.batch_size,
                   args.epochs,
                   args.checkpoint_path,
                   args.config_path,
                   args.save_path,
                   args.data_path,
                   args.num_workers,
                   args.port,
                   args.save_every),
             nprocs=world_size)
