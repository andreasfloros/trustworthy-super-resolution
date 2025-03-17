import json
import modules
import time
import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
from typing import Optional
import utils


@th.inference_mode()
def main(checkpoint_path: str,
         config_path: str,
         in_path: str,
         batch_size: int,
         use_ema: bool,
         eta: float,
         steps: Optional[int],
         seed: Optional[int]) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if seed is not None:
        th.manual_seed(seed)

    with open(config_path, "r") as f:
        config = json.load(f)
        num_channels = config["num_channels"]
        scale = config["scale"]
        transform_cfg = config["flow"]["transform"]
        config = config["dpm"]
    model = modules.SuperResolver(num_channels=num_channels,
                                  diffuser_cfg=config["diffuser"]).eval().to(device)
    model.flow = modules.DegradationFlow(num_channels=num_channels,
                                         transform_cfg=transform_cfg,
                                         scale=scale).eval().to(device)
    if use_ema:
        ema = th.optim.swa_utils.AveragedModel(model,
                                               multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
                                                   config["ema_decay"]))
        utils.load_state(checkpoint_path=checkpoint_path, ema=ema)
        model: modules.SuperResolver = ema.module
    else:
        utils.load_state(checkpoint_path=checkpoint_path, model=model)

    dataset = tv.datasets.ImageFolder(in_path,
                                      transform=tv.transforms.Compose(
                                          [tv.transforms.ToTensor(),
                                           tv.transforms.Normalize(mean=[0.5], std=[0.5])]
                                      ))
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    consistency = 0.
    psnr = 0.

    print(f"{len(dataset)} images in {in_path}, {len(dataloader)} batches of size {batch_size}, ({device}).")

    start = time.perf_counter()
    for hrs, _ in dataloader:
        hrs = hrs.to(device)
        hrs = utils.modcrop(hrs, scale)
        lrs = utils.bicubic_down_sx(hrs, s=scale)

        srs = model.sample(utils.nearest_up_zx(lrs, z=scale // 2), eta=eta, steps=steps)
        srs = utils.to01quant(srs)

        lrs = (lrs + 1) / 2
        slrs = utils.bicubic_down_sx(srs, s=scale)
        slrs = (slrs + 1) / 2

        hrs = (hrs + 1) / 2

        psnr += th.sum(th.log((srs - hrs).pow(2).mean(dim=[1, 2, 3]))) / len(dataset)
        consistency += th.sum((slrs - lrs).pow(2).mean(dim=[1, 2, 3])) / len(dataset)

    print(f"Consistency (x1e-5): {consistency.item() * 1e5:.9f}, PSNR: {psnr.item():.9f} dB, \
          Time: {time.perf_counter() - start:.2f} s.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    parser.add_argument("--in_path", type=str, required=True, help="path to input images")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--ema", action="store_true", help="use EMA")
    parser.add_argument("--eta", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    main(**vars(args))
