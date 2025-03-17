import improved_diffusion
import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
from typing import Any, Dict, Tuple
import utils


class DenseBlock(th.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_convs: int,
                 growth_rate: int) -> None:
        super().__init__()
        self.blocks = th.nn.ModuleList(th.nn.Conv2d(in_channels=in_channels + i * growth_rate,
                                                    out_channels=growth_rate,
                                                    kernel_size=3,
                                                    padding="same")
                                       for i in range(num_convs - 1))
        self.conv_out = th.nn.Conv2d(in_channels=in_channels + (num_convs - 1) * growth_rate,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     padding="same")

        for block in self.blocks:
            th.nn.init.xavier_normal_(block.weight)
            th.nn.init.zeros_(block.bias)

        th.nn.init.zeros_(self.conv_out.weight)
        th.nn.init.zeros_(self.conv_out.bias)

    def forward(self, inpt: th.Tensor) -> th.Tensor:
        for block in self.blocks:
            inpt = th.cat((inpt, th.nn.functional.leaky_relu(block(inpt), 0.2)), dim=1)
        return self.conv_out(inpt)


class InvBlock(th.nn.Module):

    def __init__(self,
                 num_channels: int,
                 num_convs: int,
                 growth_rate: int) -> None:
        super().__init__()
        self.alpha = 2.
        self.p = DenseBlock(in_channels=num_channels,
                            out_channels=6 * num_channels,
                            num_convs=num_convs,
                            growth_rate=growth_rate)
        self.u = DenseBlock(in_channels=3 * num_channels,
                            out_channels=2 * num_channels,
                            num_convs=num_convs,
                            growth_rate=growth_rate)

    def forward(self,
                coarse: th.Tensor,
                details: th.Tensor,
                forward: bool) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if forward:
            logs1, t = self.p(coarse).chunk(2, dim=1)
            logs1 = 0.636619772368 * self.alpha * th.atan(logs1 / self.alpha)
            details = logs1.exp() * details + t
            logs2, t = self.u(details).chunk(2, dim=1)
            logs2 = 0.636619772368 * self.alpha * th.atan(logs2 / self.alpha)
            coarse = logs2.exp() * coarse + t
        else:
            logs2, t = self.u(details).chunk(2, dim=1)
            logs2 = 0.636619772368 * self.alpha * th.atan(logs2 / self.alpha)
            coarse = (coarse - t) / logs2.exp()
            logs1, t = self.p(coarse).chunk(2, dim=1)
            logs1 = 0.636619772368 * self.alpha * th.atan(logs1 / self.alpha)
            details = (details - t) / logs1.exp()
        return coarse, details, (logs1.sum() + logs2.sum()) / coarse.shape[0]


class Transform(th.nn.Module):

    def __init__(self,
                 num_channels: int,
                 num_inv: int,
                 num_convs: int,
                 growth_rate: int) -> None:
        super().__init__()
        self.blocks = th.nn.ModuleList(InvBlock(num_channels=num_channels,
                                                num_convs=num_convs,
                                                growth_rate=growth_rate)
                                       for _ in range(num_inv))
        self.register_buffer("details_mean", th.zeros(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_var", th.ones(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_min", th.empty(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_max", th.empty(1, 3 * num_channels, 1, 1))
        self.details_min.copy_(-th.inf)
        self.details_max.copy_(th.inf)

    def forward(self,
                coarse: th.Tensor,
                details: th.Tensor,
                forward: bool = True) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        total_logdet = 0
        for block in (self.blocks if forward else reversed(self.blocks)):
            coarse, details, logdet = block(coarse, details, forward)
            total_logdet = total_logdet + logdet
        return coarse, details, total_logdet


class DegradationFlow(th.nn.Module):

    def __init__(self,
                 num_channels: int,
                 transform_cfg: Dict[str, Any],
                 scale: int) -> None:
        """
        Degradation Flow
        """

        super().__init__()
        self.num_channels = num_channels
        self.scale = scale
        self.transform = Transform(num_channels=num_channels,
                                   **transform_cfg)

    def forward(self, inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        DDP training.
        """

        coarse, detail = utils.haar2d(inpt)
        coarse, detail, logdet = self.transform(coarse, detail)
        degraded = utils.bicubic_down_sx_nearest_up_zx(inpt, s=self.scale, z=self.scale // 2)
        coarse_loss = th.nn.functional.mse_loss(degraded, coarse, reduction="sum") / coarse.shape[0]
        details_loss = detail.pow(2).sum() / detail.shape[0]
        return coarse_loss, details_loss, logdet


class Diffuser(th.nn.Module):
    """
    VP diffuser module.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 T: int,
                 linear: bool,
                 unet_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.unet = improved_diffusion.UNetModel(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 channel_mult=[1,] + unet_cfg["channel_mults"],
                                                 model_channels=unet_cfg["base_channels"],
                                                 num_res_blocks=unet_cfg["num_res_attn_blocks"],
                                                 attention_resolutions=[2 ** i for i, is_attn in enumerate(unet_cfg["is_attn"]) if is_attn],  # noqa: E501
                                                 dropout=unet_cfg["dropout"],
                                                 num_heads=unet_cfg["num_heads"],
                                                 use_scale_shift_norm=unet_cfg["use_scale_shift_norm"],
                                                 )
        if linear:
            betas = th.linspace(0.1 / T, 20 / T, T, dtype=th.float64)
        else:
            s = 0.008
            steps = th.linspace(0., T, T + 1, dtype=th.float64)
            ft = th.cos(((steps / T + s) / (1 + s)) * th.pi * 0.5) ** 2
            betas = th.clip(1 - ft[1:] / ft[:T], 0., 0.999)

        sqrt_betas = th.sqrt(betas)
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        one_minus_alphas_cumprod = 1 - alphas_cumprod
        sqrt_alphas = th.sqrt(alphas)

        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = th.sqrt(one_minus_alphas_cumprod)

        self.register_buffer("betas", betas.to(th.float32))
        self.register_buffer("sqrt_betas", sqrt_betas.to(th.float32))
        self.register_buffer("alphas", alphas.to(th.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(th.float32))
        self.register_buffer("one_minus_alphas_cumprod", one_minus_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_alphas", sqrt_alphas.to(th.float32))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(th.float32))

        T = th.tensor(T, dtype=th.float32).unsqueeze_(0)
        self.register_buffer("T", T)

    def forward(self, x: th.Tensor, t: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Noising from 0 to t.
        """

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        eps = th.randn_like(x)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * eps, eps

    def randint(self, batch_size: int, device: th.device) -> th.Tensor:
        """
        Sample a random time step.
        """

        return th.randint(low=0, high=len(self.betas), size=(batch_size, 1, 1, 1), device=device)

    def epsilon(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        return self.unet(x, t * 1000. / len(self.betas))


class SuperResolver(th.nn.Module):

    def __init__(self,
                 num_channels: int,
                 diffuser_cfg: Dict[str, Any]) -> None:

        super().__init__()
        self.dpm = Diffuser(in_channels=4 * num_channels,
                            out_channels=3 * num_channels,
                            T=diffuser_cfg["T"],
                            linear=diffuser_cfg["linear"],
                            unet_cfg=diffuser_cfg["unet"])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        DDP training.
        """

        coarse, details, _ = self.flow.transform(*utils.haar2d(x))
        details = (details - self.flow.transform.details_mean) / (2 * self.flow.transform.details_var.sqrt())
        t = self.dpm.randint(x.shape[0], x.device)
        details_t, eps = self.dpm(details, t)
        pred = self.dpm.epsilon(th.cat((details_t, coarse), 1), t)
        return th.nn.functional.mse_loss(pred, eps, reduction="mean")

    @th.no_grad()
    def calculate_stats(self, data_path: str, batch_size: int = 1, num_workers: int = 0) -> None:
        """
        Calculate the normalization stats for the transform.

        Args:
            data_path: the path to the data.
            batch_size: the batch size.
            num_workers: the number of workers.
        """

        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        dataset = tv.datasets.ImageFolder(data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        self.flow.transform.details_mean.zero_()
        self.flow.transform.details_var.zero_()
        self.flow.transform.details_min.copy_(th.inf)
        self.flow.transform.details_max.copy_(-th.inf)
        device = next(self.parameters()).device

        num = len(dataset)

        for inpt, _ in dataloader:
            inpt = inpt.to(device)
            _, details, _ = self.flow.transform(*utils.haar2d(inpt))
            size = num * details.shape[2] * details.shape[3]
            self.flow.transform.details_mean += details.sum(dim=(0, 2, 3), keepdim=True) / size
            self.flow.transform.details_var += details.pow(2).sum(dim=(0, 2, 3), keepdim=True) / (size - 1)
            self.flow.transform.details_min.copy_(th.min(self.flow.transform.details_min, details.min()))
            self.flow.transform.details_max.copy_(th.max(self.flow.transform.details_max, details.max()))

        self.flow.transform.details_var -= self.flow.transform.details_mean.pow(2) * size / (size - 1)

    @th.inference_mode()
    def sample(self, inpt: th.Tensor, steps: int, eta: float) -> th.Tensor:
        """
        Super resolve an image.
        """

        details = utils.ddim(diffuser=self.dpm,
                             init=th.randn(inpt.shape[0], 3 * inpt.shape[1], *inpt.shape[2:], device=inpt.device),
                             condition=inpt,
                             steps=steps,
                             eta=eta,
                             clamp_min=(self.flow.transform.details_min - self.flow.transform.details_mean)
                             / (2 * self.flow.transform.details_var.sqrt()),
                             clamp_max=(self.flow.transform.details_max - self.flow.transform.details_mean)
                             / (2 * self.flow.transform.details_var.sqrt())
                             )
        details = details * (2 * self.flow.transform.details_var.sqrt()) + self.flow.transform.details_mean
        return utils.ihaar2d(*self.flow.transform(inpt, details, forward=False)[:2])
