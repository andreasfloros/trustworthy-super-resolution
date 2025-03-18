import torch as th
import torchvision.transforms.functional as tvf
from typing import Optional, Tuple


def save_state(checkpoint_path: str,
               model: th.nn.Module,
               optimizer: Optional[th.optim.Optimizer] = None,
               ema: Optional[th.optim.swa_utils.AveragedModel] = None,
               scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None) -> None:
    """
    Save the model, optimizer and scheduler state dicts.
    """

    state_dict = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if ema is not None:
        state_dict["ema_state_dict"] = ema.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    th.save(state_dict, checkpoint_path)


def load_state(checkpoint_path: str,
               model: Optional[th.nn.Module] = None,
               optimizer: Optional[th.optim.Optimizer] = None,
               ema: Optional[th.optim.swa_utils.AveragedModel] = None,
               scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None) -> None:
    """
    Load the model, optimizer and scheduler state dicts.
    """

    checkpoint = th.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema is not None:
        ema.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def count_parameters(model: th.nn.Module) -> int:
    """
    Count the number of parameters in a model.
    """

    return sum(p.numel() for p in model.parameters())


def lwt2d(inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Lazy wavelet transform.
    (N, C, H, W) -> (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2)
    """

    ecer = inpt[..., ::2, ::2]
    ecor = inpt[..., 1::2, ::2]
    ocer = inpt[..., ::2, 1::2]
    ocor = inpt[..., 1::2, 1::2]
    return ecer, th.cat([ecor, ocer, ocor], dim=1)


def ilwt2d(coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
    """
    Inverse lazy wavelet transform.
    (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2) -> (N, C, H, W)
    """

    ecer, [ecor, ocer, ocor] = coarse, details.chunk(3, dim=1)
    x = th.empty(coarse.shape[0],
                 coarse.shape[1],
                 2 * coarse.shape[2],
                 2 * coarse.shape[3],
                 device=coarse.device)
    x[..., ::2, ::2] = ecer
    x[..., 1::2, ::2] = ecor
    x[..., ::2, 1::2] = ocer
    x[..., 1::2, 1::2] = ocor
    return x


def haar2d(inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Haar transform.
    (N, C, H, W) -> (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2)
    """

    top_left, details = lwt2d(inpt)
    bottom_left, top_right, bottom_right = details.chunk(3, dim=1)
    coarse = (top_left + bottom_left + top_right + bottom_right) / 4
    return coarse, th.cat(((top_left + top_right - bottom_left - bottom_right) / 4,
                           (top_left + bottom_left - top_right - bottom_right) / 4,
                           (top_left + bottom_right - top_right - bottom_left) / 4), dim=1)


def ihaar2d(coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
    """
    Inverse haar transform.
    (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2) -> (N, C, H, W)
    """

    d1, d2, d3 = details.chunk(3, dim=1)
    return ilwt2d(coarse + d1 + d2 + d3,
                  th.cat((coarse - d1 + d2 - d3,
                          coarse + d1 - d2 - d3,
                          coarse - d1 - d2 + d3), dim=1))


def modcrop(x: th.Tensor, scale: int) -> th.Tensor:
    """
    Crop the input image dimensions to be a multiple of the scale.
    """

    h = x.shape[-2] - x.shape[-2] % scale
    w = x.shape[-1] - x.shape[-1] % scale
    return x[..., :h, :w]


def to01quant(x: th.Tensor) -> th.Tensor:
    x = (x + 1) / 2
    return (x * 255).clamp(0, 255) / 255


def bicubic_down_sx(x: th.Tensor, s: int) -> th.Tensor:
    """
    Bicubic downsample by sx.
    """

    return tvf.resize(x, [x.shape[-2] // s, x.shape[-1] // s],
                      interpolation=tvf.InterpolationMode.BICUBIC,
                      antialias=True)


def nearest_up_zx(x: th.Tensor, z: int) -> th.Tensor:
    """
    Nearest upsample by zx.
    """

    return tvf.resize(x, [x.shape[-2] * z, x.shape[-1] * z], interpolation=tvf.InterpolationMode.NEAREST_EXACT)


def bicubic_down_sx_nearest_up_zx(x: th.Tensor, s: int, z: int) -> th.Tensor:
    """
    Bicubic downsample by sx and nearest upsample by zx.
    """

    return nearest_up_zx(bicubic_down_sx(x, s), z)


@th.inference_mode()
def ddim(diffuser: th.nn.Module,
         init: th.Tensor,
         condition: Optional[th.Tensor] = None,
         steps: Optional[int] = None,
         eta: float = 1.,
         clamp_min: float = -th.inf,
         clamp_max: float = th.inf) -> th.Tensor:
    """
    Diffuse a tensor under a DDIM schedule.
    Defaults to DDPM for steps=None, eta=1.

    Args:
        diffuser: contains the non-blind denoiser (epsilon prediction).
            Should take ([x, condition], t) as input if conditional and (x, t) otherwise.
        init: the initial tensor.
        condition: the condition tensor. None if the denoiser is not conditional.
        steps: the number of steps to diffuse.
        eta: the eta parameter.
        clamp_min: the minimum value of the clamp.
        clamp_max: the maximum value of the clamp.
    """

    x = init
    if steps is None:
        steps = int(diffuser.T)
    times = th.linspace(int(diffuser.T) - 1, 0, steps, dtype=int, device=x.device)

    if steps == 0:
        return x

    for index in range(len(times)):
        t = times[index]
        sqrt_alpha_cumprod = diffuser.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = diffuser.sqrt_one_minus_alphas_cumprod[t]

        inpt = th.cat((x, condition), 1) if condition is not None else x
        eps_pred = diffuser.epsilon(inpt, t)
        x0_pred = (x - sqrt_one_minus_alpha_cumprod * eps_pred) / sqrt_alpha_cumprod
        x0_pred.clamp_(clamp_min, clamp_max)
        eps_pred = (x - sqrt_alpha_cumprod * x0_pred) / sqrt_one_minus_alpha_cumprod

        if index == len(times) - 1:
            return x0_pred

        prev_t = times[index + 1]
        alpha_cumprod = diffuser.alphas_cumprod[t]
        alpha_cumprod_prev = diffuser.alphas_cumprod[prev_t]
        sqrt_alpha_cumprod_prev = diffuser.sqrt_alphas_cumprod[prev_t]
        sqrt_one_minus_alpha_cumprod_prev = diffuser.sqrt_one_minus_alphas_cumprod[prev_t]

        std = eta * (sqrt_one_minus_alpha_cumprod_prev / sqrt_one_minus_alpha_cumprod) * th.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)  # noqa: E501
        x = sqrt_alpha_cumprod_prev * x0_pred + th.sqrt(1 - alpha_cumprod_prev - std ** 2) * eps_pred + std * th.randn_like(x)  # noqa: E501
