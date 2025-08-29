"""
DDPM model loader utilities.
"""

import os
from typing import Callable, Optional, Tuple

import torch

from diffusion_models.models.ddpm.backbones.models import Model
from diffusion_models.models.ddpm.backbones.script_util import (
    create_model,
    create_classifier,
    classifier_defaults,
    args_to_dict,
)
from functions.ckpt_util import get_ckpt_path, download


try:
    from omegaconf import OmegaConf, DictConfig
except ImportError:  # Optional: allows usage without OmegaConf in limited contexts
    OmegaConf = None
    DictConfig = None


class DDPMModelLoader:
    """
    Loader for DDPM / OpenAI diffusion UNet models.

    Input:
        config : OmegaConf / Namespace
            Global hydra config object (must contain `.model`, `.data`, and when
            class-conditional `.classifier`).
        args : Namespace
            Runtime arguments containing at least `exp` (experiment directory).
        device : torch.device | str | None
            Target device. If None, CUDA is auto-selected when available.
        data_parallel : bool
            Whether to wrap the model (and classifier) in DataParallel.
    """

    def __init__(
        self,
        config,
        device: Optional[torch.device] = None,
        data_parallel: bool = True,
    ) -> None:
        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        # Keep full config for cross-group references
        self.config = config
        self.exp = config.general.exp \
            if hasattr(config, "general") else os.getcwd()
        self.cfg_model = config.model  # this contains nodes: ddpm, data, classifier
        self.device = device
        self.data_parallel = data_parallel

    # Public API ---------------------------------------------------------
    def load(self) -> Tuple[torch.nn.Module, Optional[Callable]]:
        """Load (or download then load) the diffusion model (& classifier).

        Returns
        -------
        model : torch.nn.Module
            The diffusion UNet (DataParallel wrapped if enabled).
        cond_fn : Callable | None
            Classifier guidance function when classifier guidance is enabled.
        """
        model, cond_fn = self._build_model()
        return model, cond_fn

    # Internal helpers ---------------------------------------------------
    def _build_model(self) -> Tuple[torch.nn.Module, Optional[Callable]]:
        cls_fn: Optional[Callable] = None
        # ddpm section should exist under model group
        ddpm_cfg = getattr(self.cfg_model, "ddpm", None)
        if ddpm_cfg is None:
            raise ValueError("Config missing model.ddpm section")
        model_type = ddpm_cfg.type

        if model_type == "simple":
            model = self._load_simple_model()
        elif model_type == "openai":
            print('Load openai DDPM.')
            model, cls_fn = self._load_openai_model()
        else:
            raise ValueError(f"Unsupported model.type: {model_type}")

        if self.data_parallel and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        return model, cls_fn

    # --- Specific loaders ------------------------------------------------
    def _load_simple_model(self) -> torch.nn.Module:
        model = Model(self.config)  # original implementation expected full config
        dataset = getattr(self.cfg_model.data, "dataset", None)
        if dataset == "CIFAR10":
            name = "cifar10"
        elif dataset == "LSUN":
            name = f"lsun_{self.cfg_model.data.category}"
        elif dataset == "CelebA_HQ":
            name = "celeba_hq"
        else:
            raise ValueError(f"Unsupported dataset for simple model: {dataset}")

        if name != "celeba_hq":
            ckpt = get_ckpt_path(f"ema_{name}", prefix=self.exp)
        else:
            ckpt = os.path.join(self.exp, "logs/celeba/celeba_hq.ckpt")
            if not os.path.exists(ckpt):
                print("Downloading CelebA HQ checkpoint...")
                download(
                    "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt",
                    ckpt,
                )

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.to(self.device)
        return model

    def _load_openai_model(self) -> Tuple[torch.nn.Module, Optional[Callable]]:
        # Build diffusion UNet
        ddpm_cfgs = self.cfg_model.ddpm
        # Properly convert DictConfig to a plain dict (vars() on DictConfig is empty)
        if OmegaConf is not None and isinstance(ddpm_cfgs, DictConfig):
            config_dict = OmegaConf.to_container(ddpm_cfgs, resolve=True)  # type: ignore
        else:
            config_dict = dict(ddpm_cfgs)

        # create_model requires certain keys; extra keys are accepted via **kwargs
        model = create_model(**config_dict)
        if getattr(ddpm_cfgs, "use_fp16", False):
            model.convert_to_fp16()

        # Diffusion checkpoint (conditional vs unconditional)
        if getattr(ddpm_cfgs, "class_cond", False):
            ckpt = os.path.join(
                self.exp,
                "logs/imagenet/%dx%d_diffusion.pt" % (
                    ddpm_cfgs.image_size,
                    ddpm_cfgs.image_size,
                ),
            )
            if not os.path.exists(ckpt):
                print("Downloading OpenAI class-conditional diffusion checkpoint...")
                download(
                    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt"
                    % (
                        ddpm_cfgs.image_size,
                        ddpm_cfgs.image_size,
                    ),
                    ckpt,
                )
        else:
            ckpt = os.path.join(self.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
            if not os.path.exists(ckpt):
                print("Downloading OpenAI unconditional diffusion checkpoint...")
                download(
                    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
                    ckpt,
                )

        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()

        cls_fn: Optional[Callable] = None
        # Optional classifier guidance
        if getattr(ddpm_cfgs, "class_cond", False):
            classifier_ckpt = os.path.join(
                self.exp,
                "logs/imagenet/%dx%d_classifier.pt"
                % (ddpm_cfgs.image_size, ddpm_cfgs.image_size),
            )
            if not os.path.exists(classifier_ckpt):
                image_size = ddpm_cfgs.image_size
                download(
                    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt"
                    % image_size,
                    classifier_ckpt,
                )
            classifier = create_classifier(
                **args_to_dict(self.cfg_model.classifier, classifier_defaults().keys())
            )
            classifier.load_state_dict(torch.load(classifier_ckpt, map_location=self.device))
            classifier.to(self.device)
            if getattr(self.cfg_model.classifier, "classifier_use_fp16", False):
                classifier.convert_to_fp16()
            classifier.eval()
            if self.data_parallel and not isinstance(classifier, torch.nn.DataParallel):
                classifier = torch.nn.DataParallel(classifier)

            import torch.nn.functional as F  # local import to avoid unused warnings

            def cond_fn(x, t, y):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    return (
                        torch.autograd.grad(selected.sum(), x_in)[0]
                        * self.cfg_model.classifier.classifier_scale
                    )

            cls_fn = cond_fn

        return model, cls_fn


__all__ = ["DDPMModelLoader"]
