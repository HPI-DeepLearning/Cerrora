"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.lora import LoRAMode
from aurora.model.perceiver import PerceiverResampler
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]


class Aurora(torch.nn.Module):
    """The Aurora model.

    Defaults to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
        input_surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl",), # (optional) add tp
        input_atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        output_surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl", ), # (optional) add tp
        output_atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        lead_time_hours: int = 6,
        use_lora: bool = False,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        autocast: bool = True, #False True
    ) -> None:
        """Construct an instance of the model.

        Args:
            input_surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model as inputs.
            input_atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model as inputs.
            output_surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model as outputs.
            output_atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model as outputs.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            window_size (tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here. For all cases except one, this is equal to `64`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            drop_path (float, optional): Drop-path rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optioanl): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            max_history_size (int, optional): Maximum number of history steps. You can load
                checkpoints with a smaller `max_history_size`, but you cannot load checkpoints
                with a larger `max_history_size`.
            lead_time_hours (int, optional): Prediction lead time in hours.
            use_lora (bool, optional): Use LoRA adaptation.
            lora_steps (int, optional): Use different LoRA adaptation for the first so-many roll-out
                steps.
            lora_mode (str, optional): LoRA mode. `"single"` uses the same LoRA for all roll-out
                steps, and `"all"` uses a different LoRA for every roll-out step. Defaults to
                `"single"`.
            autocast (bool, optional): Use `torch.autocast` with bf16 for the backbone to reduce memory usage. 
                Pre/post backbone computations remain in float32. Defaults to `True`.
        """
        super().__init__()
        self.input_surf_vars = input_surf_vars
        self.input_atmos_vars = input_atmos_vars
        self.output_surf_vars = output_surf_vars
        self.output_atmos_vars = output_atmos_vars
        self.patch_size = patch_size
        self.autocast = autocast
        print(f"autocast: {self.autocast}")
        self.max_history_size = max_history_size
        self.lead_time_hours = lead_time_hours

        self.encoder = Perceiver3DEncoder(
            surf_vars=input_surf_vars,
            static_vars=static_vars,
            atmos_vars=input_atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
        )

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
            use_lora=use_lora,
            lora_steps=lora_steps,
            lora_mode=lora_mode,
        )

        self.decoder = Perceiver3DDecoder(
            surf_vars=output_surf_vars,
            atmos_vars=output_atmos_vars,
            patch_size=patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=timedelta(hours=self.lead_time_hours),
        )
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                lead_time=timedelta(hours=self.lead_time_hours),
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        
        pred = self.decoder(
            x,
            batch,
            lead_time=timedelta(hours=self.lead_time_hours),
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        return pred


    def load_checkpoint(self, repo: str, name: str, strict: bool = True) -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    def _interpolate_encoder_patch_weights(self, weight: torch.Tensor, old_patch_size: int,
                                     new_patch_size: int) -> torch.Tensor:
        """Adjust encoder token embed weights using bilinear interpolation.

        Note: If the results are not optimal, we might need to thoroughly 
        review the paper (https://arxiv.org/pdf/2212.08013) to modify the interpolation method.
        
        Args:
            weight: Input weight tensor of shape [D, 1, T, P, P]
            old_patch_size: Original patch size
            new_patch_size: New patch size

        Returns:
            Adjusted weight tensor
        """
        D, _, T, _, _ = weight.shape

        # Calculate scaling factor to preserve input magnitude
        scale_factor = (old_patch_size / new_patch_size) ** 2

        if old_patch_size == new_patch_size:
            return weight

        # Process each time step separately
        interpolated = []
        for t in range(T):
            # Extract and reshape for interpolation [D, P, P]
            w = weight[:, 0, t]

            # Prepare for interpolation [1, D, P, P]
            w = w.unsqueeze(0)

            # Apply bilinear interpolation
            w = torch.nn.functional.interpolate(
                w,
                size=(new_patch_size, new_patch_size),
                mode='bilinear',
                align_corners=True
            )

            interpolated.append(w.squeeze(0))

        # Stack time steps back together [D, T, new_P, new_P]
        interpolated = torch.stack(interpolated, dim=1)

        # Reshape to original format and apply scaling
        return (interpolated.unsqueeze(1) * scale_factor)  # [D, 1, T, new_P, new_P]

    def _interpolate_decoder_patch_weights(self, weight: torch.Tensor, old_patch_size: int, new_patch_size: int,
                                   num_vars: int) -> torch.Tensor:
        """Adjust patch weights using bilinear interpolation.

        Note: If the results are not optimal, we might need to thoroughly 
        review the paper (https://arxiv.org/pdf/2212.08013) to modify the interpolation method.

        Args:
            weight: Input weight tensor
            old_patch_size: Original patch size
            new_patch_size: New patch size

        Returns:
            Adjusted weight tensor
        """
        if old_patch_size == new_patch_size:
            return weight

        # Calculate scaling factor to preserve input magnitude
        scale_factor = (old_patch_size / new_patch_size) ** 2
        # First reshape to separate patch dimensions and variables
        weight = weight.reshape(old_patch_size ** 2, num_vars, -1)  # [P*P, V, C]


        # Process each variable separately
        interpolated_weights = []
        for var_idx in range(num_vars):
            var_weight = weight[:, var_idx]  # [P*P, C]

            # Reshape to spatial dimensions
            var_weight = var_weight.reshape(old_patch_size, old_patch_size, -1)  # [P, P, C]

            # Prepare for interpolation
            var_weight = var_weight.permute(2, 0, 1).unsqueeze(0)  # [1, C, P, P]

            # Apply bilinear interpolation
            var_weight = torch.nn.functional.interpolate(
                var_weight,
                size=(new_patch_size, new_patch_size),
                mode='bilinear',
                align_corners=True
            )  # [1, C, new_P, new_P]

            # Reshape back
            var_weight = var_weight.squeeze(0).permute(1, 2, 0)  # [new_P, new_P, C]
            var_weight = var_weight.reshape(new_patch_size ** 2, -1)  # [new_P*new_P, C]

            interpolated_weights.append(var_weight)

        # Stack all variables back together
        weight = torch.stack(interpolated_weights, dim=1)  # [new_P*new_P, V, C]

        # Apply scaling and return
        return (weight * scale_factor).reshape(new_patch_size ** 2 * num_vars, -1)

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        # Unwrap possible checkpoint dict.
        if isinstance(d, dict) and "model" in d:
            d = d["model"]

        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        # Remove possibly prefix from the keys.
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v

        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]

            # Get original patch size from weight shape
            old_patch_size = weight.shape[3]  # Shape is [D, 1, T, P, P]

            assert weight.shape[1] == 4 + 3
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                # Extract weights for this variable and interpolate
                var_weight = weight[:, [i]]
                var_weight = self._interpolate_encoder_patch_weights(var_weight, old_patch_size, self.patch_size)
                d[f"encoder.surf_token_embeds.weights.{name}"] = var_weight

        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]

            old_patch_size = weight.shape[3]  # Shape is [D, 1, T, P, P]

            assert weight.shape[1] == 5
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                # Extract weights for this variable and interpolate
                var_weight = weight[:, [i]]
                var_weight = self._interpolate_encoder_patch_weights(var_weight, old_patch_size, self.patch_size)
                d[f"encoder.atmos_token_embeds.weights.{name}"] = var_weight

        if "decoder.surf_head.weight" in d:
            weight = d["decoder.surf_head.weight"]
            bias = d["decoder.surf_head.bias"]
            del d["decoder.surf_head.weight"]
            del d["decoder.surf_head.bias"]

            # Get original patch_size
            old_patch_size = int((weight.shape[0] / 4) ** 0.5)

            # Interpolate weights and biases
            weight = self._interpolate_decoder_patch_weights(weight, old_patch_size, self.patch_size, num_vars=4)
            bias = self._interpolate_decoder_patch_weights(bias.reshape(-1, 1), old_patch_size, self.patch_size,
                                                   num_vars=4).squeeze(-1)

            # Reshape for individual heads
            weight = weight.reshape(self.patch_size ** 2, 4, -1)
            bias = bias.reshape(self.patch_size ** 2, 4)

            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

        if "decoder.atmos_head.weight" in d:
            weight = d["decoder.atmos_head.weight"]
            bias = d["decoder.atmos_head.bias"]
            del d["decoder.atmos_head.weight"]
            del d["decoder.atmos_head.bias"]

            old_patch_size = int((weight.shape[0] / 5) ** 0.5)

            # Interpolate weights and biases
            weight = self._interpolate_decoder_patch_weights(weight, old_patch_size, self.patch_size, num_vars=5)
            bias = self._interpolate_decoder_patch_weights(bias.reshape(-1, 1), old_patch_size, self.patch_size,
                                                   num_vars=5).squeeze(-1)

            # Reshape for individual heads
            weight = weight.reshape(self.patch_size ** 2, 5, -1)
            bias = bias.reshape(self.patch_size ** 2, 5)

            for i, name in enumerate(("z", "u", "v", "t", "q")):
                d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]

        # Check if the history size is compatible and adjust weights if necessary.
        current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )

        self.load_state_dict(d, strict=strict)

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )

                # Initialize the new weight tensor.
                new_weight = torch.zeros(
                    (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                # Copy the existing weights to the new tensor by duplicating the histories provided
                # into any new history dimensions. The rest remains at zero.
                new_weight[:, :, : weight.shape[2]] = weight

                checkpoint[name] = new_weight

    def configure_activation_checkpointing(self, checkpoint_encoder_decoder: bool = False) -> None:
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        def check_fn(module):
            if isinstance(module, BasicLayer3D):
                return True
            if checkpoint_encoder_decoder and isinstance(module, PerceiverResampler):
                return True
            return False

        apply_activation_checkpointing(self, check_fn=check_fn)


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)

AuroraHighRes = partial(
    Aurora,
    patch_size=10,
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
    use_lora=False,
)