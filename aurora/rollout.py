"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["merge_boundaries", "rollout"]


def merge_boundaries(cerra_batch: Batch, boundary_batch: Batch, boundary_size: tuple[int, int, int, int]) -> Batch:
    x_range = slice(boundary_size[0], -boundary_size[1])
    y_range = slice(boundary_size[2], -boundary_size[3])

    # Merge the boundary conditions and the initial conditions
    # by replacing the center part of the boundary conditions with the initial conditions.
    surf_vars = {}
    atmos_vars = {}
    static_vars = {}
    for k in boundary_batch.surf_vars:
        values = boundary_batch.surf_vars[k].clone()
        values[:, :, x_range, y_range] = cerra_batch.surf_vars[k]

        surf_vars[k] = values

    for k in boundary_batch.atmos_vars:
        values = boundary_batch.atmos_vars[k].clone()
        values[:, :, :, x_range, y_range] = cerra_batch.atmos_vars[k]

        atmos_vars[k] = values

    for k in boundary_batch.static_vars:
        values = boundary_batch.static_vars[k].clone()
        values[x_range, y_range] = cerra_batch.static_vars[k]

        static_vars[k] = values

    merged_batch = dataclasses.replace(
        boundary_batch,
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
    )

    return merged_batch

def remove_boundaries(batch: Batch, boundary_size: tuple[int, int, int, int]) -> Batch:
    """Remove the boundary conditions from the batch.

    Args:
        batch (:class:`aurora.batch.Batch`): The batch to remove the boundaries from.
        boundary_size (tuple[int, int, int, int]): The size of the boundaries to remove.

    Returns:
        :class:`aurora.batch.Batch`: The batch without the boundaries.
    """
    x_max = batch.surf_vars[next(iter(batch.surf_vars))].shape[2] - boundary_size[1]
    y_max = batch.surf_vars[next(iter(batch.surf_vars))].shape[3] - boundary_size[3]
    x_range = slice(boundary_size[0], x_max)
    y_range = slice(boundary_size[2], y_max)

    surf_vars = {k: v[:, :, x_range, y_range] for k, v in batch.surf_vars.items()}
    atmos_vars = {k: v[:, :, :, x_range, y_range] for k, v in batch.atmos_vars.items()}
    static_vars = {k: v[x_range, y_range] for k, v in batch.static_vars.items()}

    batch_without_boundaries = dataclasses.replace(
        batch,
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
    )

    batch_without_boundaries.metadata.lon = batch_without_boundaries.metadata.lon[x_range, y_range]
    batch_without_boundaries.metadata.lat = batch_without_boundaries.metadata.lat[x_range, y_range]

    return batch_without_boundaries


def rollout(
        model: Aurora,
        batch: Batch,
        steps: int,
        boundary_size: tuple[int, int, int, int],
        boundary_batches: list[Batch] | None = None,
        return_last_input: bool = True,
) -> Generator[tuple[Batch, Batch | None], None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
        boundary_size (tuple[int, int, int, int]): The size of the rollout boundaries (left, right, top, bottom).
        boundary_batches (list[:class:`aurora.batch.Batch`], optional): A list of boundary batches to merge with the prediction.
            If None, no boundaries will be merged. Defaults to None.
        return_last_input (bool): Whether to return the rollout input for the step after the last rollout step

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        patch_size = model.module.patch_size
    else:
        patch_size = model.patch_size

    batch = batch.type(p.dtype)
    batch = batch.crop(patch_size)
    batch = batch.to(p.device)

    for step in range(steps):
        pred = model.forward(batch)
        pred = remove_boundaries(pred, boundary_size)

        if step == steps - 1 and not return_last_input:
            # In this case we don't care about the input to the next step, so we return None
            # This allows us to avoid loading the boundary data for the last step in the rollout case
            yield pred, None
        else:
            if boundary_batches is not None:
                # If we have boundary batches, merge the boundaries with the prediction.
                boundary_batch = boundary_batches.pop(0)
                merged_pred = merge_boundaries(pred, boundary_batch, boundary_size)
            else:
                # If we don't have boundary batches, just use the prediction as is.
                merged_pred = pred

            # Add the appropriate history so the model can be run on the prediction.
            batch = dataclasses.replace(
                merged_pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                    for k, v in merged_pred.surf_vars.items()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                    for k, v in merged_pred.atmos_vars.items()
                },
            )

            yield pred, batch.detach()


