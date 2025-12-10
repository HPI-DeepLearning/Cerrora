import torch
from aurora import Batch, Metadata


def collate_fn(batch_list: list[dict[str, Batch]]) -> dict[str, Batch]:
    """
    Custom collate function to merge individual Batch objects into a single Batch object.
    Latitude, longitude, and atmospheric levels are assumed to be identical across batches and
    are directly taken from the first sample.

    Args:
        batch_list (list[tuple[Batch, Batch]]): List of Batch objects to collate.

    Returns:
        Batch: A single Batch object where each variable has a new first dimension for the batch size.
    """
    # Collate variables by stacking along the first dimension
    collated_batch = {}

    for key in batch_list[0].keys():
        collated_batch[key] = collate_data(batch_list, key)

    return collated_batch

def collate_data(batch_list: list[dict[str, Batch]], batch_key: str) -> Batch:

    collated_surf_vars = {key: torch.stack([b[batch_key].surf_vars[key] for b in batch_list], dim=0)
                          for key in batch_list[0][batch_key].surf_vars.keys()}

    collated_atmos_vars = {key: torch.stack([b[batch_key].atmos_vars[key] for b in batch_list], dim=0)
                           for key in batch_list[0][batch_key].atmos_vars.keys()}

    # Squeeze the extra dimensions
    collated_surf_vars = {key: value.squeeze(1) for key, value in collated_surf_vars.items()}
    collated_atmos_vars = {key: value.squeeze(1) for key, value in collated_atmos_vars.items()}
    collated_static_vars = batch_list[0][batch_key].static_vars

    # Metadata: Take lat, lon, and atmos_levels from the first batch since they are identical
    rollout_step = torch.cat([b[batch_key].metadata.rollout_step for b in batch_list], dim=0)
    collated_metadata = Metadata(
        lat=batch_list[0][batch_key].metadata.lat,
        lon=batch_list[0][batch_key].metadata.lon,
        time=tuple(b[batch_key].metadata.time[0] for b in batch_list),  # Collect times for all batches
        atmos_levels=batch_list[0][batch_key].metadata.atmos_levels,
        rollout_step=rollout_step,
    )

    # Create a new Batch object with the collated data
    collated_batch = Batch(
        surf_vars=collated_surf_vars,
        static_vars=collated_static_vars,
        atmos_vars=collated_atmos_vars,
        metadata=collated_metadata
    )
    return collated_batch
