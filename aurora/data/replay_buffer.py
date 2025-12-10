import dataclasses

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from aurora import Batch


class ReplayBuffer:
    """
    Replay buffer similar to the one used in the Aurora paper.
    It used to store sample from the original dataset as well as the outputs from previous prediction steps.

    This implementation is based on section D4 of v2 and v3 of the Aurora paper.
    https://arxiv.org/pdf/2405.13063v2
    https://arxiv.org/pdf/2405.13063

    In addition, we make use of the clarifications provided in the Aurora GitHub repository.
    https://github.com/microsoft/aurora/issues/55
    https://github.com/microsoft/aurora/issues/47
    """

    def __init__(self,
                 dataloader: DataLoader,
                 batch_size: int,
                 buffer_size: int = 200,
                 refresh_freq: int = 10,
                 max_rollout_steps: int = 4,
                 ):
        """
        Args:
            dataloader: The dataloader from which to sample ground truth data.
            batch_size: The batch size of the dataloader.
            buffer_size: The maximum number of samples to store in the buffer.
            refresh_freq: The number of steps between refreshing the buffer.
            max_rollout_steps: The number of steps to roll out. After these steps, the sample is evicted.
        """
        super().__init__()
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.refresh_freq = refresh_freq
        self.max_rollout_steps = max_rollout_steps
        self.replay_buffer = []
        self.iterator = None
        self.sampler = dataloader.sampler
        self.refresh_counter = 0
        self.rng = np.random.default_rng()

    def _add_sample_to_buffer(self, sample: dict[str, Batch]):
        self.replay_buffer.append(sample)

        # From GitHub issue #55:
        # Once the replay buffer reaches it maximum size, the next time a prediction or sample
        # is added, the oldest element in the replay buffer is ejected. Once the replay buffer
        # reaches maximum size, it therefore acts as a queue, where sampling takes a random
        # element from the queue without taking that element out of the queue.
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def add_rollout_samples(self, input_sample: dict[str, Batch], pred: Batch):
        batch_size = next(iter(pred.surf_vars.values())).shape[0]

        for i in range(batch_size):
            input_idx = {k: v[i] for k, v in input_sample.items()}
            self.add_rollout_sample(input_idx, pred[i])

    def add_rollout_sample(self, input_sample: dict[str, Batch], pred: Batch):
        input_batch = input_sample["input"].to("cpu")

        # If we have reached the max rollout steps, we don't need to add the sample to the buffer
        if self.max_rollout_steps - pred.metadata.rollout_step[0].item() == 0:
            return

        pred = pred.detach().to("cpu")
        input_batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([input_batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([input_batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        new_sample = {"input": input_batch}

        for i in range(self.max_rollout_steps - pred.metadata.rollout_step[0].item()):
            new_sample[f"target_{i}"] = input_sample[f"target_{i + 1}"]

        # We won't use these targets, so we fill them with zeros, but we need them for the collate function
        for i in range(self.max_rollout_steps - pred.metadata.rollout_step[0].item(), self.max_rollout_steps):
            new_sample[f"target_{i}"] = input_sample[f"target_{i}"]

        self._add_sample_to_buffer(new_sample)

    def __len__(self):
        # Subtract buffer size to account for initial fill
        return self.refresh_freq * len(self.dataloader)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.replay_buffer = []
        self._add_sample_to_buffer(next(self.iterator))
        self.refresh_counter = 0
        return self

    def __next__(self) -> dict[str, Batch]:
        idx = self.rng.choice(len(self.replay_buffer), self.batch_size)

        # From GitHub issue #55:
        # In strategy D.4, samples are drawn from the reply buffer with replacement.
        samples = [self.replay_buffer[i] for i in idx]
        samples = self.dataloader.collate_fn(samples)
        self.refresh_counter += 1

        if self.refresh_counter == self.refresh_freq:
            self.refresh_counter = 0
            try:
                self._add_sample_to_buffer(next(self.iterator))
            except StopIteration:
                # From GitHub issue #55:
                # For strategy D.4, once all samples of the data have been added, training
                # just continues by shuffling the entire dataset and continuing sampling from
                # the dataset. You could count this as an "epoch", but training is not
                # interrupted at this point and just continues as if the dataset were
                # infinitely large. You're right that a sampling rate of K means that it
                # takes much longer before the end of a dataset is reached. The time it takes
                # to do roll-out fine-tuning is kept constant by limiting the number of
                # training steps, which does mean that roll-out fine-tuning might only see
                # a small fraction of the entire dataset.
                # Note: We deviate from the original implementation here, in that we do
                # interrupt the training at the end of the dataset. This is because we want
                # to run evaluation at the end of each epoch.
                raise StopIteration

        return samples
