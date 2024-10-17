import torch
import numpy as np
from dataloaders.data_utils.signal_encoding import quantize_encode
import copy


class DuplicateEvent:
    """
    Adds a rescaled version of the event to the empty part of the trace after the event.
    Event position and empty space are determined from a detection.
    Detections can be generated for example with :py:class:`~seisbench.generate.labeling.DetectionLabeller`.

    This implementation is modelled after the `implementation for EQTransformer <https://github.com/smousavi05/EQTransformer/blob/98676017f971efbb6f4475f42e415c3868d00c03/EQTransformer/core/EqT_utils.py#L255>`_.

    .. warning::
        This augmentation does **not** modify the metadata, as representing multiple picks of
        the same type is currently not supported. Workflows should therefore always first generate
        labels from metadata and then pass the labels in the key `label_keys`. These keys are automatically
        adjusted by addition of the labels.

    .. warning::
        This implementation currently has strict shape requirements:

        - (1, samples) for detection
        - (channels, samples) for data
        - (labels, samples) for labels

    :param inv_scale: The scale factor is defined by as 1/u, where u is uniform.
                      `inv_scale` defines the minimum and maximum values for u.
                      Defaults to (1, 10), e.g., scaling by factor 1 to 1/10.
    :param detection_key: Key to read detection from.
                          If key is a tuple, detection will be read from the first key and written to the second one.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key
                to read from and the second one the key to write to.
    :param label_keys: Keys for the label columns.
                       Labels of the original and duplicate events will be added and capped at 1.
                       Note that this will lead to invalid noise traces.
                       Value can either be a single key specification or a list of key specifications.
                       Each key specification is either a string, for identical input and output keys,
                       or as a tuple of two strings, input and output keys.
                       Defaults to None.
    """

    def __init__(
            self, inv_scale=(1, 10), detection_key="detections", key="X", label_keys=None
    ):
        if isinstance(detection_key, str):
            self.detection_key = (detection_key, detection_key)
        else:
            self.detection_key = detection_key

        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        # Single key
        if not isinstance(label_keys, list):
            if label_keys is None:
                label_keys = []
            else:
                label_keys = [label_keys]

        # Resolve identical input and output keys
        self.label_keys = []
        for key in label_keys:
            if isinstance(key, tuple):
                self.label_keys.append(key)
            else:
                self.label_keys.append((key, key))

        self.inv_scale = inv_scale

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        detection, _ = state_dict[self.detection_key[0]]
        detection_mask = detection[0] > 0.5

        if detection.shape[-1] != x.shape[-1]:
            raise ValueError("Number of samples in trace and detection disagree.")

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        if detection_mask.any():
            n_samples = x.shape[-1]
            event_samples = np.arange(n_samples)[detection_mask]
            event_start, event_end = np.min(event_samples), np.max(event_samples) + 1

            if event_end + 20 < n_samples:
                second_start = np.random.randint(event_end + 20, n_samples)
                scale = 1 / np.random.uniform(*self.inv_scale)

                if self.key[0] != self.key[1]:
                    # Avoid inplace modification if input and output keys differ
                    x = x.copy()

                space = min(event_end - event_start, n_samples - second_start)
                x[:, second_start: second_start + space] += (
                        x[:, event_start: event_start + space] * scale
                )

                shift = second_start - event_start

                for label_key in self.label_keys + [self.detection_key]:
                    y, metadata = state_dict[label_key[0]]
                    if y.shape[-1] != n_samples:
                        raise ValueError(
                            f"Number of samples disagree between trace and label key '{label_key[0]}'."
                        )

                    if label_key[0] != label_key[1]:
                        metadata = copy.deepcopy(metadata)
                        y = y.copy()

                    y[:, shift:] += y[:, :-shift]
                    y = np.minimum(y, 1)
                    state_dict[label_key[1]] = (y, metadata)
        else:
            # Copy entries
            for label_key in self.label_keys + [self.detection_key]:
                y, metadata = state_dict[label_key[0]]
                if label_key[0] != label_key[1]:
                    metadata = copy.deepcopy(metadata)
                    y = y.copy()
                state_dict[label_key[1]] = (y, metadata)

        state_dict[self.key[1]] = (x, metadata)


class QuantizeAugmentation:
    def __init__(self, key='X', bits=8):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.bits = bits

    def __call__(self, state_dict):
        if self.bits > 0:
            x, metadata = state_dict[self.key[0]]
            x = quantize_encode(torch.from_numpy(x), bits=self.bits)
            x = x.numpy()
            state_dict[self.key[1]] = (x, metadata)


class FillMissingComponents:
    def __init__(self, key="X", ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if isinstance(x, list):
            x = [self._fill_missing_component(y) for y in x]
        else:
            x = self._fill_missing_component(x)

        state_dict[self.key[1]] = (x, metadata)

    def _fill_missing_component(self, x):
        std_devs = np.std(x, axis=1)

        # Find rows with zero standard deviation
        zero_std_rows = np.where(std_devs == 0)[0]
        non_zero_std_rows = np.where(std_devs != 0)[0]

        # If all rows or zero rows have zero standard deviation, return as is
        if len(zero_std_rows) == 0 or len(zero_std_rows) == x.shape[0]:
            return x

        # Otherwise, randomly replace zero-std rows with non-zero std rows
        for row_idx in zero_std_rows:
            replacement_row = np.random.choice(non_zero_std_rows)
            x[row_idx] = x[replacement_row]
        return x


class FilterZChannel:
    def __init__(self, key="X", ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        state_dict[self.key[1]] = (x[0], metadata)


class AutoregressiveShift:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        state_dict[self.key[0]] = (x[:-1], metadata)
        state_dict[self.key[1]] = (x[1:], metadata)


class RandomMask:
    def __init__(self, key=('X', 'y'), p=0.5):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = p

    def __call__(self, state_dict):
        r = np.random.randint(0, 4)

        x, metadata = state_dict[self.key[0]]
        seq_len = x.shape[0]
        if r == 0:
            # zero out random elements with probability p. Channels are treated independently
            mask = np.random.choice([0, 1], size=x.shape, p=[self.p, 1.0 - self.p]).astype(np.float32)
            masked = x * mask
        elif r == 1:
            # zero out random samples with probability p. Zero out all three channels of the sample
            mask = np.random.choice([0, 1], size=seq_len, p=[self.p, 1.0 - self.p]).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            masked = x * mask
            mask = np.ones_like(x) * mask
        elif r == 2:
            # zero out a random block of the trace. Block length is 25% of trace length
            block_length = int(seq_len // 4)
            start_idx = np.random.randint(0, seq_len - block_length + 1)
            masked = x.copy()
            mask = np.ones_like(x)
            masked[start_idx:start_idx + block_length, :] = 0
            mask[start_idx:start_idx + block_length, :] = 0
        elif r == 3:
            # zero out three random blocks of 5% sequence length each
            block_length = int(seq_len * 0.05)
            masked = x.copy()
            mask = np.ones_like(x)
            for i in range(10):
                start_idx = np.random.randint(0, seq_len - block_length + 1)
                masked[start_idx:start_idx + block_length, :] = 0
                mask[start_idx:start_idx + block_length, :] = 0
        #state_dict['mask'] = np.invert(mask.astype(bool))
        state_dict[self.key[0]] = ((masked, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (x, metadata)


class TransposeLabels:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        # x, meta_x = state_dict[self.key[0]]
        y, meta_y = state_dict[self.key[1]]
        # state_dict[self.key[0]] = (np.transpose(x, (0, 2, 1)), meta_x)
        state_dict[self.key[1]] = (y.T, meta_y)


class TransposeSeqChannels:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, meta_x = state_dict[self.key[0]]
        state_dict[self.key[0]] = (np.transpose(x, (1, 0)), meta_x)


class SquashAugmentation:
    def __init__(self, key=('X', 'y'), squash_func: str = 'sqrt'):
        """
        Squash the trace with a squashing function. The squashing function is either 'sqrt' or 'log'
        :param key:
        :param squash_func: one of ['sqrt', 'log']
        """
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.squash_func = squash_func

    def __call__(self, state_dict):
        """
        Squash the trace with a squashing function. The state_dict['X'] contains the input sequence of
        dimensions [channels, seq_len]. The state_dict['y'] contains the ground truth. For pre-training,
        this is also a sequence of dimensions [channels, seq_len].
        :param state_dict:
        :return:
        """
        x, metadata = state_dict[self.key[0]]

        # remove mean
        x = x - np.mean(x, axis=-1, keepdims=True)

        # squash
        if self.squash_func == 'sqrt':
            x = np.sqrt(np.abs(x)) * np.sign(x)
        elif self.squash_func == 'log':
            print('log')
            x = np.log(np.abs(x) + 1) * np.sign(x)
            if np.sum(np.isnan(x)) > 0:
                print('WARNING: Nan')
                #print(self.squash_func)

        state_dict[self.key[0]] = (x, metadata)
