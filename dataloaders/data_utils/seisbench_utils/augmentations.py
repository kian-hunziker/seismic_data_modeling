import torch
import numpy as np
from dataloaders.data_utils.signal_encoding import quantize_encode


class QuantizeAugmentation:
    def __init__(self, key='X', bits=8):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.bits = bits

    def __call__(self, state_dict):
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
