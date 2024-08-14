from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union
from torch import Tensor


@dataclass
class InferenceParams:
    """
    Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context during inference.

    Increase seqlen_offset after each inference step
    key_value_memory_dict maps layer indices to (conv_state, ssm_state):
    conv_state, ssm_state = inference_params.key_value_memory_dict[layer_idx]
    """

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
            