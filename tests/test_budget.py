import numpy as np
import torch

from src.utils.budget import tensor_to_u8_round, tensor_to_u8_trunc_v1, to_model_scale, to_standard_scale


def test_scale_conversions():
    assert abs(to_model_scale(4) - 8 / 255) < 1e-15
    assert abs(to_model_scale(0.5) - 1 / 255) < 1e-15
    assert abs(to_standard_scale(8 / 255) - 4) < 1e-12


def test_round_vs_trunc():
    # pixel values (0..255 space) mapped to [-1, 1]: x = v / 127.5 - 1
    v = np.array([126.99999, 127.4, 0.0, 255.0], dtype=np.float32)
    x = torch.from_numpy(v / 127.5 - 1.0).view(1, 1, 1, 4).expand(1, 3, 1, 4).clone()
    r = tensor_to_u8_round(x)[0, :, 0]
    t = tensor_to_u8_trunc_v1(x)[0, :, 0]
    assert list(r) == [127, 127, 0, 255]
    assert t[0] == 126 and t[1] == 127  # v1 truncation off-by-one on 126.99999
