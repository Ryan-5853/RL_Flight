from __future__ import annotations

import unittest

import torch

from flight_deploy.action import ResidualActionTransform
from flight_deploy.history import UniformHistoryBuffer


class HistoryTests(unittest.TestCase):
    def test_reset_repeats_first_frame_and_flatten_is_oldest_first(self) -> None:
        history = UniformHistoryBuffer(1, 2, 3)
        first = torch.tensor([[1.0, 2.0]])
        history.reset(first)
        torch.testing.assert_close(
            history.observation(),
            torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]),
        )
        history.append(torch.tensor([[3.0, 4.0]]))
        actual = history.append(torch.tensor([[5.0, 6.0]]))
        torch.testing.assert_close(
            actual,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

    def test_stride_selects_sparse_frames(self) -> None:
        history = UniformHistoryBuffer(1, 1, 3, stride_steps=2)
        history.reset(torch.tensor([[0.0]]))
        for value in range(1, 5):
            actual = history.append(torch.tensor([[float(value)]]))
        torch.testing.assert_close(actual, torch.tensor([[0.0, 2.0, 4.0]]))


class ActionTests(unittest.TestCase):
    def test_residual_transform(self) -> None:
        transform = ResidualActionTransform(
            (0.53523, 0.0, 0.0, 0.0),
            (0.30, 1.0, 1.0, 1.0),
        )
        actual = transform(torch.tensor([[1.0, -0.5, 0.25, 2.0]]))
        torch.testing.assert_close(
            actual,
            torch.tensor([[0.83523, -0.5, 0.25, 1.0]]),
        )
