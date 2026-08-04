from __future__ import annotations

import unittest

import torch

from flight_train.angular_acceleration_teacher import (
    IncrementalAngularAccelerationTeacher,
)


class AngularAccelerationTeacherTests(unittest.TestCase):
    def test_incremental_teacher_uses_current_frame_and_previous_action(self):
        teacher = IncrementalAngularAccelerationTeacher(
            inverse_effectiveness=torch.eye(3),
            command_limit=torch.tensor([4.0, 4.0, 2.0]),
            correction_gain=torch.tensor([0.1, 0.2, 0.3]),
            base_observation_dim=21,
        )
        observation = torch.zeros(2, 42)
        observation[:, -21:-18] = torch.tensor([0.5, -0.25, 0.5])
        observation[:, -18:-15] = torch.tensor([0.25, 0.0, -0.5])
        observation[:, -3:] = torch.tensor([0.1, -0.2, 0.3])

        action, hidden = teacher.forward_step(observation)

        expected = torch.tensor([0.2, -0.4, 0.9]).expand_as(action)
        torch.testing.assert_close(action, expected)
        self.assertIsNone(hidden)


if __name__ == "__main__":
    unittest.main()
