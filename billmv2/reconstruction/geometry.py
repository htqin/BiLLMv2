"""Streaming teacher/student reconstruction state."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SequentialStreams:
    """Hold the current full-precision and quantized prefix activations."""

    teacher_inputs: Tensor
    student_inputs: Tensor

    def advance(self, teacher_outputs: Tensor, student_outputs: Tensor) -> None:
        """Advance independent teacher and student streams."""

        if teacher_outputs.shape != student_outputs.shape:
            raise ValueError("teacher and student stream shapes must match")
        self.teacher_inputs = teacher_outputs
        self.student_inputs = student_outputs
