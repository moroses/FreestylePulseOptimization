"""Module for the base hardware based CI matrix solution."""
from collections.abc import Mapping, Sequence
import numpy.typing as npt


from dataclasses import dataclass

from .utils import (
    TimingConstraints,
    CI_MatrixSolution,
    CI_Matrix,
    OptimizationTime,
    build_schedule,
    PaddingType,
)
from .utils import _complex_to_real, _real_to_complex
from .pulses import QubitSpecification


@dataclass
class HWCI_MatrixSolution(CI_MatrixSolution):
    """Base class for hardware based CI matrix solution."""
    raw_data_trajectory: Sequence[Mapping[str, Sequence[Mapping[str, int]]]]
    """Raw measurements of each optimization step"""
    energy_error: Sequence[float]
    """Calculated measurement error of the energy"""
    assignment_matrix: Sequence[npt.NDArray] | npt.NDArray
    """The assignment matrices used to preform error mitigation"""
