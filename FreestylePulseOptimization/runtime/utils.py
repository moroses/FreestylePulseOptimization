from re import S
import numpy as np
from numpy import typing as npt
import pandas as pd
import argparse
from pathlib import Path
from qiskit.result.mitigation.local_readout_mitigator import LocalReadoutMitigator
from qiskit.result.utils import marginal_counts
from qiskit.quantum_info import Operator
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.circuit import QuantumCircuit
from qiskit.pulse.channels import PulseChannel, DriveChannel, ControlChannel
from qiskit.pulse.schedule import ScheduleBlock

from ..pulses import QubitSpecification
from .. import (
    CI_Matrix,
)
from typing import Callable, Optional, Any, TypeAlias, TypeVar, Literal
from collections.abc import Sequence, Mapping, Iterable
from collections import defaultdict
from operator import attrgetter, itemgetter, mul
from functools import partial, reduce
from dataclasses import dataclass, field
from ..utils import PaddingType, TimingConstraints


# TODO Clean this bit ->
from ..utils import CI_MatrixSolution


@dataclass
class HWCI_MatrixSolutionRuntime(CI_MatrixSolution):
    quasi_dists_raw: Sequence[Mapping[str, Sequence[QuasiDistribution]]]
    quasi_dists_join: Sequence[Mapping[str, QuasiDistribution]]
    energy_error: Sequence[float]


@dataclass
class HWCI_MatrixSolutionRuntimeLogical(HWCI_MatrixSolutionRuntime):
    phys_to_logical: int


@dataclass
class HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(CI_MatrixSolution):
    phys_to_logical: int
    raw_measurements: Sequence[pd.DataFrame]
    joined_measurements: Sequence[Mapping[str, Mapping[str, int]]]
    energy_error: Sequence[float]


# <-


@dataclass(frozen=True, slots=True)
class Parameter:
    Nt: int
    padding: PaddingType
    total_shots: int
    distance: float
    phys_to_logical: int = 1
    additional_data: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self: "Parameter") -> str:
        return f"Nt-{self.Nt}|padding-{self.padding.value}|total_shots-{self.total_shots}|distance-{self.distance}|phys_to_logical-{self.phys_to_logical}|additional_data-{self.additional_data}"

    def todict(self: "Parameter") -> Mapping[str, Any]:
        return {
            "Nt": self.Nt,
            "padding": self.padding.value,
            "total-shots": self.total_shots,
            "distance": self.distance,
            "phys-to-logical": self.phys_to_logical,
            "additional-data": self.additional_data,
        }

    @classmethod
    def fromdict(cls: type["Parameter"], d: Mapping[str, Any]) -> "Parameter":
        Nt = int(d["Nt"])
        padding = PaddingType(int(d["padding"]))
        total_shots = int(d["total-shots"])
        distance = float(d["distance"])
        phys_to_logical = int(d.get("phys-to-logical", 1))
        additional_data = d.get("additional-data", {})
        return Parameter(
            Nt=Nt,
            padding=padding,
            total_shots=total_shots,
            distance=distance,
            phys_to_logical=phys_to_logical,
            additional_data=additional_data,
        )

    @classmethod
    def fromstr(cls: type["Parameter"], s: str) -> "Parameter":
        return cls.fromdict(
            map_keys(
                lambda ss: ss.replace("_", "-"),
                dict(map(lambda ss: ss.split("-", maxsplit=2), s.split("|"))),
            )
        )


class ParseKwargs(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = ...,
    ) -> None:
        setattr(namespace, self.dest, dict())
        if values is None:
            return
        for value in values:
            key, val = value.split("=")
            getattr(namespace, self.dest)[key] = val


def int_div_mod(n: int, m: int) -> int:
    if n == 0 or m == 0:
        raise ValueError("0? What do I do with it?")
    return n // m + int(n % m != 0)


def hex_key_to_binary(n_qubits: int) -> Callable[[str], str]:
    def _internal(s: str) -> str:
        n = int(s, 16)
        return f"{n:0{n_qubits}b}"

    return _internal


def to_binary_keys(n_qubits: int) -> Callable[[QuasiDistribution], Mapping[str, float]]:
    def _internal(quasi_dist: QuasiDistribution) -> Mapping[str, float]:
        return quasi_dist.binary_probabilities(num_bits=n_qubits)

    return _internal


def mitigate_error(
    em: LocalReadoutMitigator,
) -> Callable[[Mapping[str, float]], tuple[float, float]]:
    def _internal(counts: Mapping[str, float]) -> tuple[float, float]:
        return em.expectation_value(counts)

    return _internal


def marginalize_counts(
    qubits: Sequence[int],
) -> Callable[[Mapping[str, float]], Mapping[str, float]]:
    def _internal(counts: Mapping[str, float]) -> Mapping[str, float]:
        return marginal_counts(counts, indices=qubits)

    return _internal


def map_values(f: Callable, d: Mapping) -> Mapping:
    return {k: f(v) for k, v in d.items()}


def map_keys(f: Callable, d: Mapping) -> Mapping:
    return {f(k): v for k, v in d.items()}


def prob_to_count(n_shots: int) -> Callable[[Mapping], Mapping]:
    return partial(map_values, partial(mul, n_shots))


def run_on_index(f: Callable, index: int) -> Callable[[tuple], tuple]:
    """
    run_in_index(f, index) - applies f to a single site of a tuple.

    Return a function that only applies to a single site of a tuple.

    Parameters
    ----------
    f : Callable
        The function to apply.
    index : int
        The index to apply `f` on.

    Returns
    -------
    Callable[[tuple], tuple]
        A callable that applies `f` on `index` of a given tuple
    """

    def _internal(t: tuple) -> tuple:
        return *t[:index], f(t[index]), *t[index + 1 :]

    return _internal


def compose(*fs: Callable) -> Callable:
    """
    composes a given list of functions.

    Chains functions in apply one after the other.

    Parameters
    ----------
    fs : Callable
        List of functions

    Returns
    -------
    Callable
        The chain function.
    """
    joiner = lambda f, g: lambda *a, **kw: f(g(*a, **kw))
    return reduce(joiner, fs)


def combine_values(ds: Iterable[Mapping[str, int]]) -> Mapping[str, int]:
    """
    combines the values of a dictionary.

    Sums the values of the dictionary according to their key.

    Parameters
    ----------
    ds : Iterable[Mapping[str, int]]
        List of dictionaries to join.

    Returns
    -------
    Mapping[str, int]
        The combined dictionary.
    """
    ret: dict[str, int] = defaultdict(lambda: 0)
    for d in ds:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


def matrix_ev(
    ci_matrix: CI_Matrix,
) -> Callable[[Mapping[str, tuple[float, float]]], tuple[float, float]]:
    """
    calculate the matrix expectation value for a given CI

    Returns a function to calculate the expectation value of a given CI.

    Parameters
    ----------
    ci_matrix : CI_Matrix
        The CI configuration.

    Returns
    -------
    Callable[[Mapping[str, tuple[float, float]]], tuple[float, float]]
        A function to convert a mapping of {"OPERATOR": (MEASUREMENT, MEASUREMENT_ERROR), ...} to an expectation value of CI.
    """
    # dim = 2 ** ci_matrix.n_dim
    dim = ci_matrix.n_dim
    I = np.eye(dim, dtype=complex)
    base_energy = (
        ci_matrix.nuclear_repulsion_energy + (ci_matrix.matrix @ I).trace().real / dim
    )

    def _internal(values: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
        energy = base_energy
        error2 = 0
        for oper, (measurement, measurement_error) in values.items():
            O = Operator.from_label(oper.upper()).to_matrix()
            o_coeff = (ci_matrix.matrix @ O).trace().real / dim
            energy += measurement * o_coeff
            error2 += (measurement_error * o_coeff) ** 2
        error = np.sqrt(error2)
        return energy, error

    return _internal


def _x_measure(qc: QuantumCircuit, qubit: int) -> None:
    qc.h(qubit)


def _y_measure(qc: QuantumCircuit, qubit: int) -> None:
    qc.h(qubit)
    qc.s(qubit)


def _z_measure(qc: QuantumCircuit, qubit: int) -> None:
    pass


MEASUREMENT_FUNCTION: TypeAlias = Callable[[QuantumCircuit, int], None]


def get_relevant_qubits(qubits: Sequence[int], pauli_string: str) -> Sequence[int]:
    return [qubit for qubit, p in zip(qubits, pauli_string) if p != "I"]


def measure_pauli_string(
    qubits: Sequence[int], base_circuit: QuantumCircuit, pauli_strings: Sequence[str]
) -> Sequence[QuantumCircuit]:
    ret_circuits: list[QuantumCircuit] = []
    pre_measure_gates: Mapping[str, MEASUREMENT_FUNCTION] = {
        "X": _x_measure,
        "Y": _y_measure,
        "Z": _z_measure,
    }

    for pauli_string in pauli_strings:
        curr = base_circuit.copy()
        curr.metadata["measure"] = pauli_string
        curr.barrier()
        measure_qubits: list[int] = []
        for i, p in enumerate(pauli_string):
            if p == "I":
                continue
            qubit = qubits[i]
            measure_qubits.append(qubit)
            pre_measure_gates[p](curr, qubit)
        curr.barrier()
        if len(measure_qubits) == 0:
            continue
        for qubit in measure_qubits:
            curr.measure(qubit, qubit)
        ret_circuits.append(curr)

    return ret_circuits


def pauli_strings(
    strings: Mapping[str, float], base: float = 0.0
) -> Callable[[Mapping[str, tuple[float, float]]], tuple[float, float]]:
    base += sum([v for k, v in strings.items() if all(map(lambda s: s == "I", k))])

    def _internal(values: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
        energy = base
        error2 = 0

        for k, (val, err) in values.items():
            coeff = strings.get(k, 0.0)
            energy += coeff * val
            error2 += (coeff * err) ** 2

        error = np.sqrt(error2)
        return np.real(energy), np.real(error)

    return _internal


CHANNEL_TYPES: TypeAlias = Literal["d"] | Literal["u"] | Literal["a"]


def get_channel_names(
    qubit_spec: Sequence[QubitSpecification], *, channel_types: CHANNEL_TYPES = "a"
) -> Sequence[str]:
    qubit_inds = [q.index for q in qubit_spec]
    drive_channels_str = [f"d{i}" for i in qubit_inds]

    if channel_types == "d":
        return drive_channels_str

    control_channels = []
    for q in qubit_spec:
        if q.control_channels is None:
            continue
        for oq, c_ind in q.control_channels.items():
            if oq in qubit_inds and c_ind not in control_channels:
                control_channels.append(c_ind)
    control_channels_str = [f"u{i}" for i in control_channels]

    if channel_types == "u":
        return control_channels_str

    return drive_channels_str + control_channels_str


def convert_parameters_to_channels(
    phys_to_logical: int,
    parameter_names: Sequence[str],
    padding_type: Optional[PaddingType] = None,
    timing_constraints: Optional[TimingConstraints] = None,
) -> Callable[[Sequence[complex]], Mapping[str, npt.NDArray[np.complex_]]]:
    # pattern = re.compile(r"^(?P<channel>\w+)(?P<qubit>\d+)_(?P<index>\d+)$")
    if padding_type is None:
        padding_type = PaddingType.NO
    if timing_constraints is None:
        timing_constraints = STANDARD_TIMING

    def _internal(values: Sequence[complex]) -> Mapping[str, npt.NDArray[complex]]:
        assert len(values) == len(
            parameter_names
        ), f"Parameter names and values must have the same shape ({len(parameter_names)=}, {len(values)=})"
        ret: dict[str, list[complex]] = {}
        for key, val in zip(parameter_names, values):
            channel_qubit, index = key.split("_", maxsplit=2)
            if channel_qubit not in ret:
                ret[channel_qubit] = []
            assert int(index) * phys_to_logical == len(
                ret[channel_qubit]
            ), f"Something went wrong at {key=}"
            ret[channel_qubit].extend([val] * phys_to_logical)
        np_ret: dict[str, npt.NDArray[complex]] = {}
        for key in ret.keys():
            np_ret[key] = padding_type.pad(ret[key], timing_constraints)
        return np_ret

    return _internal


def get_channel(name: str) -> PulseChannel:
    channel = name[:1]
    qubit = int(name[1:])
    match channel:
        case "d":
            return DriveChannel(qubit)
        case "u":
            return ControlChannel(qubit)
        case _:
            raise AttributeError(f"I do not know {name=}!")


def convert_channels_to_scheduleblock(
    pulses: Mapping[str, Sequence[complex]]
) -> ScheduleBlock:
    with build(name="VQE") as schd:
        for key, samples in pulses.items():
            channel = get_channel(key)
            waveform = Waveform(
                samples=np.array(samples), epsilon=0.2, limit_amplitude=False
            )
            play(waveform, channel)
    return schd
