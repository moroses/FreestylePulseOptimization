#!/bin/env python3
# Author: Mor M. Roses

from __future__ import annotations
from collections.abc import Sequence, Iterable, Iterator, Mapping
import enum
from functools import reduce
from itertools import product
from operator import itemgetter
from typing import Any, Callable, Optional, TypeVar
from pathlib import Path
import json
import dataclasses
import qiskit
import qiskit.providers
import qiskit.providers.backend
import qiskit.pulse
import qiskit.pulse.channels
import qiskit.pulse.library
import qiskit.quantum_info
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.statevector import Statevector
import qiskit_dynamics as qd
import datetime
import time
import numpy as np
import numpy.typing as npt
import scipy
import scipy.optimize
import re
import enum
from . import pulses

# TODO this needs to move to another file
# something like CI solver


def pauli_composer(
    N_qubits: int, *, factor: float = 1
) -> Callable[[Mapping[str, float | complex]], npt.NDArray[np.complex_ | np.float_]]:
    pauli_map = {
        "I": np.eye(2, dtype=complex) * factor,
        "X": np.array([[0, 1], [1, 0]], dtype=complex) * factor,
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex) * factor,
        "Z": np.diag([1, -1]).astype(complex) * factor,
    }
    dim = 2**N_qubits

    def _internal(
        pauli_strings: Mapping[str, complex | float]
    ) -> npt.NDArray[np.complex_ | npt.float_]:
        r = np.zeros((dim, dim), dtype=complex)
        for pstr, coeff in pauli_strings.items():
            M = np.array(1)
            for p in pstr.upper():
                M = np.kron(M, pauli_map[p])
            r += coeff * M
        return np.real_if_close(r)

    return _internal


def pauli_decomposer(
    N_qubits: int, *, factor: float = 1
) -> Callable[[npt.NDArray[np.complex_ | np.float_]], Mapping[str, float | complex]]:
    pauli_labels = [
        "I",
        "X",
        "Y",
        "Z",
    ]
    pauli_operators = [
        np.eye(2, dtype=complex) * factor,
        np.array([[0, 1], [1, 0]], dtype=complex) * factor,
        np.array([[0, -1j], [1j, 0]], dtype=complex) * factor,
        np.diag([1, -1]).astype(complex) * factor,
    ]
    dim = 2**N_qubits

    def _internal(
        matrix: npt.NDArray[np.complex_ | np.float_],
    ) -> Mapping[str, float | complex]:
        ret: dict[str, complex | float] = {}

        for elm in product(zip(pauli_labels, pauli_operators), repeat=N_qubits):
            name = "".join(map(itemgetter(0), elm))
            operator = reduce(np.kron, map(itemgetter(1), elm), np.array(1))
            coeff = (matrix @ operator).trace() / dim
            operator = None

            ret[name] = coeff

        return ret

    return _internal


@dataclasses.dataclass(frozen=True, order=True)
class CI_Matrix:
    mol_name: str
    distance: float
    n_dim: int
    nuclear_repulsion_energy: float
    ci_ground_state_energy: float
    matrix: np.ndarray


def load_ci_matrix_from_folder(
    base_folder: Path | str, mol_dist: str, n_dim: int
) -> CI_Matrix:
    base_folder = Path(base_folder)
    bdir = base_folder / f"{mol_dist}"
    json_file = bdir / "CI_result.json"
    matrix_file = bdir / "CI_matrices" / f"{mol_dist}_cimat__{n_dim}.out"
    assert json_file.is_file(), f"No json file for {bdir}!"
    assert matrix_file.is_file(), f"No matrix file for {bdir}, {n_dim}!"
    json_data = json.loads(json_file.read_text(encoding="utf8"))
    mol_name, dist_str = mol_dist.split("_", maxsplit=2)
    distance = float(dist_str)
    nuclear_repulsion_energy = json_data["nuclear_repulsion_energy"]
    ci_ground_state_energy = json_data["ci_ground_state_energy"]
    # matrix = np.zeros((n_dim, n_dim), dtype=float)
    matrix_lines = matrix_file.read_text(encoding="utf8")
    matrix = np.asarray(
        [
            [float(elm) for elm in line.split(" ")]
            for line in matrix_lines.split("\n")
            if line != ""
        ],
        dtype=float,
    )
    return CI_Matrix(
        mol_name=mol_name,
        distance=distance,
        n_dim=n_dim,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        ci_ground_state_energy=ci_ground_state_energy,
        matrix=matrix,
    )


def load_all_ci_matrices_from_folder(
    base_dir: str | Path, n_dim: int
) -> Sequence[CI_Matrix]:
    ret: list[CI_Matrix] = []
    pbase_dir = Path(base_dir)
    pattern = re.compile(r"\w+_[\d.]+")
    for ci_file in pbase_dir.iterdir():
        if not ci_file.is_dir():
            continue
        if re.match(pattern, ci_file.name) is None:
            continue
        ret.append(load_ci_matrix_from_folder(pbase_dir, ci_file.name, n_dim))
    return ret


# TODO change file name to ci_matrix


@dataclasses.dataclass(kw_only=True)
class CI_MatrixSolution:
    ci_matrix: CI_Matrix
    dt: float
    success: bool
    Nt: int
    parameters_trajectory: Sequence[Sequence[complex]]
    parameter_names: Sequence[str]
    energy_trajectory: Sequence[float]
    qubit_spec: Sequence[pulses.QubitSpecification]
    qubit_noise: Optional[Sequence[pulses.QubitNoiseParameters]] = None
    additional_data: Optional[Mapping[str, Any]] = None


def random_complex(*shape: int) -> np.ndarray:
    r = np.random.random(*shape)
    theta = np.random.random(*shape) * 2 * np.pi
    rr = np.sqrt(r)
    return rr * np.cos(theta) + rr * np.sin(theta) * 1j


def _get_channel_by_name(channel_name: str) -> qiskit.pulse.channels.PulseChannel:
    c_type = channel_name[:1]
    c_index = int(channel_name[1:])
    match c_type:
        case "d":
            return qiskit.pulse.DriveChannel(c_index)
        case "u":
            return qiskit.pulse.ControlChannel(c_index)
        case _:
            raise AttributeError(f"Got a bad channel! {channel_name=}")


def build_schedule(
    parameter_names: Sequence[str],
    parameters: Sequence[complex],
    padding_type: Optional[PaddingType] = None,
    timing_constraints: Optional[TimingConstraints] = None,
    phys_to_logical: Optional[int] = None,
) -> qiskit.pulse.ScheduleBlock:
    assert len(parameter_names) == len(
        parameters
    ), f"Got bad arguments! {len(parameter_names)=}, {len(parameters)=}"
    if phys_to_logical is None:
        phys_to_logical = 1
    waveform_map: dict[str, list[complex]] = {}
    for name, value in zip(parameter_names, parameters):
        n, i = name.split("_", maxsplit=2)
        if n not in waveform_map:
            waveform_map[n] = []
        assert (
            len(waveform_map[n]) == int(i) * phys_to_logical
        ), f"Bad counting somehow, {name=}"
        waveform_map[n].extend(
            [
                value,
            ]
            * phys_to_logical
        )
    if padding_type is None:
        padding_type = PaddingType.NO
    if timing_constraints is None:
        timing_constraints = STANDARD_TIMING

    with qiskit.pulse.build() as schedule:
        for name, waveform in waveform_map.items():
            wf = qiskit.pulse.library.Waveform(
                samples=padding_type.pad(waveform, timing_constraints),
                epsilon=0.2,
                limit_amplitude=False,
            )
            qiskit.pulse.play(wf, _get_channel_by_name(name))
    return schedule


QISKIT_STATE = Statevector | DensityMatrix


def _real_to_complex(z: npt.NDArray[float]) -> npt.NDArray[complex]:
    return z[: len(z) // 2] + z[len(z) // 2 :] * 1j


def _complex_to_real(z: npt.NDArray[complex]) -> npt.NDArray[float]:
    return np.concatenate((np.real(z), np.imag(z)))


def _get_single_direction(
    qubit_specification: Sequence[pulses.QubitSpecification],
    real_to_sim_map: Mapping[int, int],
    control_channels: Mapping[tuple[int, int], int],
) -> Sequence[str]:
    single_channels: dict[tuple[int, int], int] = {}
    for qubit in qubit_specification:
        if qubit.control_channels is None:
            continue
        for other, channel in qubit.control_channels.items():
            if other not in real_to_sim_map:
                continue
            nother = real_to_sim_map[other]
            nqubit = real_to_sim_map[qubit.index]
            if (nqubit, nother) in single_channels:
                continue
            single_channels[(nqubit, nother)] = single_channels[(nother, nqubit)] = (
                channel
            )
    return [f"u{c}" for c in np.unique([*single_channels.values()])]


@dataclasses.dataclass(frozen=True)
class OptimizationTime:
    prep_time: float
    optimization_time: float
    energy_time: float


def gaussian_build_schedule(
    Np: int,
    default_values: Mapping[str, float | complex],
    parameter_names: Sequence[str],
) -> Callable[[Sequence[float | complex]], qiskit.pulse.ScheduleBlock]:
    def _internal(parameters: Sequence[float | complex]) -> qiskit.pulse.ScheduleBlock:
        pulse_channel_map: dict[
            qiskit.pulse.channels.PulseChannel, qiskit.pulse.library.Gaussian
        ] = {}

        for i in range(0, len(parameters), Np):
            cur_dict = default_values.copy()
            channel_name, _ = parameter_names[i].split("_", maxsplit=1)
            channel = _get_channel_by_name(channel_name=channel_name)
            for j in range(Np):
                _, p_name = parameter_names[i + j].split("_", maxsplit=1)
                cur_dict[p_name] = parameters[i + j]
            if "amp_r" in cur_dict:
                cur_dict["amp"] = complex(cur_dict["amp_r"], cur_dict["amp_i"])
                del cur_dict["amp_r"]
                del cur_dict["amp_i"]
            pulse_channel_map[channel] = qiskit.pulse.library.Gaussian(
                **cur_dict, limit_amplitude=False
            )
        with qiskit.pulse.build() as schd:
            for channel, pulse in pulse_channel_map.items():
                qiskit.pulse.play(pulse, channel)
        return schd

    return _internal


def optimize_gaussian(
    dt: float,
    qubit_specification: Sequence[pulses.QubitSpecification],
    ci_matrix: CI_Matrix,
    default_values: Mapping[str, float],
    init_state: Mapping[str, float],
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[pulses.QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    single_connection: bool = False,
    **kwargs,
) -> tuple[CI_MatrixSolution, OptimizationTime, Any]:
    # Prep ->
    start_time = time.time()

    solver, q_map, ch_map, signal_maker = pulses.generate_solver(
        qubit_specifications=qubit_specification,
        dt=dt,
        cross_talk=cross_talk,
        qubit_noise_model=qubit_noise_model,
    )

    all_channels: Sequence[str] = [
        *[f"d{q}" for q in q_map.values()],
        *[f"u{c}" for c in ch_map.values()],
    ]

    # base_parameters = {
    #         "duration": Nt,
    #         "amp": 1.0,
    #         "sigma": 0.5,
    #        }

    parameter_names: Sequence[str] = [
        f"d{q}_{i}" for q in q_map.values() for i in init_state.keys()
    ]
    if single_connection:
        single_channels = _get_single_direction(
            qubit_specification=qubit_specification,
            real_to_sim_map=q_map,
            control_channels=ch_map,
        )
        parameter_names = [
            *parameter_names,
            *[f"{c}_{i}" for c in single_channels for i in init_state.keys()],
        ]
    else:
        parameter_names = [  # TODO consider some smart use of channels
            *parameter_names,
            *[f"u{c}_{i}" for c in ch_map.values() for i in init_state.keys()],
        ]

    p0 = [init_state[c.split("_", maxsplit=1)[1]] for c in parameter_names]

    if t_span is None:
        if "duration" in default_values:
            t_span = (0, dt * default_values["duration"])
        else:
            raise ArgumentError(f"How should I know for how long to run?!?")

    if y0 is None:
        if qubit_noise_model is None:
            y0 = Statevector.from_label("0" * len(qubit_specification))
        else:
            y0 = DensityMatrix.from_label("0" * len(qubit_specification))

    pulse_builder = gaussian_build_schedule(
        Np=len(init_state.keys()),
        default_values=default_values,
        parameter_names=parameter_names,
    )

    def _cost(parameters: Sequence[float]) -> float:
        schedule = pulse_builder(parameters)
        schedule = pulses.pad_schedule(schedule, all_channels)
        result = solver.solve(t_span=t_span, y0=y0, signals=schedule)
        last_state: QISKIT_STATE = result.y[-1]
        energy = last_state.expectation_value(ci_matrix.matrix).real
        return energy

    parameter_trajectory: list[Sequence[complex]] = [p0]
    energy_trajectory: list[float] = []

    def _update_trajectory(parameters: Sequence[float]) -> None:
        parameter_trajectory.append(parameters)
        return

    def _const(parameters: Sequence[float | complex]) -> npt.ArrayLike:
        duration_const: list[float] = []
        sigma_const: list[float] = []
        amp_const: list[float] = []
        for i in range(0, len(parameters), len(init_state.keys())):
            cur_key = {}
            for j in range(len(init_state.keys())):
                _, k = parameter_names[i + j].split("_", maxsplit=1)
                cur_key[k] = parameters[i + j]
            if "amp_r" in cur_key:
                amp_const.append(
                    1 - np.abs(cur_key["amp_r"] + 1j * cur_key["amp_i"]) ** 2
                )
            elif "amp" in cur_key:
                amp_const.append(1 - np.abs(cur_key["amp"]) ** 2)
            if "sigma" in cur_key:
                sigma_const.append(cur_key["sigma"] - 0.1)
            if "duration" in cur_key:
                duration_const.append(cur_key["duration"] - 1)
        return np.concatenate(
            (
                duration_const,
                sigma_const,
                amp_const,
            )
        )

    end_time = time.time()
    prep_time = end_time - start_time
    # Prep <-

    # Optimization ->
    start_time = time.time()

    constraint = {"type": "ineq", "fun": _const}
    solution = scipy.optimize.minimize(
        fun=_cost,
        x0=p0,
        method="COBYLA",
        callback=_update_trajectory,
        constraints=[constraint],
        **kwargs,
    )

    end_time = time.time()
    optimization_time = end_time - start_time
    # Optimization <-

    # Energy ->
    start_time = time.time()

    for step in parameter_trajectory:
        energy_trajectory.append(_cost(step))
    ci_matrix_solution = CI_MatrixSolution(
        ci_matrix=ci_matrix,
        parameter_names=parameter_names,
        parameters_trajectory=parameter_trajectory,
        energy_trajectory=energy_trajectory,
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        dt=dt,
        success=solution.success,
        Nt=0,
        additional_data=dict(default_values=default_values),
    )

    end_time = time.time()
    energy_time = end_time - start_time
    # Energy <-
    optimization_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=optimization_time,
        energy_time=energy_time,
    )
    return (
        ci_matrix_solution,
        optimization_time,
        solution,
    )


@dataclasses.dataclass(frozen=True)
class TimingConstraints:
    acquire_alignment: int
    granularity: int
    min_length: int
    pulse_alignment: int

    @classmethod
    def from_backend(
        cls: type[TimingConstraints],
        backend: qiskit.providers.backend.Backend,
        ignore_min: bool = False,
    ) -> TimingConstraints:
        options = backend.configuration().timing_constraints
        return cls(
            acquire_alignment=options.get("acquire_alignment", 1),
            granularity=options.get("granularity", 1),
            min_length=options.get("min_length", 0) if not ignore_min else 0,
            pulse_alignment=options.get("pulse_alignment", 1),
        )

    @staticmethod
    def _get_close_multiple(value: int, base_number: int) -> int:
        nv = int(value + base_number / 2)
        return nv - (nv % base_number)

    def get_length(self: TimingConstraints, length: int) -> int:
        # TODO find potential fix to this weird behavior
        nv = int(length + self.granularity - 1)
        return nv - nv % self.granularity
        return self._get_close_multiple(length, self.granularity)

    def fix_min(self: TimingConstraints, length: int) -> int:
        return max(length, self.min_length)

    def get_delta(self: TimingConstraints, time: int) -> int:
        lcm = np.lcm(self.acquire_alignment, self.pulse_alignment)
        return self._get_close_multiple(time, lcm)


STANDARD_TIMING = TimingConstraints(
    acquire_alignment=16,
    granularity=16,
    min_length=64,
    pulse_alignment=1,
)


class PaddingType(enum.Enum):
    NO = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    MIDDLE = enum.auto()

    def pad(
        self: PaddingType,
        values: npt.ArrayLike,
        timing_const: TimingConstraints = STANDARD_TIMING,
    ) -> npt.NDArray[complex]:
        if self == self.NO:
            return np.asarray(values, dtype=complex)
        cur_len = len(values)
        new_len = timing_const.get_length(cur_len)
        new_len = timing_const.fix_min(new_len)
        ret = np.zeros(shape=(new_len,), dtype=complex)
        match self:
            case self.LEFT:
                ret[:cur_len] = values
            case self.MIDDLE:
                s = new_len // 2 - cur_len // 2
                ret[s : s + cur_len] = values
            case self.RIGHT:
                ret[-cur_len:] = values
        return ret


def create_empty_solution(
    Nt: int,
    dt: float,
    qubit_specification: Sequence[pulses.QubitSpecification],
    ci_matrix: CI_Matrix,
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[pulses.QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    single_connection: bool = False,
    complex_amplitude: bool = True,  # TODO add functionality
    random_initial_pulse: bool = True,
    padding_type: Optional[PaddingType] = None,
    timing_const: Optional[TimingConstraints] = None,
    **kwargs,
) -> tuple[CI_MatrixSolution, OptimizationTime]:
    s = time.time()

    solver, q_map, ch_map, signal_maker = pulses.generate_solver(
        qubit_specification,
        dt,
        cross_talk=cross_talk,
        qubit_noise_model=qubit_noise_model,
    )

    all_channels: Sequence[str] = [
        *[f"d{q}" for q in q_map.values()],
        *[f"u{c}" for c in ch_map.values()],
    ]

    parameter_names: Sequence[str] = [
        f"d{q}_{i}" for q in q_map.values() for i in range(Nt)
    ]
    if single_connection:
        single_channels = _get_single_direction(
            qubit_specification=qubit_specification,
            real_to_sim_map=q_map,
            control_channels=ch_map,
        )
        parameter_names = [
            *parameter_names,
            *[f"{c}_{i}" for c in single_channels for i in range(Nt)],
        ]
    else:
        parameter_names = [  # TODO consider some smart use of channels
            *parameter_names,
            *[f"u{c}_{i}" for c in ch_map.values() for i in range(Nt)],
        ]

    # TODO add seed control
    p0: npt.NDArray[complex]
    if random_initial_pulse:
        p0 = random_complex(len(parameter_names))
    else:
        p0 = np.zeros(len(parameter_names), dtype=complex)

    e_sol = CI_MatrixSolution(
        ci_matrix=ci_matrix,
        parameters_trajectory=[
            p0,
        ],
        parameter_names=parameter_names,
        energy_trajectory=[
            ci_matrix.nuclear_repulsion_energy + ci_matrix.matrix[0, 0],
        ],
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        dt=dt,
        success=False,
        Nt=Nt,
    )

    e = time.time()
    prep_time = e - s
    times = OptimizationTime(
        prep_time=prep_time,
        optimization_time=0,
        energy_time=0,
    )
    return e_sol, times


# TODO divide the function to several smaller ones and add a call manager
def optimize(
    Nt: int,
    dt: float,
    qubit_specification: Sequence[pulses.QubitSpecification],
    ci_matrix: CI_Matrix,
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[pulses.QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    single_connection: bool = False,
    complex_amplitude: bool = True,  # TODO add functionality
    random_initial_pulse: bool = True,
    padding_type: Optional[PaddingType] = None,
    timing_const: Optional[TimingConstraints] = None,
    **kwargs,
) -> tuple[CI_MatrixSolution, OptimizationTime, Any]:
    start_time = time.time()

    solver, q_map, ch_map, signal_maker = pulses.generate_solver(
        qubit_specifications=qubit_specification,
        dt=dt,
        cross_talk=cross_talk,
        qubit_noise_model=qubit_noise_model,
    )

    all_channels: Sequence[str] = [
        *[f"d{q}" for q in q_map.values()],
        *[f"u{c}" for c in ch_map.values()],
    ]

    parameter_names: Sequence[str] = [
        f"d{q}_{i}" for q in q_map.values() for i in range(Nt)
    ]
    if single_connection:
        single_channels = _get_single_direction(
            qubit_specification=qubit_specification,
            real_to_sim_map=q_map,
            control_channels=ch_map,
        )
        parameter_names = [
            *parameter_names,
            *[f"{c}_{i}" for c in single_channels for i in range(Nt)],
        ]
    else:
        parameter_names = [  # TODO consider some smart use of channels
            *parameter_names,
            *[f"u{c}_{i}" for c in ch_map.values() for i in range(Nt)],
        ]

    # TODO add seed control
    p0: npt.NDArray[complex]
    if random_initial_pulse:
        p0 = random_complex(len(parameter_names))
    else:
        p0 = np.zeros(len(parameter_names), dtype=complex)

    parameters_trajectory: list[Sequence[complex]] = [
        p0 if complex_amplitude else p0.real,
    ]
    energy_trajectory: list[float] = []

    if t_span is None:
        t_end = Nt
        if padding_type is not None:
            if timing_const is None:
                t_end = len(padding_type.pad(p0))
            else:
                t_end = len(padding_type.pad(p0, timing_const=timing_const))
        t_span = (0, dt * t_end)

    if y0 is None:
        lbl = "0" * len(qubit_specification)
        if qubit_noise_model is None:
            y0 = Statevector.from_label(lbl)
        else:
            y0 = DensityMatrix.from_label(lbl)

    def _update_trajectory_complex(parameters: npt.ArrayLike) -> None:
        c_parameters: npt.NDArray[complex] = _real_to_complex(np.asarray(parameters))
        parameters_trajectory.append(c_parameters.tolist())
        return

    def _update_trajectory_float(parameters: npt.ArrayLike) -> None:
        parameters_trajectory.append(np.asarray(parameters, dtype=complex).tolist())
        return

    def _cost_complex(parameters: Sequence[float]) -> float:
        c_parameters: Sequence[complex] = _real_to_complex(
            np.asarray(parameters)
        ).tolist()
        return __cost_complex(c_parameters)

    def __cost_complex(parameters: Sequence[complex]) -> float:
        schedule = build_schedule(
            parameter_names,
            parameters,
            padding_type=padding_type,
            timing_constraints=timing_const,
        )
        schedule = pulses.pad_schedule(schedule, all_channels)
        results = solver.solve(t_span=t_span, y0=y0, signals=schedule)
        last_state: QISKIT_STATE = results.y[-1]
        # TODO Might need to tweak this to allow for smooth operation with Transmon.
        energy = (
            last_state.expectation_value(ci_matrix.matrix).real
            + ci_matrix.nuclear_repulsion_energy
        )
        return energy

    def _cost_float(parameters: Sequence[float]) -> float:
        return __cost_complex(np.asarray(parameters, dtype=complex))

    def _const_complex(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(_real_to_complex(np.asarray(parameters))) ** 2 - 1)

    def _const_float(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(parameters) ** 2 - 1)

    prep_time = time.time() - start_time
    start_time = time.time()

    minimize_options = {
        "method": "COBYLA",
    }
    minimize_options.update(kwargs)

    if complex_amplitude:
        constraint = {"type": "ineq", "fun": _const_complex}
        solution = scipy.optimize.minimize(
            fun=_cost_complex,
            x0=_complex_to_real(p0),
            callback=_update_trajectory_complex,
            constraints=[constraint],
            **kwargs,
        )

    else:
        constraint = {"type": "ineq", "fun": _const_float}
        solution = scipy.optimize.minimize(
            fun=_cost_float,
            x0=np.real(p0),
            callback=_update_trajectory_float,
            constraints=[constraint],
            **kwargs,
        )

    optimization_time = time.time() - start_time
    start_time = time.time()

    for step in parameters_trajectory:
        energy_trajectory.append(__cost_complex(step))

    ci_matrix_solution = CI_MatrixSolution(
        ci_matrix=ci_matrix,
        parameters_trajectory=parameters_trajectory,
        parameter_names=parameter_names,
        energy_trajectory=energy_trajectory,
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        dt=dt,
        success=solution.success,
        Nt=Nt,
        additional_data={
            "padding": padding_type,
            "timing_constraints": timing_const,
        },
    )

    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=optimization_time,
        energy_time=end_time - start_time,
    )

    return (
        ci_matrix_solution,
        opt_time,
        solution,
    )


RT = TypeVar("RT")


def set_minimal_execution_time(
    not_until: datetime.datetime,
    function: Callable[[], RT],
    verbose: bool = False,
    seconds_divider: float = 2,
) -> RT:
    """
    set_minimal_execution_time: waits until a specific datetime to run some function.

    Runs the `function` only after `not_until` and returns the return value.
    If `verbose` is set to True, the function will print out the waiting times.
    `seconds_divider` sets to how many "waiting blocks" to divide the total wait time.

    Parameters
    ----------
    not_until : datetime.datetime
        Date and time to wait for
    function : Callable[[], RT]
        The function to execute
    verbose : bool
        Print sleep times (default: False)
    seconds_divider : float
        The divider for the waiting time (default: 2)

    Returns
    -------
    RT
        The return value of `function()`.
    """
    now = datetime.datetime.now()
    while now <= not_until:
        wait_seconds = (not_until - now) / seconds_divider
        if verbose:
            print(f"Waiting for {wait_seconds}...")
        time.sleep(wait_seconds.total_seconds())
        now = datetime.datetime.now()
    if verbose:
        print(f"Starting to run {datetime.datetime.now()}>{not_until}")
    return function()


S = TypeVar("S", DensityMatrix, Statevector)


def project01(n_qubits: int, transmon_dim: int) -> Callable[[S], S]:
    proj_01 = np.array(
        [
            np.concatenate(([1], np.repeat(0, transmon_dim - 1))),
            np.concatenate(([0, 1], np.repeat(0, transmon_dim - 2))),
        ]
    ).astype(complex)
    P01 = np.array([1])
    for _ in range(n_qubits):
        P01 = np.kron(P01, proj_01)

    def _internal(state: S) -> S:
        data = state.data
        if isinstance(state, DensityMatrix):
            data = P01 @ data @ P01.T
            data /= data.trace()
            return DensityMatrix(data, (2,) * n_qubits)
        elif isinstance(state, Statevector):
            data = data @ P01.T
            data /= np.sqrt(np.sum(np.abs(data) ** 2))
            return Statevector(data, (2,) * n_qubits)
        else:
            raise NotImplementedError(f"What am I to do? {type(state)=}, {state=}.")

    return _internal


# TODO Think of maybe doing it better, for now, this will do
def optimize_transmon(
    Nt: int,
    dt: float,
    qubit_specification: Sequence[pulses.QubitSpecification],
    ci_matrix: CI_Matrix,
    transmon_dim: int = 3,
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[pulses.QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    single_connection: bool = False,
    complex_amplitude: bool = True,  # TODO add functionality
    random_initial_pulse: bool = True,
    phys_to_logical: int = 1,
    padding_type: Optional[PaddingType] = None,
    timing_const: Optional[TimingConstraints] = None,
    H_schedule: Optional[ScheduleBlock] = None,
    **kwargs,
) -> tuple[CI_MatrixSolution, OptimizationTime, Any]:
    start_time = time.time()

    solver, q_map, ch_map, signal_maker = pulses.generate_solver(
        qubit_specification,
        dt,
        cross_talk,
        qubit_noise_model,
        method="transmon",
        transmon_dim=transmon_dim,
    )

    all_channels: Sequence[str] = [
        *[f"d{q}" for q in q_map.values()],
        *[f"u{c}" for c in ch_map.values()],
    ]

    parameter_names: Sequence[str] = [
        f"d{q}_{i}" for q in q_map.values() for i in range(Nt)
    ]
    if single_connection:
        single_channels = _get_single_direction(
            qubit_specification=qubit_specification,
            real_to_sim_map=q_map,
            control_channels=ch_map,
        )
        parameter_names = [
            *parameter_names,
            *[f"{c}_{i}" for c in single_channels for i in range(Nt)],
        ]
    else:
        parameter_names = [  # TODO consider some smart use of channels
            *parameter_names,
            *[f"u{c}_{i}" for c in ch_map.values() for i in range(Nt)],
        ]

    dur_H: Optional[int]
    sig_H: Optional[ScheduleBlock]
    if H_schedule is None:
        dur_H = sig_H = None
    else:
        dur_H = H_schedule.duration
        sig_H = signal_maker(H_schedule)

    # TODO add seed control
    p0: npt.NDArray[complex]
    if random_initial_pulse:
        p0 = random_complex(len(parameter_names))
    else:
        p0 = np.zeros(len(parameter_names), dtype=complex)

    parameters_trajectory: list[Sequence[complex]] = [
        p0 if complex_amplitude else p0.real,
    ]
    energy_trajectory: list[float] = []
    energy_trajectory2: list[float] = []
    final_state_trajectory: list[Statevector | DensityMatrix] = []

    if t_span is None:
        t_end = Nt
        if padding_type is not None:
            if timing_const is None:
                t_end = len(padding_type.pad(p0))
            else:
                t_end = len(padding_type.pad(p0, timing_const=timing_const))
        t_span = (0, dt * t_end)

    n_qubits = len(qubit_specification)
    if y0 is None:
        state = np.concatenate(
            (
                [
                    1,
                ],
                np.repeat(0, transmon_dim**n_qubits - 1),
            )
        ).astype(complex)
        if qubit_noise_model is None:
            y0 = Statevector(data=state, dims=(transmon_dim,) * n_qubits)
        else:
            y0 = DensityMatrix(data=np.diag(state), dims=(transmon_dim,) * n_qubits)

    # TODO Make a better fix that will hold for more than 1 "qubit"
    # matrix_for_ev = np.zeros((transmon_dim, transmon_dim)).astype(complex)
    # matrix_for_ev[:2, :2] = ci_matrix.matrix

    projector = project01(n_qubits, transmon_dim)

    def _update_trajectory_complex(parameters: npt.ArrayLike) -> None:
        c_parameters: npt.NDArray[complex] = _real_to_complex(np.asarray(parameters))
        parameters_trajectory.append(c_parameters.tolist())
        return

    def _update_trajectory_float(parameters: npt.ArrayLike) -> None:
        parameters_trajectory.append(np.asarray(parameters, dtype=complex).tolist())
        return

    def _cost_complex(parameters: Sequence[float]) -> float:
        c_parameters: Sequence[complex] = _real_to_complex(
            np.asarray(parameters)
        ).tolist()
        return __cost_complex(c_parameters)

    def __cost_complex(parameters: Sequence[complex]) -> float:
        cost_time = time.time()
        schedule = build_schedule(
            parameter_names,
            parameters,
            padding_type=padding_type,
            timing_constraints=timing_const,
            phys_to_logical=phys_to_logical,
        )
        schedule = pulses.pad_schedule(schedule, all_channels)
        results = solver.solve(t_span=t_span, y0=y0, signals=schedule)
        cost_time = time.time() - cost_time
        last_state: QISKIT_STATE = results.y[-1]
        # TODO Might need to tweak this to allow for smooth operation with Transmon.
        if H_schedule is None:
            # TODO Add additional simulation of H gate on the state w. measurement split into X and Z
            energy = (
                # last_state.expectation_value(matrix_for_ev).real
                # type(last_state)(LP01 @ (last_state.data @ P01.T), dims=(2,) * n_qubits).expectation_value(ci_matrix.matrix).real
                projector(last_state).expectation_value(ci_matrix.matrix).real
                # PO(last_state).expectation_value(ci_matrix.matrix).real
                + ci_matrix.nuclear_repulsion_energy
            )
        else:
            more_steps = solver.solve(
                t_span=[0, (dur_H + 1) * dt],
                y0=last_state,
                signals=sig_H,
            )
            lastX = more_steps.y[-1]
            # TODO Make more general (more than 1 qubit)
            Z = np.diag([1, -1])
            X = np.array([[0, 1], [1, 0]])
            H = (Z + X) / np.sqrt(2)
            Zc = (ci_matrix.matrix @ Z).trace() / 2
            Xc = (ci_matrix.matrix @ X).trace() / 2
            Ic = (ci_matrix.matrix @ np.eye(2)).trace() / 2
            Zev = projector(last_state).expectation_value(Z).real
            Xev = projector(lastX).expectation_value(Z).real
            energy = Ic + Xc * Xev + Zc * Zev + ci_matrix.nuclear_repulsion_energy
            # last_state = |\psi><\psi|
            Xev = (
                DensityMatrix(H @ projector(last_state).data @ H, dims=(2,))
                .expectation_value(Z)
                .real
            )
            energy2 = Ic + Xc * Xev + Zc * Zev + ci_matrix.nuclear_repulsion_energy
            energy_trajectory2.append(energy2)
        print(
            f"Iteration: {len(energy_trajectory)} - Time: {cost_time} - Energy: {energy} - Distance {energy-ci_matrix.ci_ground_state_energy}."
        )
        energy_trajectory.append(energy)
        final_state_trajectory.append(last_state)
        return energy

    def _cost_float(parameters: Sequence[float]) -> float:
        return __cost_complex(np.asarray(parameters, dtype=complex))

    def _const_complex(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(_real_to_complex(np.asarray(parameters))) ** 2 - 1)

    def _const_float(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(parameters) ** 2 - 1)

    prep_time = time.time() - start_time
    start_time = time.time()

    minimize_options = {
        "method": "COBYLA",
    }
    minimize_options.update(kwargs)

    if complex_amplitude:
        constraint = {"type": "ineq", "fun": _const_complex}
        solution = scipy.optimize.minimize(
            fun=_cost_complex,
            x0=_complex_to_real(p0),
            callback=_update_trajectory_complex,
            constraints=[constraint],
            **kwargs,
        )

    else:
        constraint = {"type": "ineq", "fun": _const_float}
        solution = scipy.optimize.minimize(
            fun=_cost_float,
            x0=np.real(p0),
            method="COBYLA",
            callback=_update_trajectory_float,
            constraints=[constraint],
            **kwargs,
        )

    optimization_time = time.time() - start_time
    start_time = time.time()

    # for step in parameters_trajectory:
    #     energy_trajectory.append(__cost_complex(step))

    ci_matrix_solution = CI_MatrixSolution(
        ci_matrix=ci_matrix,
        parameters_trajectory=parameters_trajectory,
        parameter_names=parameter_names,
        energy_trajectory=energy_trajectory,
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        dt=dt,
        success=solution.success,
        Nt=Nt,
        additional_data={
            "phys_to_logical": phys_to_logical,
            "padding": padding_type,
            "timing_constraints": timing_const,
            "last_states": final_state_trajectory,
            "energy2": energy_trajectory2,
        },
    )

    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=optimization_time,
        energy_time=end_time - start_time,
    )

    return (
        ci_matrix_solution,
        opt_time,
        solution,
    )
