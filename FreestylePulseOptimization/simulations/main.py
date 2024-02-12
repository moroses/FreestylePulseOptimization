from re import L
import re
import numpy as np
from numpy import typing as npt
from collections.abc import Sequence, Mapping, MutableSequence
from collections import defaultdict
from typing import Any, Callable, Optional, NamedTuple
from pathlib import Path
import time
from dataclasses import dataclass
from qiskit_dynamics.array import Array

from scipy.optimize import minimize

from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit.pulse.schedule import ScheduleBlock, Schedule

from ..utils import (
    CI_Matrix,
    QISKIT_STATE,
    PaddingType,
    TimingConstraints,
    OptimizationTime,
    STANDARD_TIMING,
    project01,
    _complex_to_real,
    _real_to_complex,
)
from .. import pulses

from ..utils import CI_MatrixSolution

from .utils import (
    CA_region,
    QubitChannels,
    ChannelProperties,
    get_channel_list,
    get_channel,
    SimulationCI_MatrixSolution,
)


from qiskit import pulse as qpulse


def create_schedule(
    parameter_names: Sequence[str],
    channel_properties: Mapping[str, ChannelProperties],
    timing_constraints: TimingConstraints,
) -> Callable[[Sequence[complex]], ScheduleBlock]:
    def _internal(parameters: Sequence[complex]) -> ScheduleBlock:
        channel_map: dict[Channel, MutableSequence[complex]] = defaultdict(list)
        for name, value in zip(parameter_names, parameters):
            channel_name, index = name.split("_", maxsplit=1)
            index = int(index)
            channel = get_channel(channel_name)
            phys_to_logical = channel_properties[channel_name].phys_to_logical
            channel_map[channel].extend(
                [
                    value,
                ]
                * phys_to_logical
            )
        with qpulse.build() as schd:
            for channel, values in channel_map.items():
                padding = channel_properties[channel.name].padding
                qpulse.play(
                    qpulse.Waveform(
                        padding.pad(values, timing_constraints),
                        epsilon=0.5,
                        limit_amplitude=False,
                    ),
                    channel,
                )
        return schd

    return _internal


def optimize_transmon(
    Nt: int,
    dt: float,
    qubit_specification: Sequence[pulses.QubitSpecification],
    ci_matrix: CI_Matrix,
    channel_properties: Mapping[str, ChannelProperties],
    transmon_dim: int = 2,
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[pulses.QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    channels_to_remove: Optional[Sequence[str]] = None,
    timing_const: Optional[TimingConstraints] = None,
    verbose: bool = True,
    solver_options: Mapping[str, Any] = {},
    previous_parameters: Optional[npt.NDArray] = None,
    **kwargs,
) -> tuple[SimulationCI_MatrixSolution, OptimizationTime, Any]:
    start_time = time.time()

    if timing_const is None:
        timing_const = STANDARD_TIMING
    solver, q_map, ch_map, signal_maker = pulses.generate_solver(
        qubit_specification,
        dt,
        cross_talk,
        qubit_noise_model,
        method="transmon-dicke",
        transmon_dim=transmon_dim,
    )

    all_channels: Sequence[str] = [
        *(drive_channels := [f"d{q}" for q in q_map.values()]),
        *(control_channels := [f"u{c}" for c in ch_map.values()]),
    ]

    if channels_to_remove is not None:
        rel_channels = list(filter(lambda s: s not in channels_to_remove, all_channels))
    else:
        rel_channels = all_channels

    # TODO See "TODO" next to "ChannelProperties"
    parameter_names: list[str] = [f"{c}_{i}" for c in rel_channels for i in range(Nt)]
    # parameter_names: Sequence[str] = [
    #     f"d{q}_{i}" for q in q_map.values() for i in range(Nt)
    # ]
    # parameter_names = [  # TODO consider some smart use of channels
    #     *parameter_names,
    #     *[f"u{c}_{i}" for c in ch_map.values() for i in range(Nt)],
    # ]

    # TODO add seed control
    p0: npt.NDArray[complex]
    if previous_parameters is None:
        p0 = np.zeros(len(parameter_names), dtype=complex)
    else:
        p0 = np.array(previous_parameters)

    parameters_trajectory: list[Sequence[complex]] = [p0]
    energy_trajectory: list[float] = []
    final_state_trajectory: list[Statevector | DensityMatrix] = []

    schedule_maker = create_schedule(parameter_names, channel_properties, timing_const)

    if t_span is None:
        t_end = schedule_maker(p0).duration
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
        y0 = Array(state)
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

    def _cost_complex(parameters: Sequence[float]) -> float:
        c_parameters: Sequence[complex] = _real_to_complex(
            np.asarray(parameters)
        ).tolist()
        cost_time = time.time()
        value = __cost_complex(c_parameters)
        cost_time = time.time() - cost_time
        if is_jit and hasattr(value, "block_until_ready"):
            value.block_until_ready()
        energy, last_state, _ = value
        # if hasattr(energy, "squeeze"):
        #     energy = energy.squeeze()
        if verbose:
            print(
                f"Iteration: {len(energy_trajectory)} - Time: {cost_time} - Energy: {energy} - Distance {energy-ci_matrix.ci_ground_state_energy}."
            )
        energy_trajectory.append(energy)
        final_state_trajectory.append(last_state)
        # import pdb; pdb.set_trace()
        return energy

    def __cost_complex(parameters: Sequence[complex]) -> float:
        cost_time = time.time()
        schedule = schedule_maker(parameters)
        if isinstance(schedule, (Schedule, ScheduleBlock)):
            schedule = pulses.pad_schedule(schedule, all_channels)
        results = solver.solve(
            t_span=t_span,
            y0=y0,
            signals=schedule,
            convert_results=not is_jit,
            **solver_options,
        )
        cost_time = time.time() - cost_time
        last_state: QISKIT_STATE = results.y[-1]
        energy = (
            projector(last_state).expectation_value(ci_matrix.matrix).real
            + ci_matrix.nuclear_repulsion_energy
        )

        # energy_trajectory.append(energy)
        # final_state_trajectory.append(last_state)
        return energy, last_state, cost_time

    is_jit = solver_options.pop("jit", False)
    if is_jit:
        from qiskit_dynamics.array import wrap
        from jax import jit as jax_jit
        from ..jax_utils import convert_to_jax_properties

        jit = wrap(jax_jit, decorator=True)
        # TODO return here, make the padding better
        from .jax import create_schedule as jax_schedule
        from ..jax_utils import project01 as jax_project01

        projector = jax_project01(n_qubits, transmon_dim)
        frequencies = pulses.get_frequencies(
            qubit_specification,
            q_map,
            ch_map,
        )
        schedule_maker = jax_schedule(
            dt,
            frequencies,
            parameter_names,
            {
                ch: convert_to_jax_properties(prop)
                for ch, prop in channel_properties.items()
            },
            # channel_properties,
            timing_const,
            drive_channels,
            control_channels,
            channels_to_remove,
        )
        __cost_complex = jit(__cost_complex)
        # schedule_maker = # TODO Feel this part with a JAX version

    def _const_complex(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(_real_to_complex(np.asarray(parameters))) ** 2 - 1)

    prep_time = time.time() - start_time
    start_time = time.time()

    minimize_options = {
        "method": "COBYLA",
    }
    minimize_options.update(kwargs)

    constraint = {"type": "ineq", "fun": _const_complex}
    solution = minimize(
        fun=_cost_complex,
        x0=_complex_to_real(p0),
        callback=_update_trajectory_complex,
        constraints=[constraint],
        **kwargs,
    )

    optimization_time = time.time() - start_time
    start_time = time.time()

    # for step in parameters_trajectory:
    #     energy_trajectory.append(__cost_complex(step))

    ci_matrix_solution = SimulationCI_MatrixSolution(
        Nt=Nt,
        ci_matrix=ci_matrix,
        parameter_names=parameter_names,
        parameters_trajectory=parameters_trajectory,
        energy_trajectory=energy_trajectory,
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        dt=dt,
        success=solution.success,
        channel_properties=channel_properties,
        transmon_dim=transmon_dim,
        states=final_state_trajectory,
        additional_data={
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
