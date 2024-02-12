from typing import Optional, Any
from collections.abc import Mapping, Sequence
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import scipy
import scipy.optimize

import qiskit
import qiskit.quantum_info
import qiskit.quantum_info.operators
import qiskit.utils
import qiskit.utils.mitigation
import qiskit.utils.mitigation.fitters
import qiskit.circuit
import qiskit.providers
import qiskit.providers.ibmq
import qiskit.providers.ibmq.managed
import qiskit.providers.backend
import qiskit.providers.job
import qiskit.pulse
import qiskit.pulse.channels
import qiskit.pulse.library
import qiskit.result
import qiskit.result.utils
import qiskit.result.models
import qiskit.result.mitigation
import qiskit.result.mitigation.local_readout_mitigator

import qiskit_experiments
import qiskit_experiments.library
import time
import datetime
import re
from pathlib import Path
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

from .hw import _fix_counts, BACKEND_TYPE, HWCI_MatrixSolution


@dataclass(kw_only=True)
class HWCI_MatrixSolution2(CI_MatrixSolution):
    raw_data_trajectory: Sequence[Mapping[str, Mapping[str, int]]]
    energy_error: Sequence[float]
    assignment_matrices: Sequence[Sequence[npt.NDArray] | npt.NDArray]


BACKEND_TYPE = qiskit.providers.backend.BackendV1 | qiskit.providers.backend.BackendV2


def optimize_optimized(
    qubit_specification: Sequence[QubitSpecification],
    Nt: int,
    backend: BACKEND_TYPE,
    padding_type: PaddingType,
    timing_const: TimingConstraints,
    prev_solution: CI_MatrixSolution,
    verbose: bool = False,
    n_shots: Optional[int] = None,
    repetitions: int = 1,
    exit_file: Optional[Path] = None,
    live_feed_file: Optional[Path] = None,
    failsafe: bool = True,
    **kwargs,
) -> tuple[HWCI_MatrixSolution2, OptimizationTime, Any]:
    if verbose:
        print(f"Starting prep... {datetime.datetime.now()}")
    start_time = time.time()  # Prep time

    channel_pattern = re.compile(r"^(?P<channel>\w)(?P<qubit>\d+)_(?P<index>\d+)$")
    channel_map: dict[str, dict[int, int]] = {}
    d_chans = channel_map["d"] = {}
    for i, q in enumerate(qubit_specification):
        d_chans[i] = q.index

    device_control_channels = lambda i, j: backend.configuration().control_channels[
        (i, j)
    ][0]
    u_chans = channel_map["u"] = {}
    for q in qubit_specification:
        for oq in sorted(q.coupling_map.keys()):
            u_chans[len(u_chans)] = device_control_channels(q.index, oq)

    # TODO fix the way channels get named. Current scheme allows for problems in the control channels.
    # TODO I will currently ignore the control channels until a better solution is available.
    # TODO Potential fix, when creating s simulation, sort according to the qubit index.
    prev_names = prev_solution.parameter_names
    new_names = [
        "",
    ] * len(prev_names)

    for i, name in enumerate(prev_names):
        match = channel_pattern.match(name)
        if match is None:
            raise RuntimeError(f"What is this '{name}'?!")
        md = match.groupdict()
        channel, qubit, index = md["channel"], int(md["qubit"]), int(md["index"])
        new_names[i] = f"{channel}{channel_map[channel][qubit]}_{index}"

    assign_mats: list[Sequence[npt.NDArray] | npt.NDArray] = []

    end_time = time.time()
    prep_time = end_time - start_time

    start_time = time.time()
    energy_trajectory: list[float] = []
    energy_error: list[float] = []
    parameter_trajectory: list[Sequence[complex]] = [
        prev_solution.parameters_trajectory[-1],
    ]
    p0 = _complex_to_real(np.asarray(parameter_trajectory[0])).tolist()
    raw_data: list[Mapping[str, Sequence[Mapping[str, int]]]] = []
    job_ids: list[str] = []
    end_time = time.time()

    stop_optimization: bool = False

    gate_name = "VQE"
    n_qubits = backend.configuration().n_qubits
    if n_shots is None:
        n_shots = backend.configuration().max_shots
    qubits = [q.index for q in qubit_specification]

    def _base_circuit(
        schd: qiskit.pulse.ScheduleBlock, ind: Optional[int] = None
    ) -> qiskit.circuit.QuantumCircuit:
        qc = qiskit.circuit.QuantumCircuit(n_qubits, n_qubits)
        gate = qiskit.circuit.Gate(gate_name, len(qubit_specification), [])
        qc.add_calibration(gate, qubits, schd, [])
        qc.append(gate, qubits, [])
        qc.metadata = {
            "Nt": Nt,
            "padding": padding_type.value,
            "measure": "z",
        }
        if ind is not None:
            qc.metadata["i"] = ind
        return qc

    def _update_trajectory(parameters: npt.ArrayLike) -> None:
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return
        c_param = _real_to_complex(np.asarray(parameters)).tolist()
        parameter_trajectory.append(c_param)
        if verbose:
            print(
                f"Another point {len(parameter_trajectory)} {datetime.datetime.now()}"
            )

    def _calc_energy(
        values: Mapping[str, Sequence[tuple[float, float]]]
    ) -> tuple[float, float]:
        dim = 2 ** len(qubit_specification)
        I = np.eye(dim, dtype=complex)
        energy = prev_solution.ci_matrix.nuclear_repulsion_energy + (
            prev_solution.ci_matrix.matrix @ I
        ).trace().real / (dim)
        error2 = 0
        agg_values: dict[str, tuple[float, float]] = {}
        for lbl, expstds in values.items():
            mat = np.array(expstds)
            x_val = mat[:, 0].mean()
            x_stat = (
                mat[:, 0].std(ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else 0
            )
            x_std = mat[:, 1] ** 2
            agg_values[lbl] = x_val, np.sqrt(x_stat**2 + x_std.sum())
        for lbl, (exp, std) in agg_values.items():
            o = qiskit.quantum_info.operators.Operator.from_label(lbl.upper()).data
            oc = (prev_solution.ci_matrix.matrix @ o).trace().real / (dim)
            energy += exp * oc
            error2 += (std * oc) ** 2
        error = np.sqrt(error2)
        if verbose:
            print(
                f"Got the following energy {energy:.10f}±{error:.10f}... {datetime.datetime.now()}"
            )
        return energy, error

    def _cost(parameters: npt.ArrayLike) -> float:
        nonlocal stop_optimization
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return -np.inf
        # TODO more complex measurements of different axes
        c_param = _real_to_complex(np.asarray(parameters)).tolist()
        schd = build_schedule(
            parameter_names=new_names,
            parameters=c_param,
            padding_type=padding_type,
            timing_constraints=timing_const,
        )
        b_circ = _base_circuit(schd)
        # TODO Add here the entire tomography process
        z_circ = b_circ.copy()
        x_circ = b_circ.copy()
        x_circ.metadata["measure"] = "x"
        for q in qubit_specification:
            x_circ.h(q.index)
            for qc in (z_circ, x_circ):
                qc.measure(q.index, q.index)
        t_circs = qiskit.transpile([z_circ, x_circ] * repetitions, backend)

        try:
            job: qiskit.providers.job.JobV1 = backend.run(t_circs, shots=n_shots)

            job_ids.append(job.job_id())

            if verbose:
                print(f"Running experiment... {datetime.datetime.now()}")
            # TODO think of how to handle errors in the machine
            job.wait_for_final_state()
            if verbose:
                print(f"Finished experiment... {datetime.datetime.now()}")
            results: qiskit.result.Result = job.result()
        except Exception as e:
            if not failsafe:
                raise e from None
            _log_error(e)
            stop_optimization = True
            return -np.inf

        # TODO more complex evaluation of each data point
        l_raw: dict[str, list[Mapping[str, int]]] = defaultdict(list)
        em: dict[str, list[tuple[float, float]]] = defaultdict(list)

        # TODO maybe add a graceful fail with potential to recover?
        if not results.success:
            err = RuntimeError(f"Job '{job.job_id()}' has failed!\n{results.status}")
            if not failsafe:
                raise err from None
            _log_error(err)
            stop_optimization = True
            return -np.inf

        res: Sequence[qiskit.result.models.ExperimentResult] = results.results
        for r in res:
            metadata = r.header.metadata
            counts = r.data.counts
            fixed_counts = _fix_counts(counts, n_qubits)
            marginal_counts = qiskit.result.utils.marginal_counts(fixed_counts, qubits)
            l_raw[metadata["measure"]].append(counts)
            # l_raw[metadata["measure"]] = counts
            # TODO move all the post-process of the results to the calc function
            em[metadata["measure"]].append(em_obj.expectation_value(marginal_counts))
            # em[metadata["measure"]] = em_obj.expectation_value(marginal_counts)

        raw_data.append(l_raw)
        energy, error = _calc_energy(em)
        # TODO move this to the post-optimize part of the code, that way the parameters_trajectory and energy_trajectory will have the same size
        energy_trajectory.append(energy)
        energy_error.append(error)
        return energy

    log_file = (
        exit_file.with_suffix(".log")
        if exit_file is not None
        else Path.cwd() / "cur.log"
    )
    log_file.touch(exist_ok=True)

    def _log_error(e: Exception) -> None:
        prev = log_file.read_text(encoding="utf-8")
        log_file.write_text(f"{prev}\n{e}", encoding="utf-8")

    def _const(parameters: npt.ArrayLike) -> npt.ArrayLike:
        c_param = _real_to_complex(np.asarray(parameters))
        return -(np.abs(c_param) ** 2 - 1)

    constraint = {"type": "ineq", "fun": _const}
    if verbose:
        print("Starting optimization...")
    solution = scipy.optimize.minimize(
        fun=_cost,
        x0=p0,
        method="COBYLA",
        callback=_update_trajectory,
        constraints=[
            constraint,
        ],
        **kwargs,
    )

    if stop_optimization:
        print(f"Optimization failed! (look into {log_file})")

    if verbose:
        print(f"Finished optimization... {datetime.datetime.now()}")
    opt_time = end_time - start_time

    start_time = time.time()
    hwci_matrixsolution = HWCI_MatrixSolution(
        ci_matrix=prev_solution.ci_matrix,
        Nt=Nt,
        parameter_names=new_names,
        parameters_trajectory=parameter_trajectory,
        energy_trajectory=energy_trajectory,
        qubit_spec=qubit_specification,
        dt=backend.configuration().dt,
        success=solution.success,
        raw_data_trajectory=raw_data,
        energy_error=energy_error,
        assignment_matrix=assign_mat,
        additional_data={
            "job_ids": job_ids,
            "backend": backend.name(),
            "repetitions": repetitions,
            "n_shots": n_shots,
            "total_shots": n_shots * repetitions,
            "padding": padding_type,
            "timing_constraints": timing_const,
        },
    )
    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=opt_time,
        energy_time=end_time - start_time,
    )
    return (hwci_matrixsolution, opt_time, solution)
