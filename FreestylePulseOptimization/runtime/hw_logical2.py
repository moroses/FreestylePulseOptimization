import re
import datetime
import time

import pandas as pd
import numpy as np
import numpy.typing as npt

from collections.abc import Sequence, Mapping
from typing import Callable, Any, Optional
from dataclasses import dataclass, field

from collections import defaultdict

from pathlib import Path
from qiskit import pulse
from qiskit.pulse.channels import PulseChannel
from qiskit import transpile
from qiskit.pulse.library import waveform
from qiskit.result.models import ExperimentResult
from qiskit.result.result import Result

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    Sampler,
    IBMBackend,
    Options,
)
from qiskit.primitives.sampler import SamplerResult
from qiskit.providers.jobstatus import JobStatus
from qiskit.result.mitigation.local_readout_mitigator import LocalReadoutMitigator
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.pulse import (
    ScheduleBlock,
    build,
    DriveChannel,
    ControlChannel,
    Waveform,
    play,
    barrier,
    align_left,
    align_sequential,
)

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter

from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit_experiments.framework import ExperimentData

from qiskit_dynamics.signals import Signal
from qiskit.quantum_info.states import DensityMatrix, Statevector

from functools import partial

import scipy.optimize

from ..iq.utils import FreqHunt, AmpHunt, BaseDragPulseFactors

from ..utils import CI_MatrixSolution, QISKIT_STATE, build_schedule, project01
from ..pulses import (
    QubitSpecification,
    get_qubit_noise_parameters,
    get_si_mult,
    QubitNoiseParameters,
    generate_solver,
    pad_schedule,
)
from ..utils import (
    _complex_to_real,
    _real_to_complex,
    STANDARD_TIMING,
    _get_single_direction,
)

from ..utils import (
    TimingConstraints,
    OptimizationTime,
    CI_Matrix,
    CI_MatrixSolution,
    PaddingType,
    pauli_decomposer,
)

from .utils import (
    int_div_mod,
    compose,
    map_values,
    prob_to_count,
    to_binary_keys,
    mitigate_error,
    marginalize_counts,
    run_on_index,
    matrix_ev,
    get_relevant_qubits,
    measure_pauli_string,
    pauli_strings,
    get_channel_names,
)

from ..runners import RunOnBackend, BACKEND_TYPE, JOB_TYPE, get_n_qubits
from qiskit.qobj.utils import MeasLevel, MeasReturnType


from ..iq.logic_advanced import find_01_freq, find_01_amp

from .utils import (
    HWCI_MatrixSolutionRuntimeLogicalCircuitRunner,
    convert_parameters_to_channels,
    convert_channels_to_scheduleblock,
    get_channel,
)


# The first issue would be to redesign the running protocol.
# The new system should depend on the following:
#   1. Matrix to measure => Automatic generation of measuring circuits.
#   2. Factory (?) to generate the following:
#      - Parameter names;
#      - Base circuit template or generator;
#      - Some representation of the base circuit from the parameters;
#      - ...
#   3. Potential custom schedules for the various rotations.
#   4. Potential custom measurement schema.
#   *  Points 3 and 4 could be tweaked via the RunOnBackend protocol by supplying a new instruction map.


def optimize_optimized(
    qubit_specification: Sequence[QubitSpecification],
    Nt: int,
    phys_to_logical: int,
    runner: RunOnBackend,
    padding_type: PaddingType,
    timing_const: TimingConstraints,
    prev_solution: CI_MatrixSolution,
    verbose: bool = False,
    n_shots: Optional[int] = None,
    exit_file: Optional[Path] = None,
    live_file_feed: Optional[Path] = None,
    failsafe: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> tuple[HWCI_MatrixSolutionRuntimeLogicalCircuitRunner, OptimizationTime, Any]:
    if verbose:
        print(f"Starting prep... {datetime.datetime.now()}")
    start_time = time.time()  # Prep time

    ci_identity = f"{prev_solution.ci_matrix.mol_name}.{prev_solution.ci_matrix.distance}.{prev_solution.ci_matrix.n_dim}"

    backend: BACKEND_TYPE = runner.backend

    if live_file_feed is not None:
        live_file_feed.parent.mkdir(parents=True, exist_ok=True)
        if not live_file_feed.exists():
            live_file_feed.write_text(
                "energy,error,Nt,n_shots,repetitions,padding,distance,phys_to_logical\n",
                encoding="utf-8",
            )
        live_file_feed.touch(exist_ok=True)

    if False:
        channel_pattern = re.compile(r"^(?P<channel>\w)(?P<qubit>\d+)_(?P<index>\d+)$")
        channel_map: dict[str, dict[int, int]] = {}
        d_chans = channel_map["d"] = {}
        for i, q in enumerate(qubit_specification):
            d_chans[i] = q.index

        if backend.configuration() is not None:
            device_control_channels = lambda i, j: (
                backend.configuration()
                .control_channels.get(
                    (i, j), backend.configuration().control_channels.get((j, i), [None])
                )[0]
                .index
            )
        else:
            device_control_channels = None

        u_chans = channel_map["u"] = {}
        for q in qubit_specification:
            for oq in sorted(q.coupling_map.keys()):
                if device_control_channels is not None:
                    u_chans[len(u_chans)] = device_control_channels(q.index, oq)
                else:
                    u_chans[len(u_chans)] = len(u_chans)

        # TODO fix the way channels get named. Current scheme allows for problems in the control channels.
        # TODO I will currently ignore the control channels until a better solution is available.
        # TODO Potential fix, when creating s simulation, sort according to the qubit index.
        # TODO Look into the previous TODOs, I might have already dealt with them.
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

    else:
        # Insertion point for new name generation block ->
        new_names = [
            f"{channel}_{i}"
            for channel in get_channel_names(qubit_specification)
            for i in range(Nt)
        ]

        # <-
    # Current prep time is very short !

    end_time = time.time()
    prep_time = end_time - start_time

    start_time = time.time()
    parameter_trajectory: list[Sequence[complex]] = [
        prev_solution.parameters_trajectory[-1],
    ]
    p0 = _complex_to_real(np.asarray(parameter_trajectory[0])).tolist()

    stop_optimization: bool = False

    session: Session

    job_ids: list[str] = []
    raw_measurements: list[pd.DataFrame] = []
    joined_measurements: list[Mapping[str, Mapping[str, int]]] = []
    energies: list[float] = []
    energy_errors: list[float] = []
    mitigated_exp: list[Mapping[str, tuple[float, float]]] = []

    gate_name = "VQE"
    n_qubits = get_n_qubits(backend)
    max_shots = backend.configuration().max_shots
    assert max_shots is not None, "What?!?!"
    repetitions = 1
    if n_shots is None:
        n_shots = max_shots
    elif n_shots > max_shots:
        repetitions = int_div_mod(n_shots, max_shots)
        n_shots = int_div_mod(n_shots, repetitions)

    qubits = [q.index for q in qubit_specification]

    qN = pauli_decomposer(len(qubits))

    pauli_strings_map = {
        k: v
        for k, v in qN(prev_solution.ci_matrix.matrix).items()
        if not np.isclose(v, 0, rtol=rtol, atol=atol)
    }

    build_schedule = compose(
        convert_channels_to_scheduleblock,
        convert_parameters_to_channels(
            phys_to_logical, new_names, padding_type, timing_const
        ),
    )

    # ev_to_energy_error = matrix_ev(prev_solution.ci_matrix)

    ev_to_energy_error = pauli_strings(
        pauli_strings_map, base=prev_solution.ci_matrix.nuclear_repulsion_energy
    )
    marginal_cnts = marginalize_counts(qubits)

    def _base_circuit(schd: ScheduleBlock, ind: Optional[int] = None) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, n_qubits)
        gate = Gate(gate_name, len(qubit_specification), [])
        qc.add_calibration(gate, qubits, schd, [])
        qc.append(gate, qubits, [])
        qc.metadata = {
            "Nt": Nt,
            "padding": padding_type.value,
            "n_shots": n_shots,
            "repetitions": repetitions,
            "ci_identity": ci_identity,
            "measure": "",
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

    def _cost(parameters: npt.ArrayLike) -> float:
        nonlocal stop_optimization
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return -np.inf
        # TODO more complex measurements of different axes
        c_param = _real_to_complex(np.asarray(parameters)).tolist()
        schd = build_schedule(c_param)
        b_circ = _base_circuit(schd)
        # Marker ->
        # Marker <-
        # TODO Add here the entire tomography process
        # z_circ = b_circ.copy()
        # x_circ = b_circ.copy()
        # z_circ.metadata["measure"] = "z"
        # x_circ.metadata["measure"] = "x"
        # for q in qubit_specification:
        #     x_circ.h(q.index)
        #     for qc in (z_circ, x_circ):
        #         qc.measure(q.index, q.index)
        m_circs = measure_pauli_string(qubits, b_circ, pauli_strings_map.keys())
        # TODO Add the new EM here
        leme = LocalReadoutError(physical_qubits=qubits, backend=runner.backend)
        leme.set_run_options(shots=n_shots)
        # leme_circuits = leme.circuits()

        # t_circs = transpile([*leme_circuits, z_circ, x_circ] * repetitions, backend)
        # TODO Find a better way then using a hidden function...
        leme_circuits = leme._transpiled_circuits()
        n_em = len(leme_circuits)
        t_circs = [*leme_circuits, *transpile([*[*m_circs] * repetitions], backend)]

        # HACK
        # In the Runtime protocol, there is no circuit metadata...
        # circuit_metadata = [q.metadata for q in t_circs]

        try:
            # job: qiskit.providers.job.JobV1 = backend.run(t_circs, shots=n_shots)
            job = runner.run(
                t_circs,
                meas_level=MeasLevel.CLASSIFIED,
                meas_return=MeasReturnType.AVERAGE,
                shots=n_shots,
            )
            # backend.properties(refresh=True)
            # em = LocalReadoutMitigator(qubits=qubits, backend=backend)

            job_ids.append(job.job_id())

            if verbose:
                print(f"Running experiment... {datetime.datetime.now()}")
                print(f"Session id - {job.session_id} Job ID - {job.job_id()}")
            # TODO think of how to handle errors in the machine
            job.wait_for_final_state()
            if verbose:
                print(f"Finished experiment... {datetime.datetime.now()}")
        except Exception as e:
            if not failsafe:
                raise e from None
            _log_error(e)
            stop_optimization = True
            return -np.inf

        # TODO maybe add a graceful fail with potential to recover?
        if job.status() != JobStatus.DONE:
            err = RuntimeError(
                f"Job '{job.job_id()}' has failed!\n{job.error_message()}"
            )
            if not failsafe:
                raise err from None
            _log_error(err)
            stop_optimization = True
            return -np.inf

        result: Result = job.result()

        def _result_to_mapping(
            experiment_result: ExperimentResult,
        ) -> Mapping[str, Any]:
            measure = experiment_result.header.metadata.get("measure", "z")
            counts = experiment_result.data.counts
            cnts: dict[str, int] = {}
            for k, v in counts.items():
                kk = k.replace("0x", "")  # FIXME TODO think of something better
                cnts[f"{int(kk):0{n_qubits}b}"] = v
            return {
                "measure": measure,
                **marginal_cnts(cnts),
            }

        def injest_ready(d: Mapping[str, Any]) -> Mapping[str, Any]:
            # FIXME TODO make this more robust?
            if "header" in d and "metadata" in d["header"]:
                d["metadata"] = d["header"]["metadata"]
            if "data" in d and "counts" in d["data"]:
                d["counts"] = d["data"]["counts"]
            return d

        lem_ed = ExperimentData(leme)
        for i in range(n_em):
            lem_ed.add_data(injest_ready(result.results[i].to_dict()))

        leme.analysis.run(lem_ed)
        em: LocalReadoutMitigator = lem_ed.analysis_results()[0].value

        raw_result = pd.DataFrame(map(_result_to_mapping, result.results[n_em:]))
        joined_result = raw_result.groupby("measure").sum().T.to_dict()

        # TODO: Fix to account for measurements of I
        # evs = map_values(em.expectation_value, joined_result)
        # Fixed?
        evs = {
            string: em.expectation_value(
                meas, qubits=get_relevant_qubits(qubits, string)
            )
            for string, meas in joined_result.items()
        }
        mitigated_exp.append(evs)

        energy, error = ev_to_energy_error(evs)

        raw_measurements.append(raw_result)
        joined_measurements.append(joined_result)
        energies.append(energy)
        energy_errors.append(error)
        if verbose:
            print(f"Got the following: {energy:.10f}±{error:.10f}")
        if live_file_feed is not None:
            with live_file_feed.open(mode="a", encoding="utf-8") as io:
                io.write(
                    f"{energy},{error},{Nt},{n_shots},{repetitions},{padding_type.value},{prev_solution.ci_matrix.distance},{phys_to_logical}\n"
                )
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
    # TODO Add session code

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

    qubit_noise = None
    try:
        qubit_noise = get_qubit_noise_parameters(backend, qubits)
        if verbose:
            print(f"Got the noise!")
    except Exception as e:
        if verbose:
            print(f"Failed to get noise.\n{e}")

    hwci_matrix_solution_runtime_logical_circuit_runner = (
        HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(
            ci_matrix=prev_solution.ci_matrix,
            dt=backend.dt,
            success=solution.success,
            qubit_spec=qubit_specification,
            qubit_noise=qubit_noise,
            parameter_names=new_names,
            parameters_trajectory=parameter_trajectory,
            energy_trajectory=energies,
            phys_to_logical=phys_to_logical,
            Nt=Nt,
            energy_error=energy_errors,
            raw_measurements=raw_measurements,
            joined_measurements=joined_measurements,
            additional_data={
                "backend_name": runner.backend_name,
                "job_ids": job_ids,
                "repetitions": repetitions,
                "n_shots": n_shots,
                "total_shots": repetitions * n_shots,
                "padding": padding_type,
                "timing_constraints": timing_const,
                "mitigated_exp": mitigated_exp,
            },
        )
    )
    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=opt_time,
        energy_time=end_time - start_time,
    )
    return (hwci_matrix_solution_runtime_logical_circuit_runner, opt_time, solution)


@dataclass(frozen=True, kw_only=True, slots=True)
class XOptions:
    freq: FreqHunt = field(default_factory=FreqHunt)
    amp: AmpHunt = field(default_factory=AmpHunt)
    shots: int

    @classmethod
    def from_dict(cls: type["XOptions"], d: Mapping[str, Any]) -> Optional["XOptions"]:
        if len(d) == 0:
            return None
        freq = FreqHunt.from_dict(d.get("freq", {}))
        amp = AmpHunt.from_dict(d.get("amp", {}))
        shots = d.get("shots", 1024)

        return cls(freq=freq, amp=amp, shots=shots)


# TODO FIXME maybe do this part better later
def optimize_optimized_custom_x(
    qubit_specification: Sequence[QubitSpecification],
    Nt: int,
    phys_to_logical: int,
    runner: RunOnBackend,
    padding_type: PaddingType,
    timing_const: TimingConstraints,
    prev_solution: CI_MatrixSolution,
    custom_x_options: XOptions,
    base_dir: Path,
    verbose: bool = False,
    n_shots: Optional[int] = None,
    exit_file: Optional[Path] = None,
    live_file_feed: Optional[Path] = None,
    failsafe: bool = True,
    **kwargs,
) -> tuple[HWCI_MatrixSolutionRuntimeLogicalCircuitRunner, OptimizationTime, Any]:
    if verbose:
        print(f"Starting prep... {datetime.datetime.now()}")
    start_time = time.time()  # Prep time

    ci_identity = f"{prev_solution.ci_matrix.mol_name}.{prev_solution.ci_matrix.distance}.{prev_solution.ci_matrix.n_dim}"

    backend: BACKEND_TYPE = runner.backend

    if live_file_feed is not None:
        live_file_feed.parent.mkdir(parents=True, exist_ok=True)
        if not live_file_feed.exists():
            live_file_feed.write_text(
                "energy,error,Nt,n_shots,repetitions,padding,distance,phys_to_logical\n",
                encoding="utf-8",
            )
        live_file_feed.touch(exist_ok=True)

    channel_pattern = re.compile(r"^(?P<channel>\w)(?P<qubit>\d+)_(?P<index>\d+)$")
    channel_map: dict[str, dict[int, int]] = {}
    d_chans = channel_map["d"] = {}
    for i, q in enumerate(qubit_specification):
        d_chans[i] = q.index

    if backend.configuration() is not None:
        device_control_channels = lambda i, j: backend.configuration().control_channels[
            (i, j)
        ][0]
    else:
        device_control_channels = None

    u_chans = channel_map["u"] = {}
    for q in qubit_specification:
        for oq in sorted(q.coupling_map.keys()):
            if device_control_channels is not None:
                u_chans[len(u_chans)] = device_control_channels(q.index, oq)
            else:
                u_chans[len(u_chans)] = len(u_chans)

    # TODO fix the way channels get named. Current scheme allows for problems in the control channels.
    # TODO I will currently ignore the control channels until a better solution is available.
    # TODO Potential fix, when creating s simulation, sort according to the qubit index.
    # TODO Look into the previous TODOs, I might have already dealt with them.
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

    # Current prep time is very short !

    end_time = time.time()
    prep_time = end_time - start_time

    start_time = time.time()
    parameter_trajectory: list[Sequence[complex]] = [
        prev_solution.parameters_trajectory[-1],
    ]
    p0 = _complex_to_real(np.asarray(parameter_trajectory[0])).tolist()

    stop_optimization: bool = False

    session: Session

    job_ids: list[str] = []
    raw_measurements: list[pd.DataFrame] = []
    joined_measurements: list[Mapping[str, Mapping[str, int]]] = []
    energies: list[float] = []
    energy_errors: list[float] = []
    mitigated_exp: list[Mapping[str, tuple[float, float]]] = []

    gate_name = "VQE"
    n_qubits = get_n_qubits(backend)
    max_shots = backend.configuration().max_shots
    assert max_shots is not None, "What?!?!"
    repetitions = 1
    if n_shots is None:
        n_shots = max_shots
    elif n_shots > max_shots:
        repetitions = int_div_mod(n_shots, max_shots)
        n_shots = int_div_mod(n_shots, repetitions)

    qubits = [q.index for q in qubit_specification]

    build_schedule = compose(
        convert_channels_to_scheduleblock,
        convert_parameters_to_channels(
            phys_to_logical, new_names, padding_type, timing_const
        ),
    )

    ev_to_energy_error = matrix_ev(prev_solution.ci_matrix)
    marginal_cnts = marginalize_counts(qubits)

    freq_mult = 10 ** get_si_mult(custom_x_options.freq.units[:-2])
    freq_start = custom_x_options.freq.start * freq_mult
    freq_stop = custom_x_options.freq.stop * freq_mult

    base_factors = BaseDragPulseFactors(
        parameters={
            "beta": {"func": "=", "value": 0},
            "duration": {"func": "=", "value": 64},
            "sigma": {"func": "=", "value": 32},
        }
    )

    # TODO generalize for more than one qubit
    if verbose:
        print(f"Starting the hunt for 0 -> 1 frequency, {datetime.datetime.now()}.")

    freq01, job_id_freq01 = find_01_freq(
        runner,
        qubits[0],
        base_dir,
        custom_x_options.freq.number,
        freq_start,
        freq_stop,
        custom_x_options.shots,
        base_factors=base_factors,
    )
    if verbose:
        print(f"Got the following frequency {freq01:.2e} Hz.")
        print(f"Starting the hunt for the amplitude, {datetime.datetime.now()}.")

    amp01, job_id_amp01 = find_01_amp(
        runner,
        qubits[0],
        base_dir,
        freq01,
        custom_x_options.amp.number,
        custom_x_options.amp.start,
        custom_x_options.amp.stop,
        custom_x_options.shots,
        base_factors=base_factors,
    )
    if verbose:
        print(
            f"Found the following π amplitude {amp01:.2e}, will use {amp01/2:.2e} for π/2."
        )

    with pulse.build(name="custom H", backend=runner.backend) as ch_sched:
        d0 = pulse.drive_channel(qubits[0])
        with pulse.frequency_offset(freq01, d0):
            pulse.play(
                pulse.Gaussian(
                    duration=64,
                    sigma=32,
                    amp=amp01 / 2,
                ),
                d0,
            )

    def _base_circuit(schd: ScheduleBlock, ind: Optional[int] = None) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, n_qubits)
        gate = Gate(gate_name, len(qubit_specification), [])
        qc.add_calibration(gate, qubits, schd, [])
        qc.append(gate, qubits, [])
        qc.metadata = {
            "Nt": Nt,
            "padding": padding_type.value,
            "n_shots": n_shots,
            "repetitions": repetitions,
            "ci_identity": ci_identity,
            "measure": "",
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

    def _cost(parameters: npt.ArrayLike) -> float:
        nonlocal stop_optimization
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return -np.inf
        # TODO more complex measurements of different axes
        c_param = _real_to_complex(np.asarray(parameters)).tolist()
        schd = build_schedule(c_param)
        b_circ = _base_circuit(schd)
        # TODO Add here the entire tomography process
        z_circ = b_circ.copy()
        x_circ = b_circ.copy()
        z_circ.metadata["measure"] = "z"
        x_circ.metadata["measure"] = "x"
        ch_gate = Gate("CustomH", 1, [])
        x_circ.add_calibration(ch_gate, qubits[:1], ch_sched, [])
        for q in qubit_specification:
            x_circ.append(ch_gate, qubits[:1], [])
            for qc in (z_circ, x_circ):
                qc.measure(q.index, q.index)
        t_circs = transpile([z_circ, x_circ] * repetitions, backend)

        # HACK
        # In the Runtime protocol, there is no circuit metadata...
        circuit_metadata = [q.metadata for q in t_circs]

        try:
            # job: qiskit.providers.job.JobV1 = backend.run(t_circs, shots=n_shots)
            job = runner.run(
                t_circs,
                meas_level=MeasLevel.CLASSIFIED,
                meas_return=MeasReturnType.AVERAGE,
                shots=n_shots,
            )
            backend.properties(refresh=True)
            em = LocalReadoutMitigator(qubits=qubits, backend=backend)

            job_ids.append(job.job_id())

            if verbose:
                print(f"Running experiment... {datetime.datetime.now()}")
                print(f"Session id - {job.session_id} Job ID - {job.job_id()}")
            # TODO think of how to handle errors in the machine
            job.wait_for_final_state()
            if verbose:
                print(f"Finished experiment... {datetime.datetime.now()}")
        except Exception as e:
            if not failsafe:
                raise e from None
            _log_error(e)
            stop_optimization = True
            return -np.inf

        # TODO maybe add a graceful fail with potential to recover?
        if job.status() != JobStatus.DONE:
            err = RuntimeError(
                f"Job '{job.job_id()}' has failed!\n{job.error_message()}"
            )
            if not failsafe:
                raise err from None
            _log_error(err)
            stop_optimization = True
            return -np.inf

        result: Result = job.result()

        def _result_to_mapping(
            experiment_result: ExperimentResult,
        ) -> Mapping[str, Any]:
            measure = experiment_result.header.metadata.get("measure", "z")
            counts = experiment_result.data.counts
            cnts: dict[str, int] = {}
            for k, v in counts.items():
                kk = k.replace("0x", "")  # FIXME TODO think of something better
                cnts[f"{int(kk):0{n_qubits}b}"] = v
            return {
                "measure": measure,
                **marginal_cnts(cnts),
            }

        raw_result = pd.DataFrame(map(_result_to_mapping, result.results))
        joined_result = raw_result.groupby("measure").sum().T.to_dict()

        evs = map_values(em.expectation_value, joined_result)
        mitigated_exp.append(evs)

        energy, error = ev_to_energy_error(evs)

        raw_measurements.append(raw_result)
        joined_measurements.append(joined_result)
        energies.append(energy)
        energy_errors.append(error)
        if verbose:
            print(f"Got the following: {energy:.10f}±{error:.10f}")
        if live_file_feed is not None:
            with live_file_feed.open(mode="a", encoding="utf-8") as io:
                io.write(
                    f"{energy},{error},{Nt},{n_shots},{repetitions},{padding_type.value},{prev_solution.ci_matrix.distance},{phys_to_logical}\n"
                )
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
    # TODO Add session code

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
    hwci_matrix_solution_runtime_logical_circuit_runner = (
        HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(
            ci_matrix=prev_solution.ci_matrix,
            dt=backend.dt,
            success=solution.success,
            qubit_spec=qubit_specification,
            parameter_names=new_names,
            parameters_trajectory=parameter_trajectory,
            energy_trajectory=energies,
            phys_to_logical=phys_to_logical,
            Nt=Nt,
            energy_error=energy_errors,
            raw_measurements=raw_measurements,
            joined_measurements=joined_measurements,
            additional_data={
                "backend_name": runner.backend_name,
                "job_ids": job_ids,
                "repetitions": repetitions,
                "n_shots": n_shots,
                "total_shots": repetitions * n_shots,
                "padding": padding_type,
                "timing_constraints": timing_const,
                "mitigated_exp": mitigated_exp,
                "custom_x": {
                    "freq_job_id": job_id_freq01,
                    "freq": freq01,
                    "amp_job_id": job_id_amp01,
                    "amp": amp01,
                },
            },
        )
    )
    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=opt_time,
        energy_time=end_time - start_time,
    )
    return (hwci_matrix_solution_runtime_logical_circuit_runner, opt_time, solution)


def optimize_optimized_Ry(
    qubit_specification: Sequence[QubitSpecification],
    runner: RunOnBackend,
    ci_matrix: CI_Matrix,
    verbose: bool = False,
    n_shots: Optional[int] = None,
    exit_file: Optional[Path] = None,
    live_file_feed: Optional[Path] = None,
    failsafe: bool = True,
    **kwargs,
) -> tuple[HWCI_MatrixSolutionRuntimeLogicalCircuitRunner, OptimizationTime, Any]:
    if verbose:
        print(f"Starting prep... {datetime.datetime.now()}")
    start_time = time.time()  # Prep time

    ci_identity = f"{ci_matrix.mol_name}.{ci_matrix.distance}.{ci_matrix.n_dim}"

    backend: BACKEND_TYPE = runner.backend

    if live_file_feed is not None:
        live_file_feed.parent.mkdir(parents=True, exist_ok=True)
        if not live_file_feed.exists():
            live_file_feed.write_text(
                "energy,error,n_shots,repetitions,distance\n",
                encoding="utf-8",
            )
        live_file_feed.touch(exist_ok=True)

    parameter_names = [f"q{q.index}-theta" for q in qubit_specification]

    # Current prep time is very short !

    end_time = time.time()
    prep_time = end_time - start_time

    start_time = time.time()
    parameter_trajectory: list[Sequence[float]] = [
        [
            0.0,
        ]
        * len(qubit_specification)
    ]
    p0 = parameter_trajectory[0]

    stop_optimization: bool = False

    job_ids: list[str] = []
    raw_measurements: list[pd.DataFrame] = []
    joined_measurements: list[Mapping[str, Mapping[str, int]]] = []
    energies: list[float] = []
    energy_errors: list[float] = []
    mitigated_exp: list[Mapping[str, tuple[float, float]]] = []

    n_qubits = get_n_qubits(backend)
    max_shots = backend.configuration().max_shots
    assert max_shots is not None, "What?!?!"
    repetitions = 1
    if n_shots is None:
        n_shots = max_shots
    elif n_shots > max_shots:
        repetitions = int_div_mod(n_shots, max_shots)
        n_shots = int_div_mod(n_shots, repetitions)

    qubits = [q.index for q in qubit_specification]

    ev_to_energy_error = matrix_ev(ci_matrix)
    marginal_cnts = marginalize_counts(qubits)

    base_qc = QuantumCircuit(n_qubits, n_qubits)
    parameter_parameters: list[Parameter] = []
    for qubit, name in zip(qubits, parameter_names):
        base_qc.ry(p := Parameter(name), qubit)
        parameter_parameters.append(p)
    base_qc.metadata = {
        "n_shots": n_shots,
        "repetitions": repetitions,
        "ci_identity": ci_identity,
        "measure": "",
    }

    def _update_trajectory(parameters: npt.ArrayLike) -> None:
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return
        parameter_trajectory.append(parameters)
        if verbose:
            print(
                f"Another point {len(parameter_trajectory)} {datetime.datetime.now()}"
            )

    def _cost(parameters: npt.ArrayLike) -> float:
        nonlocal stop_optimization
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return -np.inf
        # TODO more complex measurements of different axes
        b_circ = base_qc.assign_parameters(
            {param: val for param, val in zip(parameter_parameters, parameters)}
        )
        # TODO Add here the entire tomography process
        z_circ = b_circ.copy()
        x_circ = b_circ.copy()
        z_circ.metadata["measure"] = "z"
        x_circ.metadata["measure"] = "x"
        for q in qubit_specification:
            x_circ.h(q.index)
            for qc in (z_circ, x_circ):
                qc.measure(q.index, q.index)
        t_circs = transpile([z_circ, x_circ] * repetitions, backend)

        # HACK
        # In the Runtime protocol, there is no circuit metadata...
        circuit_metadata = [q.metadata for q in t_circs]

        try:
            # job: qiskit.providers.job.JobV1 = backend.run(t_circs, shots=n_shots)
            job = runner.run(
                t_circs,
                meas_level=MeasLevel.CLASSIFIED,
                meas_return=MeasReturnType.AVERAGE,
                shots=n_shots,
            )
            backend.properties(refresh=True)
            em = LocalReadoutMitigator(qubits=qubits, backend=backend)

            job_ids.append(job.job_id())

            if verbose:
                print(f"Running experiment... {datetime.datetime.now()}")
                print(f"Session id - {job.session_id} Job ID - {job.job_id()}")
            # TODO think of how to handle errors in the machine
            job.wait_for_final_state()
            if verbose:
                print(f"Finished experiment... {datetime.datetime.now()}")
        except Exception as e:
            if not failsafe:
                raise e from None
            _log_error(e)
            stop_optimization = True
            return -np.inf

        # TODO maybe add a graceful fail with potential to recover?
        if job.status() != JobStatus.DONE:
            err = RuntimeError(
                f"Job '{job.job_id()}' has failed!\n{job.error_message()}"
            )
            if not failsafe:
                raise err from None
            _log_error(err)
            stop_optimization = True
            return -np.inf

        result: Result = job.result()

        def _result_to_mapping(
            experiment_result: ExperimentResult,
        ) -> Mapping[str, Any]:
            measure = experiment_result.header.metadata.get("measure", "z")
            counts = experiment_result.data.counts
            cnts: dict[str, int] = {}
            for k, v in counts.items():
                kk = k.replace("0x", "")  # FIXME TODO think of something better
                cnts[f"{int(kk):0{n_qubits}b}"] = v
            return {
                "measure": measure,
                **marginal_cnts(cnts),
            }

        raw_result = pd.DataFrame(map(_result_to_mapping, result.results))
        joined_result = raw_result.groupby("measure").sum().T.to_dict()

        evs = map_values(em.expectation_value, joined_result)
        mitigated_exp.append(evs)

        energy, error = ev_to_energy_error(evs)

        raw_measurements.append(raw_result)
        joined_measurements.append(joined_result)
        energies.append(energy)
        energy_errors.append(error)
        if verbose:
            print(f"Got the following: {energy:.10f}±{error:.10f}")
        if live_file_feed is not None:
            with live_file_feed.open(mode="a", encoding="utf-8") as io:
                io.write(
                    f"{energy},{error},{n_shots},{repetitions},{ci_matrix.distance}\n"
                )
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

    if verbose:
        print("Starting optimization...")
    # TODO Add session code

    solution = scipy.optimize.minimize(
        fun=_cost,
        x0=p0,
        method="COBYLA",
        callback=_update_trajectory,
        **kwargs,
    )

    if stop_optimization:
        print(f"Optimization failed! (look into {log_file})")

    if verbose:
        print(f"Finished optimization... {datetime.datetime.now()}")
    opt_time = end_time - start_time

    start_time = time.time()
    hwci_matrix_solution_runtime_logical_circuit_runner = (
        HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(
            ci_matrix=ci_matrix,
            dt=backend.dt,
            success=solution.success,
            qubit_spec=qubit_specification,
            parameter_names=parameter_names,
            parameters_trajectory=parameter_trajectory,
            energy_trajectory=energies,
            phys_to_logical=1,
            Nt=1,
            energy_error=energy_errors,
            raw_measurements=raw_measurements,
            joined_measurements=joined_measurements,
            additional_data={
                "backend_name": runner.backend_name,
                "job_ids": job_ids,
                "repetitions": repetitions,
                "n_shots": n_shots,
                "total_shots": repetitions * n_shots,
                "mitigated_exp": mitigated_exp,
            },
        )
    )
    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=opt_time,
        energy_time=end_time - start_time,
    )
    return (hwci_matrix_solution_runtime_logical_circuit_runner, opt_time, solution)


def simulate_transmon(
    Nt: int,
    dt: float,
    qubit_specification: Sequence[QubitSpecification],
    ci_matrix: CI_Matrix,
    transmon_dim: int = 3,
    cross_talk: bool = True,
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
    t_span: Optional[tuple[float, float]] = None,
    y0: Optional[QISKIT_STATE] = None,
    single_connection: bool = False,
    complex_amplitude: bool = True,  # TODO add functionality
    random_initial_pulse: bool = False,
    phys_to_logical: int = 1,
    padding_type: Optional[PaddingType] = None,
    timing_const: Optional[TimingConstraints] = None,
    H_schedule: Optional[ScheduleBlock] = None,
    **kwargs,
) -> tuple[HWCI_MatrixSolutionRuntimeLogicalCircuitRunner, OptimizationTime, Any]:
    start_time = time.time()

    solver, q_map, ch_map, signal_maker = generate_solver(
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
    sig_H: Optional[Sequence[Signal]]
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
    mitigated_exp: list[Mapping[str, tuple[float, float]]] = []
    final_state_trajectory: list[QISKIT_STATE] = []

    energy_w_H: list[float] = []
    mitigated_w_H_exp: list[Mapping[str, tuple[float, float]]] = []
    after_H_last_state: list[QISKIT_STATE] = []

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
        schedule = build_schedule(
            parameter_names,
            parameters,
            padding_type=padding_type,
            timing_constraints=timing_const,
            phys_to_logical=phys_to_logical,
        )
        schedule = pad_schedule(schedule, all_channels)
        results = solver.solve(t_span=t_span, y0=y0, signals=schedule)
        last_state: QISKIT_STATE = results.y[-1]
        # TODO Might need to tweak this to allow for smooth operation with Transmon.
        Z = np.diag([1, -1]).astype(complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        H = (Z + X) / np.sqrt(2)
        I = np.eye(2)

        M = ci_matrix.matrix

        Ic = ((M @ I).trace() / 2).real
        Zc = ((M @ Z).trace() / 2).real
        Xc = ((M @ X).trace() / 2).real

        Pstate = projector(last_state)
        Zev = Pstate.expectation_value(Z).real
        Xev = Pstate.expectation_value(X).real

        energy = Pstate.expectation_value(M).real + ci_matrix.nuclear_repulsion_energy
        mit_exp = {
            "z": (Zev, 0.0),
            "x": (Xev, 0.0),
        }

        energy_trajectory.append(energy)
        mitigated_exp.append(mit_exp)
        final_state_trajectory.append(last_state)

        if H_schedule is not None:
            # TODO Add additional simulation of H gate on the state w. measurement split into X and Z
            more_steps = solver.solve(
                t_span=[0, (dur_H + 1) * dt],
                y0=last_state,
                signals=sig_H,
            )
            lastX = more_steps.y[-1]
            # TODO Make more general (more than 1 qubit)
            Xev = projector(lastX).expectation_value(Z).real
            mit_w_H = {
                "z": (Zev, 0.0),
                "x": (Xev, 0.0),
            }
            energy_H = Ic + Xc * Xev + Zc * Zev + ci_matrix.nuclear_repulsion_energy
            # TODO Consider measuring H @ M @ H
            energy_w_H.append(energy_H)
            mitigated_w_H_exp.append(mit_w_H)
            after_H_last_state.append(lastX)
            energy = energy_H
            # last_state = |\psi><\psi|
        return energy

    def _cost_float(parameters: Sequence[float]) -> float:
        return __cost_complex(np.asarray(parameters, dtype=complex))

    def _const_complex(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(_real_to_complex(np.asarray(parameters))) ** 2 - 1)

    def _const_float(parameters: Sequence[float]) -> npt.ArrayLike:
        return -(np.abs(parameters) ** 2 - 1)

    prep_time = time.time() - start_time
    start_time = time.time()

    if complex_amplitude:
        constraint = {"type": "ineq", "fun": _const_complex}
        solution = scipy.optimize.minimize(
            fun=_cost_complex,
            x0=_complex_to_real(p0),
            method="COBYLA",
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

    ci_matrix_solution = HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(
        ci_matrix=ci_matrix,
        dt=dt,
        success=solution.success,
        qubit_spec=qubit_specification,
        qubit_noise=qubit_noise_model,
        parameter_names=parameter_names,
        parameters_trajectory=parameters_trajectory,
        energy_trajectory=energy_trajectory,
        phys_to_logical=phys_to_logical,
        Nt=Nt,
        energy_error=np.zeros(len(energy_trajectory)).tolist(),
        raw_measurements=[],
        joined_measurements=[],
        additional_data={
            "backend_name": "Simulation",
            "transmon_dim": transmon_dim,
            "padding": padding_type,
            "timing_constraints": timing_const,
            "mitigated_exp": mitigated_exp,
            "mitigated_w_H_exp": mitigated_w_H_exp,
            "last_states": final_state_trajectory,
            "last_states_w_H": after_H_last_state,
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


# This is a quick fix function
# FIXME make this more robust to changes


def optimize_optimized_custom_order(
    qubit_specification: Sequence[QubitSpecification],
    channel_order_str: str,
    phys_to_logical: int,
    runner: RunOnBackend,
    padding_type: PaddingType,
    timing_const: TimingConstraints,
    # prev_solution: CI_MatrixSolution,
    ci_matrix: CI_Matrix,
    verbose: bool = False,
    n_shots: Optional[int] = None,
    exit_file: Optional[Path] = None,
    live_file_feed: Optional[Path] = None,
    failsafe: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> tuple[HWCI_MatrixSolutionRuntimeLogicalCircuitRunner, OptimizationTime, Any]:
    if verbose:
        print(f"Starting prep... {datetime.datetime.now()}")
    start_time = time.time()  # Prep time

    ci_identity = f"{ci_matrix.mol_name}.{ci_matrix.distance}.{ci_matrix.n_dim}"

    backend: BACKEND_TYPE = runner.backend

    if live_file_feed is not None:
        live_file_feed.parent.mkdir(parents=True, exist_ok=True)
        if not live_file_feed.exists():
            live_file_feed.write_text(
                "energy,error,Nt,n_shots,repetitions,padding,distance,phys_to_logical\n",
                encoding="utf-8",
            )
        live_file_feed.touch(exist_ok=True)

    # TODO Make prettier, this is a quick and dirty hack
    channel_order = [
        (spec[0].lower(), int(spec[1:])) for spec in channel_order_str.split("_")
    ]
    new_names = []
    cur = 0
    for channel_type, length in channel_order:
        assert channel_type in (
            "d",
            "u",
            "a",
        ), f"What is this channel type? {channel_type=}"
        for channel in get_channel_names(
            qubit_specification, channel_types=channel_type
        ):
            new_names.extend([f"{channel}_{i+cur}" for i in range(length)])
        cur += length
    Nt = len(new_names)

    # <-
    # Current prep time is very short !

    end_time = time.time()
    prep_time = end_time - start_time

    start_time = time.time()
    parameter_trajectory: list[Sequence[complex]] = [
        # prev_solution.parameters_trajectory[-1],
        [
            0 + 0j,
        ]
        * len(new_names),
    ]
    p0 = _complex_to_real(np.asarray(parameter_trajectory[0])).tolist()

    stop_optimization: bool = False

    session: Session

    job_ids: list[str] = []
    raw_measurements: list[pd.DataFrame] = []
    joined_measurements: list[Mapping[str, Mapping[str, int]]] = []
    energies: list[float] = []
    energy_errors: list[float] = []
    mitigated_exp: list[Mapping[str, tuple[float, float]]] = []

    gate_name = "VQE"
    n_qubits = get_n_qubits(backend)
    max_shots = backend.configuration().max_shots
    assert max_shots is not None, "What?!?!"
    repetitions = 1
    if n_shots is None:
        n_shots = max_shots
    elif n_shots > max_shots:
        repetitions = int_div_mod(n_shots, max_shots)
        n_shots = int_div_mod(n_shots, repetitions)

    qubits = [q.index for q in qubit_specification]

    qN = pauli_decomposer(len(qubits))

    pauli_strings_map = {
        k: v
        for k, v in qN(ci_matrix.matrix).items()
        if not np.isclose(v, 0, rtol=rtol, atol=atol)
    }

    # Marker ->
    # build_schedule = compose(
    #     convert_channels_to_scheduleblock,
    #     convert_parameters_to_channels(
    #         phys_to_logical, new_names, padding_type, timing_const
    #     ),
    # )
    build_schedule = compose(
        convert_ordered_channels_to_schedule,
        convert_parameters_to_channels_ordered(
            qubit_specification,
            channel_order,
            phys_to_logical,
            new_names,
            padding_type,
            timing_const,
        ),
    )

    # ev_to_energy_error = matrix_ev(prev_solution.ci_matrix)

    ev_to_energy_error = pauli_strings(
        pauli_strings_map, base=ci_matrix.nuclear_repulsion_energy
    )
    marginal_cnts = marginalize_counts(qubits)

    order_string = "-".join(
        [f"{channel.upper()}{length}" for channel, length in channel_order]
    )

    def _base_circuit(schd: ScheduleBlock, ind: Optional[int] = None) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, n_qubits)
        gate = Gate(gate_name, len(qubit_specification), [])
        qc.add_calibration(gate, qubits, schd, [])
        qc.append(gate, qubits, [])
        qc.metadata = {
            "order": order_string,
            "padding": padding_type.value,
            "n_shots": n_shots,
            "repetitions": repetitions,
            "ci_identity": ci_identity,
            "measure": "",
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

    def _cost(parameters: npt.ArrayLike) -> float:
        nonlocal stop_optimization
        if stop_optimization or (exit_file is not None and exit_file.exists()):
            return -np.inf
        # TODO more complex measurements of different axes
        c_param = _real_to_complex(np.asarray(parameters)).tolist()
        schd = build_schedule(c_param)
        b_circ = _base_circuit(schd)
        # Marker ->
        # Marker <-
        # TODO Add here the entire tomography process
        # z_circ = b_circ.copy()
        # x_circ = b_circ.copy()
        # z_circ.metadata["measure"] = "z"
        # x_circ.metadata["measure"] = "x"
        # for q in qubit_specification:
        #     x_circ.h(q.index)
        #     for qc in (z_circ, x_circ):
        #         qc.measure(q.index, q.index)
        m_circs = measure_pauli_string(qubits, b_circ, pauli_strings_map.keys())
        # TODO Add the new EM here
        leme = LocalReadoutError(physical_qubits=qubits, backend=runner.backend)
        leme.set_run_options(shots=n_shots)
        # leme_circuits = leme.circuits()

        # t_circs = transpile([*leme_circuits, z_circ, x_circ] * repetitions, backend)
        # TODO Find a better way then using a hidden function...
        leme_circuits = leme._transpiled_circuits()
        n_em = len(leme_circuits)
        t_circs = [*leme_circuits, *transpile([*[*m_circs] * repetitions], backend)]

        # HACK
        # In the Runtime protocol, there is no circuit metadata...
        # circuit_metadata = [q.metadata for q in t_circs]

        try:
            # job: qiskit.providers.job.JobV1 = backend.run(t_circs, shots=n_shots)
            job = runner.run(
                t_circs,
                meas_level=MeasLevel.CLASSIFIED,
                meas_return=MeasReturnType.AVERAGE,
                shots=n_shots,
            )
            # backend.properties(refresh=True)
            # em = LocalReadoutMitigator(qubits=qubits, backend=backend)

            job_ids.append(job.job_id())

            if verbose:
                print(f"Running experiment... {datetime.datetime.now()}")
                print(f"Session id - {job.session_id} Job ID - {job.job_id()}")
            # TODO think of how to handle errors in the machine
            job.wait_for_final_state()
            if verbose:
                print(f"Finished experiment... {datetime.datetime.now()}")
        except Exception as e:
            if not failsafe:
                raise e from None
            _log_error(e)
            stop_optimization = True
            return -np.inf

        # TODO maybe add a graceful fail with potential to recover?
        if job.status() != JobStatus.DONE:
            err = RuntimeError(
                f"Job '{job.job_id()}' has failed!\n{job.error_message()}"
            )
            if not failsafe:
                raise err from None
            _log_error(err)
            stop_optimization = True
            return -np.inf

        result: Result = job.result()

        def _result_to_mapping(
            experiment_result: ExperimentResult,
        ) -> Mapping[str, Any]:
            measure = experiment_result.header.metadata.get("measure", "z")
            counts = experiment_result.data.counts
            cnts: dict[str, int] = {}
            for k, v in counts.items():
                kk = k.replace("0x", "")  # FIXME TODO think of something better
                cnts[f"{int(kk):0{n_qubits}b}"] = v
            return {
                "measure": measure,
                **marginal_cnts(cnts),
            }

        def injest_ready(d: Mapping[str, Any]) -> Mapping[str, Any]:
            # FIXME TODO make this more robust?
            if "header" in d and "metadata" in d["header"]:
                d["metadata"] = d["header"]["metadata"]
            if "data" in d and "counts" in d["data"]:
                d["counts"] = d["data"]["counts"]
            if "counts" in d and any(
                map(lambda s: s.startswith("0x"), d["counts"].keys())
            ):
                nc = {f"{int(k, 16):0{n_qubits}b}": v for k, v in d["counts"].items()}
                d["counts"] = nc
            return d

        lem_ed = ExperimentData(leme)
        for i in range(n_em):
            lem_ed.add_data(injest_ready(result.results[i].to_dict()))

        leme.analysis.run(lem_ed)
        em: LocalReadoutMitigator = lem_ed.analysis_results()[0].value

        raw_result = pd.DataFrame(map(_result_to_mapping, result.results[n_em:]))
        joined_result = raw_result.groupby("measure").sum().T.to_dict()

        # TODO: Fix to account for measurements of I
        # evs = map_values(em.expectation_value, joined_result)
        # Fixed?
        evs = {
            string: em.expectation_value(
                meas, qubits=get_relevant_qubits(qubits, string)
            )
            for string, meas in joined_result.items()
        }
        mitigated_exp.append(evs)

        energy, error = ev_to_energy_error(evs)

        raw_measurements.append(raw_result)
        joined_measurements.append(joined_result)
        energies.append(energy)
        energy_errors.append(error)
        if verbose:
            print(f"Got the following: {energy:.10f}±{error:.10f}")
        if live_file_feed is not None:
            with live_file_feed.open(mode="a", encoding="utf-8") as io:
                io.write(
                    f"{energy},{error},{Nt},{n_shots},{repetitions},{padding_type.value},{ci_matrix.distance},{phys_to_logical}\n"
                )
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
    # TODO Add session code

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

    qubit_noise = None
    try:
        qubit_noise = get_qubit_noise_parameters(backend, qubits)
        if verbose:
            print(f"Got the noise!")
    except Exception as e:
        if verbose:
            print(f"Failed to get noise.\n{e}")

    hwci_matrix_solution_runtime_logical_circuit_runner = (
        HWCI_MatrixSolutionRuntimeLogicalCircuitRunner(
            ci_matrix=ci_matrix,
            dt=backend.dt,
            success=solution.success,
            qubit_spec=qubit_specification,
            qubit_noise=qubit_noise,
            parameter_names=new_names,
            parameters_trajectory=parameter_trajectory,
            energy_trajectory=energies,
            phys_to_logical=phys_to_logical,
            Nt=Nt,
            energy_error=energy_errors,
            raw_measurements=raw_measurements,
            joined_measurements=joined_measurements,
            additional_data={
                "backend_name": runner.backend_name,
                "job_ids": job_ids,
                "repetitions": repetitions,
                "n_shots": n_shots,
                "total_shots": repetitions * n_shots,
                "padding": padding_type,
                "timing_constraints": timing_const,
                "mitigated_exp": mitigated_exp,
                "order": order_string,
            },
        )
    )
    end_time = time.time()

    opt_time = OptimizationTime(
        prep_time=prep_time,
        optimization_time=opt_time,
        energy_time=end_time - start_time,
    )
    return (hwci_matrix_solution_runtime_logical_circuit_runner, opt_time, solution)


def convert_parameters_to_channels_ordered(
    qubits: Sequence[QubitSpecification],
    channel_order: Sequence[tuple[str, int]],
    phys_to_logical: int,
    parameter_names: Sequence[str],
    padding_type: PaddingType,
    timing_const: TimingConstraints,
) -> Callable[[Sequence[complex]], Sequence[Mapping[str, Sequence[complex]]]]:
    # TODO Add testing for padding and timing_const

    channels = [
        get_channel_names(qubits, channel_types=channel_type)
        for channel_type, _ in channel_order
    ]

    def _internal(
        values: Sequence[complex],
    ) -> Sequence[Mapping[str, Sequence[complex]]]:
        ret: list[dict[str, list[complex]]] = []

        prev = 0

        # import pdb; pdb.set_trace()
        for i, (_, length) in enumerate(channel_order):
            N_channel = len(channels[i])
            cur = length * N_channel
            cur_names = parameter_names[prev : prev + cur]
            cur_values = values[prev : prev + cur]
            cur_ret = {channel: [] for channel in channels[i]}
            # print(cur_ret.keys())
            for name, value in zip(cur_names, cur_values):
                channel, _ = name.split("_", maxsplit=2)
                cur_ret[channel].extend(
                    [
                        value,
                    ]
                    * phys_to_logical
                )

            prev += cur

            ret.append(
                {
                    k: padding_type.pad(v, timing_const).tolist()
                    for k, v in cur_ret.items()
                }
            )

        return ret

    return _internal


def convert_ordered_channels_to_schedule(
    ordered_channels: Sequence[Mapping[str, Sequence[complex]]]
) -> ScheduleBlock:
    with build(name="VQE") as schd:
        with align_sequential():
            for section in ordered_channels:
                with align_left():
                    for key, values in section.items():
                        channel = get_channel(key)
                        waveform = Waveform(
                            samples=np.array(values),
                            epsilon=0.2,
                            limit_amplitude=False,
                        )
                        play(waveform, channel)
    return schd
