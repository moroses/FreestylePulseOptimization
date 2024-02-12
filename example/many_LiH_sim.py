#!/usr/bin/env python
"""This file aims to run many different styles of simulations for finding the ground state energy of LiH.
Author: Mor M. Roses
"""

have_jax: bool = False
solver_options = {}
try:
    import jax

    have_jax = True
    device_list = jax.devices()
    print(f"Found the following devices: {device_list}.\nUsing the first one.")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", device_name := device_list[0].platform)
    from qiskit_dynamics.array import Array

    Array.set_default_backend("jax")
    solver_options["method"] = "jax_odeint"
    solver_options["jit"] = True
except:
    ...

from pickle import dumps
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import product
from typing import Any, Callable, Optional, NamedTuple
from collections.abc import Mapping, Sequence

from FreestylePulseOptimization import (
    get_device_parameters,
    get_qubit_noise_parameters,
    load_all_ci_matrices_from_folder,
    PaddingType,
    TimingConstraints,
    STANDARD_TIMING,
    CI_Matrix,
)
from FreestylePulseOptimization.runners import RuntimeRunner
from FreestylePulseOptimization.simulations.main import (
    get_channel_list,
    ChannelProperties,
    optimize_transmon,
    SimulationCI_MatrixSolution,
    OptimizationTime,
)
from FreestylePulseOptimization.pulses import QubitNoiseParameters, QubitSpecification


def find_ci_distance(
    distance: float,
) -> Callable[[Sequence[CI_Matrix]], Optional[CI_Matrix]]:
    def _internal(cis: Sequence[CI_Matrix]) -> Optional[CI_Matrix]:
        for ci in cis:
            if ci.distance == distance:
                return ci
        return None

    return _internal


class SimulationResult(NamedTuple):
    solution: SimulationCI_MatrixSolution
    times: OptimizationTime
    rhobeg: float
    maxiter: int


class SimulationParameter(NamedTuple):
    ci: CI_Matrix
    channel_properties: Mapping[str, ChannelProperties]
    qubit_specs: Sequence[QubitSpecification]
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]]
    timing_constraints: TimingConstraints
    transmon_dim: int
    dt: float
    rhobeg: float
    maxiter: int
    Nt: int
    solver_options: Mapping[str, Any] = {}


def sim_param_to_name(parameter: SimulationParameter) -> str:
    ci_desc = f"{parameter.ci.mol_name}-{parameter.ci.distance}-{parameter.ci.n_dim}"
    R, padding = parameter.channel_properties["d0"]
    d_desc = f"{R}-{padding.name}"
    R, padding = parameter.channel_properties["u0"]
    u_desc = f"{R}-{padding.name}"
    min_desc = f"{parameter.rhobeg}-{parameter.maxiter}-{parameter.Nt}"
    return f"{ci_desc}|{d_desc}|{u_desc}|{min_desc}"


def action_item(parameter: SimulationParameter) -> SimulationResult:
    solver_options = parameter.solver_options
    solution, times, _ = optimize_transmon(
        Nt=parameter.Nt,
        qubit_specification=parameter.qubit_specs,
        qubit_noise_model=parameter.qubit_noise_model,
        channel_properties=parameter.channel_properties,
        dt=parameter.dt,
        transmon_dim=parameter.transmon_dim,
        timing_const=parameter.timing_constraints,
        ci_matrix=parameter.ci,
        method="COBYLA",
        options={
            "rhobeg": parameter.rhobeg,
            "maxiter": parameter.maxiter,
            "disp": not False,
        },
        tol=5e-8,
        verbose=True,
        solver_options=solver_options,
    )
    result = SimulationResult(
        solution=solution,
        times=times,
        rhobeg=parameter.rhobeg,
        maxiter=parameter.maxiter,
    )
    base_dir = Path.cwd() / "NEW-simulations"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / f"{sim_param_to_name(parameter)}.pkl").write_bytes(dumps(result))
    return result


def make_simulation_parameter(
    ci: CI_Matrix,
    transmon_dim: int,
    dt: float,
    qubit_specs: Sequence[QubitSpecification],
    timing_constraints: TimingConstraints,
    solver_options: Mapping[str, Any] = {},
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
) -> Callable[
    [int, int, float, int, PaddingType, int, PaddingType], SimulationParameter
]:
    channels = get_channel_list(qubit_specs)

    def _internal(
        Nt: int,
        maxiter: int,
        rhobeg: float,
        dR: int,
        dP: PaddingType,
        uR: int,
        uP: PaddingType,
    ) -> SimulationParameter:
        ch_props: dict[str, ChannelProperties] = {}
        for channel in channels.values():
            ch_props[channel.drive] = ChannelProperties(dR, dP)
            for con in channel.control.values():
                ch_props[con] = ChannelProperties(uR, uP)
        return SimulationParameter(
            Nt=Nt,
            ci=ci,
            channel_properties=ch_props,
            qubit_specs=qubit_specs,
            qubit_noise_model=qubit_noise_model,
            transmon_dim=transmon_dim,
            dt=dt,
            maxiter=maxiter,
            rhobeg=rhobeg,
            timing_constraints=timing_constraints,
            solver_options=solver_options,
        )

    return _internal


def main() -> None:
    # Try to load jax
    # If jax is here, try to use gpu and than cpu

    qubit_specs: Sequence[QubitSpecification]
    dt: float
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]]
    try:
        runner = RuntimeRunner(
            channel="ibm_quantum",
            instance="ibm-q-research/bar-ilan-uni-2/main",
            backend_name="ibm_osaka",
        )
        backend = runner.backend
        qubit_specs, dt = get_device_parameters(backend, qubits := (0, 1, 2))
        qubit_noise_model = None  # For the time being
        timing_const = TimingConstraints.from_backend(runner.backend)
        qubit_noise_model = get_qubit_noise_parameters(runner.backend, qubits)
    except:
        print("Backend did not work.\nUsing made up numbers...")
        dt = 2.22e-1
        qubit_specs = [
            QubitSpecification(0, 5, -0.01, 0.1, {1: 0.001}, {1: 0}),
            QubitSpecification(1, 5.1, -0.01, 0.1, {0: 0.001, 2: 0.001}, {0: 1, 2: 2}),
            QubitSpecification(2, 4.9, -0.01, 0.1, {1: 0.001}, {1: 3}),
        ]
        qubit_noise_model = None
        qubit_noise_model = [
            QubitNoiseParameters(q.index, 300, None) for q in qubit_specs
        ]
        timing_const = STANDARD_TIMING

    ci = find_ci_distance(1.5)(
        load_all_ci_matrices_from_folder("data/LiH_dist_8configs", 8)
    )
    assert ci is not None, "Where is my CI Matrix?!?!"
    Nts = [2, 5, 10, 20][:1]
    maxiters = [
        10,
        250,
        2_000,
        4_000,
    ][:1]
    rhobegs = [
        0.05,
        0.5,
        1.0,
    ][1:2]
    Rs = [10, 50, 100][:1]
    Ps = [PaddingType.LEFT, PaddingType.RIGHT, PaddingType.MIDDLE][:1]
    all_params = product(
        Nts,
        maxiters,
        rhobegs,
        Rs,
        Ps,
        Rs,
        Ps,
    )
    all_params = list(all_params)

    print(all_params)

    sim_params: list[SimulationParameter] = []
    f = make_simulation_parameter(
        qubit_specs=qubit_specs,
        qubit_noise_model=qubit_noise_model,
        dt=dt,
        ci=ci,
        transmon_dim=3,
        timing_constraints=timing_const,
        solver_options=solver_options,
    )
    for parm in tqdm(all_params, desc="Preparing parameters"):
        sim_params.append(f(*parm))

    print("Starting...")
    action_function = action_item
    results = None
    if False and (not have_jax or device_name == "cpu"):
        results = process_map(
            action_function,
            sim_params,
            max_workers=10,
            chunksize=50,
            desc="Simulating...",
        )
    else:
        results = [action_function(p) for p in tqdm(sim_params, desc="Simulating...")]
    print("Saving...")
    (Path.cwd() / "all-results.pkl").write_bytes(dumps(results))


if __name__ == "__main__":
    main()
