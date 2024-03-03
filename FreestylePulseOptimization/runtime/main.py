#!/usr/bin/env python
# Author: Mor M. Roses

from collections.abc import Sequence, Mapping
from typing import Any, Iterable, Optional, Callable

import sys

import qiskit
import qiskit.providers
import qiskit.providers.backend
import qiskit.pulse
from qiskit.visualization.pulse_v2 import IQXStandard
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_ibm_runtime.qiskit_runtime_service import QiskitRuntimeService
from qiskit_ibm_runtime.session import Session

from ..utils import CI_MatrixSolution
from ..iq.utils import IQOptions
from ..iq.main import find_the_leak_advanced
from ..runners.runtime import RuntimeRunner
from ..runners import RetryRuntimeRunner
from ..runners.runtime_retry import try_new_session

from ..utils import OptimizationTime, set_minimal_execution_time
from ..hw import HWCI_MatrixSolution
from .hw_logical2 import XOptions, optimize_optimized as optimize_logical_new
from .hw_logical2 import optimize_optimized_custom_x
from .hw_logical2 import optimize_optimized_custom_order

from ..utils import (
    load_all_ci_matrices_from_folder,
    TimingConstraints,
    create_empty_solution,
)
from ..pulses import get_device_parameters


import numpy as np

import matplotlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt

import datetime
import json
import pickle

from pathlib import Path

import argparse

from .utils import Parameter


def load_service_and_backend(
    conf: Mapping[str, Any]
) -> tuple[QiskitRuntimeService, IBMBackend, Sequence[int], str | None]:
    service_name = conf["service-name"]
    backend_name = conf["backend-name"]
    qubits = conf["qubits"]
    instance = conf.get("instance", None)
    token_file = conf.get("token-file", None)
    token = None
    if token_file is not None:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    # TODO might need additional data for IBM cloud connection
    service = QiskitRuntimeService(channel=service_name, instance=instance, token=token)
    if backend_name == "least_busy":
        backend = service.least_busy(min_num_qubits=len(qubits), simulator=False)
    else:
        backend = service.backend(name=backend_name, instance=instance)
    return service, backend, qubits, instance


def main():
    parser = argparse.ArgumentParser(
        description="Execute a very smart VQE on IBM's quantum experience using Qiskit Runtime."
    )
    parser.add_argument(
        "--file", "-f", type=Path, required=True, help="The JSON input file"
    )

    args = parser.parse_args()
    config = json.loads(args.file.read_text(encoding="utf-8"))

    verbose = config.get("verbose", False)

    if verbose:
        print("Starting...")

    ci_conf: Optional[Mapping[str, Any]] = config.get("ci-matrix", None)
    if ci_conf is None:
        print("Must provide CI matrix files!")
        sys.exit(1)
    ci_matrices = load_all_ci_matrices_from_folder(
        base_dir=ci_conf["directory"], n_dim=int(ci_conf["nq"])
    )

    if verbose:
        print(f"Loaded CI matrices")

    parameters: Iterable[Parameter] = list(
        map(Parameter.fromdict, config.get("parameters", []))
    )

    if "ibm-runtime" not in config:
        print("Must provide with IBM configuration as well!!")
        sys.exit(2)
    ibm_conf = config["ibm-runtime"]

    if verbose:
        print("Connecting to IBM...")
    service, backend, qubits, instance = load_service_and_backend(ibm_conf)
    device_name = backend.name
    if False:
        runner = RuntimeRunner.from_service_and_session(service, device_name)
        runner.open_session(max_time="4h")
    else:
        runner = RetryRuntimeRunner.from_service_and_session(
            service,
            device_name,
        )
        runner.reaction = try_new_session
        # runner.session_kwargs = {
        #         'max_time': '2h',
        #         }

    if verbose:
        print(f"Connected! {device_name}")
        p_str = "\n".join(map(str, parameters))
        print(f"Will do the following: \n{p_str}")

    qubit_spec, dt = get_device_parameters(backend, qubits)
    # TODO Add noise reading?

    timing_const = TimingConstraints.from_backend(backend)

    date_fmt = r"%Y/%m/%d/%H-%M"

    base_dir = Path(config["output-directory"])

    start_time = datetime.datetime.now()

    data_dir = base_dir / "data"
    img_dir = (
        base_dir
        / "images"
        / f"{device_name}"
        / f"{qubits}"
        / f"{start_time:{date_fmt}}"
    )
    hw_dir = (
        data_dir
        / "hardware"
        / f"{device_name}"
        / f"{qubits}"
        / f"{start_time:{date_fmt}}"
    )

    for d in (data_dir, img_dir, hw_dir):
        d.mkdir(parents=True, exist_ok=True)

    hw_results: dict[
        Parameter,
        CI_MatrixSolution,
    ] = {}
    hw_times: dict[Parameter, OptimizationTime] = {}

    hw_config: dict[str, Any] = {
        "stop-file": "./exit-file.file",
        "live-feed-file": None,
    }
    for k in hw_config.keys():
        if k in config:
            hw_config[k] = config[k]
    hw_options: Mapping[str, Any] = config.get("optimizer", {})

    exit_file = Path(hw_config["stop-file"])
    live_file_feed = (
        Path(hw_config["live-feed-file"])
        if hw_config["live-feed-file"] is not None
        else None
    )

    print(f"Exit file is {exit_file.resolve()}.")
    if verbose and live_file_feed is not None:
        print(f"Live feed is here {live_file_feed.resolve()}")

    dist_to_index = {ci.distance: i for i, ci in enumerate(ci_matrices)}

    iq_config = config.get("iq", {})
    iq_options = None
    iq_run = iq_config.get("run", False)
    if iq_run:
        iq_options = IQOptions.from_dict(iq_config)
        if verbose:
            print(f"Will also do IQ\n{iq_options=}")

    custom_x = XOptions.from_dict(ibm_conf.get("custom-x", {}))

    def _hw_task() -> None:
        for i, parameter in enumerate(parameters):
            d_i = dist_to_index[parameter.distance]
            padding = parameter.padding
            Nt = parameter.Nt
            total_shots = parameter.total_shots
            d_ci = ci_matrices[d_i]
            phys_to_logical = parameter.phys_to_logical
            if verbose:
                print(
                    f"Starting to run {d_ci.distance}, {padding.name}, {Nt}, {total_shots}"
                )
                print(f"{parameter=}")
                for i, q in enumerate(qubit_spec):
                    print(f"Qubit {i:03d} = {q}")
            hw_file = (
                hw_dir
                / f"{d_ci.mol_name}"
                / f"{d_ci.distance}"
                / f"{parameter}.{i}.pkl"
                # / f"{repetitions}-{n_shots}-{padding.value}-{Nt}.pkl"
            )
            hw_file.parent.mkdir(parents=True, exist_ok=True)
            ci_sol, _ = create_empty_solution(
                Nt=Nt,
                dt=dt,
                qubit_specification=qubit_spec,
                ci_matrix=d_ci,
                cross_talk=not False,
                single_connection=True,
                complex_amplitude=True,
                random_initial_pulse=False,
                padding_type=padding,
                timing_const=timing_const,
            )
            runner.close_session()
            runner.open_session()
            if iq_run:
                try:
                    # TODO FIXME Make it more general
                    # Again, currently it only works for a single qubit!
                    iq_options.output_directory = hw_file.with_suffix(".d")
                    pre_IQ = find_the_leak_advanced(
                        runner,
                        parameter,
                        qubit_spec[0].index,
                        iq_options,
                        verbose,
                        calibration_only=True,
                        field_name="pre_leakage",
                    )
                except Exception as e:
                    print(f"Failed to do pre - IQ measurements for {parameter}.\n{e=}.")
                    pre_IQ = None
            order: Optional[str] = parameter.additional_data.get("order", None)
            if custom_x is None and not order:
                (
                    hw_sol,
                    hw_sol_times,
                    _,
                ) = optimize_logical_new(
                    qubit_specification=qubit_spec,
                    Nt=Nt,
                    phys_to_logical=phys_to_logical,
                    runner=runner,
                    padding_type=padding,
                    n_shots=total_shots,
                    timing_const=timing_const,
                    prev_solution=ci_sol,
                    verbose=verbose,
                    exit_file=exit_file,
                    live_file_feed=live_file_feed,
                    failsafe=True,
                    **hw_options,
                )
            elif custom_x is None and order:
                (hw_sol, hw_sol_times, _) = optimize_optimized_custom_order(
                    qubit_specification=qubit_spec,
                    channel_order_str=order,
                    channel_alignment_str=parameter.additional_data.get(
                        "alignment", "_".join(["c"] * order.count("_"))
                    ),
                    phys_to_logical=phys_to_logical,
                    runner=runner,
                    padding_type=padding,
                    n_shots=total_shots,
                    timing_const=timing_const,
                    ci_matrix=d_ci,
                    verbose=verbose,
                    exit_file=exit_file,
                    live_file_feed=live_file_feed,
                    failsafe=True,
                    **hw_options,
                )
            else:
                (
                    hw_sol,
                    hw_sol_times,
                    _,
                ) = optimize_optimized_custom_x(
                    qubit_specification=qubit_spec,
                    Nt=Nt,
                    phys_to_logical=phys_to_logical,
                    runner=runner,
                    padding_type=padding,
                    n_shots=total_shots,
                    timing_const=timing_const,
                    prev_solution=ci_sol,
                    custom_x_options=custom_x,
                    base_dir=hw_file.parent / f"{parameter}",
                    verbose=verbose,
                    exit_file=exit_file,
                    live_file_feed=live_file_feed,
                    failsafe=True,
                    **hw_options,
                )

            if iq_run:
                if hw_sol.additional_data is None:
                    hw_sol.additional_data = {}
                hw_sol.additional_data["pre_leakage"] = pre_IQ
            if verbose:
                print(f"Finished running for {parameter}.")

            if iq_run:
                try:
                    hw_sol = find_the_leak_advanced(
                        runner, parameter, hw_sol, iq_options, verbose
                    )
                except Exception as e:
                    print(f"Failed to do IQ measurements for {parameter}.\n{e=}.")
                # hw_sol = find_the_leak(runner, parameter, hw_sol, iq_options, verbose)

            runner.close_session()

            hw_file.parent.mkdir(exist_ok=True, parents=True)
            hw_file.write_bytes(
                pickle.dumps({"solution": hw_sol, "times": hw_sol_times})
            )
            if verbose:
                print(f"Finished {d_ci.distance}, {padding.name}, {Nt}, {total_shots}")
        return

    reservation_time: dict[str, int] = {
        "year": start_time.year,
        "month": start_time.month,
        "day": start_time.day,
        "hour": start_time.hour,
        "minute": start_time.minute,
    }
    reservation_time.update(config.get("reservation", {}))
    reser_time = datetime.datetime(**reservation_time)
    if verbose:
        print(f"Waiting until {reser_time:{date_fmt}}")
    set_minimal_execution_time(not_until=reser_time, function=_hw_task, verbose=verbose)


if __name__ == "__main__":
    main()
