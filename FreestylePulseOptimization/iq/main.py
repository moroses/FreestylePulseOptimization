#!/usr/bin/env python
"""This file will house the code to execute the IQ measurement for a selected parameters trajectory."""

from datetime import datetime

from typing import Any, overload
from collections.abc import Mapping

from typing_extensions import Literal

from ..runners.protocol import RunOnBackend

from ..utils import CI_MatrixSolution
from ..runtime.utils import Parameter

from ..pulses import get_si_mult

from .utils import IQOptions, get_default_parameters

from . import logic_advanced as advanced

S = CI_MatrixSolution


@overload
def find_the_leak_advanced(
    runner: RunOnBackend,
    parameter: Parameter,
    solution: S,
    run_options: IQOptions = IQOptions(),
    verbose: bool = False,
    field_name: str = "leakage",
    calibration_only: Literal[False] = False,
) -> S: ...


@overload
def find_the_leak_advanced(
    runner: RunOnBackend,
    parameter: Parameter,
    solution: int,
    run_options: IQOptions = IQOptions(),
    verbose: bool = False,
    field_name: str = "leakage",
    calibration_only: Literal[True] = True,
) -> Mapping[str, Any]: ...


# @overload
# def find_the_leak_advanced(
#         runner: RunOnBackend,
#         parameter: Parameter,
#         solution: S|int,
#         run_options: IQOptions = IQOptions(),
#         verbose: bool = False,
#         field_name: str = 'leakage',
#         calibration_only: bool = False) -> S|Mapping[str, Any]:
#     ...


def find_the_leak_advanced(
    runner: RunOnBackend,
    parameter: Parameter,
    solution: S | int,
    run_options: IQOptions = IQOptions(),
    verbose: bool = False,
    field_name: str = "leakage",
    calibration_only: bool = False,
) -> S | Mapping[str, Any]:
    # TODO In the future, might add Freq(Amp)Hunt for the 01 separately from the 12
    base_dir = run_options.output_directory
    base_dir /= f"{field_name}"
    # base_dir /= f"{parameter}"

    base_dir.mkdir(parents=True, exist_ok=True)
    # base_dir /= "leakage-exam"

    if isinstance(solution, int):
        qubit = solution
    elif isinstance(solution, S):
        qubit = solution.qubit_spec[0].index  # TODO remove hard limit to single qubit

    shots = (
        run_options.shots
        if run_options.shots != -1
        else runner.backend.configuration().max_shots
    )

    freq_01_mult = 10 ** get_si_mult(run_options.freq_hunt_01.units[:-2])
    freq_01_start = run_options.freq_hunt_01.start * freq_01_mult
    freq_01_stop = run_options.freq_hunt_01.stop * freq_01_mult
    freq_01_num = run_options.freq_hunt_01.number

    base_factors = run_options.base_factors

    if verbose:
        print(f"Starting to run frequency hunt for 0 -> 1, {datetime.now()}.")
    freq01, job_id_01_freq = advanced.find_01_freq(
        runner,
        qubit,
        base_dir,
        freq_01_num,
        freq_01_start,
        freq_01_stop,
        shots,
        base_factors=base_factors,
    )

    if verbose:
        print(f"Found the following frequency {freq01:.2e} Hz.")
        print(f"Starting to run amplitude hunt for 0 -> 1, {datetime.now()}.")

    amp_01_start = run_options.amp_hunt_01.start
    amp_01_stop = run_options.amp_hunt_01.stop
    amp_01_num = run_options.amp_hunt_01.number

    amp01, job_id_01_amp = advanced.find_01_amp(
        runner,
        qubit,
        base_dir,
        freq01,
        amp_01_num,
        amp_01_start,
        amp_01_stop,
        shots,
        base_factors=base_factors,
    )

    if verbose:
        print(f"Found the following amplitude {amp01:.2e}.")
        print(f"Starting to run frequency hunt for 1 -> 2, {datetime.now()}.")

    default_pulse_parameters01 = get_default_parameters(
        runner.backend, qubit
    )  # Currently use the standard pulse specs
    if base_factors is not None:
        default_pulse_parameters01 = base_factors(default_pulse_parameters01)
    default_pulse_parameters01["amp"] = amp01
    x01_schedule = advanced.freq_offset_drag_schedule(
        runner.backend,
        qubit,
        freq01,
        name="x01",
        **default_pulse_parameters01,
    )

    freq_12_mult = 10 ** get_si_mult(run_options.freq_hunt_12.units[:-2])
    freq_12_start = run_options.freq_hunt_12.start * freq_12_mult
    freq_12_stop = run_options.freq_hunt_12.stop * freq_12_mult
    freq_12_num = run_options.freq_hunt_12.number

    freq12, job_id_12_freq = advanced.find_12_freq(
        runner,
        qubit,
        base_dir,
        freq_12_num,
        freq_12_start,
        freq_12_stop,
        x01_schedule,
        shots,
        base_factors=base_factors,
    )
    if verbose:
        print(f"Found the following frequency {freq12:.2e} Hz.")
        print(f"Starting to run amplitude hunt 1 -> 2, {datetime.now()}.")

    amp_12_start = run_options.amp_hunt_12.start
    amp_12_stop = run_options.amp_hunt_12.stop
    amp_12_num = run_options.amp_hunt_12.number

    amp12, job_id_12_amp = advanced.find_12_amp(
        runner,
        qubit,
        base_dir,
        freq12,
        amp_12_num,
        amp_12_start,
        amp_12_stop,
        x01_schedule,
        shots,
        base_factors=base_factors,
    )
    if verbose:
        print(f"Found the following amplitude {amp12:.2e}.")
        print(f"Find the regions for the |0⟩, |1⟩ and |2⟩ states, {datetime.now()}.")

    default_pulse_parameters12 = get_default_parameters(
        runner.backend, qubit
    )  # Currently use the standard pulse specs
    if base_factors is not None:
        default_pulse_parameters12 = base_factors(default_pulse_parameters12)
    default_pulse_parameters12["amp"] = amp12
    x12_schedule = advanced.freq_offset_drag_schedule(
        runner.backend,
        qubit,
        freq12,
        name="x12",
        **default_pulse_parameters12,
    )

    LDA, LDA_score, (Irange, Qrange), job_id_LDA = advanced.find_012_LDA(
        runner,
        qubit,
        base_dir,
        x01_schedule,
        x12_schedule,
        shots,
        create_figure=5_00,
    )
    if verbose:
        print(f"LDA score {LDA_score:.2f}.")

    if calibration_only:
        return {
            "job_ids": {
                "freq01": job_id_01_freq,
                "amp01": job_id_01_amp,
                "freq12": job_id_12_freq,
                "amp12": job_id_12_amp,
                "LDA": job_id_LDA,
            },
            "fit": {
                "freq01": freq01,
                "amp01": amp01,
                "freq12": freq12,
                "amp12": amp12,
            },
            "Irange": Irange,
            "Qrange": Qrange,
            "LDA": LDA,
        }

    assert isinstance(solution, S), "What did you do?"

    if verbose:
        n_options = len(
            range(len(solution.parameters_trajectory))[run_options.slice_obj]
        )
        print(f"Starting to examine {n_options} parameters, {datetime.now()}.")

    IQ_df, job_id_leakage = measure_leakage_for_solutions(
        runner,
        base_dir,
        {parameter: solution},
        run_options.slice_obj,
        measure_x=run_options.measure_x,
    )

    if verbose:
        print(f"Starting image generation...")

    if solution.additional_data is None:
        solution.additional_data = {}
    solution.additional_data[field_name] = {
        "job_ids": {
            "freq01": job_id_01_freq,
            "amp01": job_id_01_amp,
            "freq12": job_id_12_freq,
            "amp12": job_id_12_amp,
            "LDA": job_id_LDA,
            "leakage": job_id_leakage,
        },
        "fit": {
            "freq01": freq01,
            "amp01": amp01,
            "freq12": freq12,
            "amp12": amp12,
        },
        "Irange": Irange,
        "Qrange": Qrange,
        "LDA": LDA,
        "IQ_df": IQ_df,
    }
    return solution
