from typing import Optional
from collections.abc import Sequence, Mapping
import matplotlib.pyplot as plt
from matplotlib import animation, ticker
from matplotlib.axes import Axes
from matplotlib.pyplot import Artist, Figure

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..utils import CI_MatrixSolution, PaddingType, TimingConstraints
from ..runtime.hw_logical2 import convert_parameters_to_channels

from enum import auto, Flag


class EnergyType(Flag):
    RAW = auto()
    MITIGATED = auto()
    IQ = auto()
    LEAKAGE = auto()
    ANGLES = auto()
    # TODO Think about more?


CA = 0.0016
PLOT_COLORS = (
    "#7fc97f",
    "#beaed4",
    "#fdc086",
    "#ffff99",
)


def generate_movie(
    solution: CI_MatrixSolution,
    top_scale: float = 5,
    bot_scale: float = 2,
    energy_type: EnergyType = EnergyType.MITIGATED,
) -> animation.Animation:
    if solution.additional_data is None:
        raise RuntimeError(f"Cannot work with no 'additional_data'. {solution=}")
    add_data = solution.additional_data
    padding: PaddingType = add_data.get("padding", PaddingType.NO)
    timing_constraints: Optional[TimingConstraints] = add_data.get(
        "timing_constraints", None
    )
    dt = solution.dt

    mosaic = [
        ["e-as-i", "pulse"],
        ["e-as-i", "pulse-fft"],
        ["e-as-i-zoom", "e-as-i-zoom"],
        # TODO Add angle handling
    ]
    if energy_type & EnergyType.LEAKAGE:
        mosaic.append(["leakage", "leakage"])
    if energy_type & EnergyType.ANGLES:
        mosaic.append(["angles", "angles"])

    fig, axs = plt.subplot_mosaic(
        mosaic=mosaic, figsize=(2, len(mosaic)) * plt.figaspect(1)
    )
    phys_to_logical = 1
    if hasattr(solution, "phys_to_logical"):
        phys_to_logical = solution.phys_to_logical
    elif "phys_to_logical" in add_data:
        phys_to_logical = add_data["phys_to_logical"]

    convert_trajectory_to_channels = convert_parameters_to_channels(
        phys_to_logical, solution.parameter_names, padding, timing_constraints
    )

    exact = solution.ci_matrix.ci_ground_state_energy

    fill_area = np.array([-CA, CA]) + exact
    zoom_area = np.array([-CA * bot_scale, CA * top_scale]) + exact

    energy = solution.energy_trajectory
    energy_error: Optional[Sequence[float]] = None
    if hasattr(solution, "energy_error"):
        energy_error = solution.energy_error

    parameters_trajectory = solution.parameters_trajectory[: len(energy)]

    iterations = np.arange(len(energy))

    for k in ("", "-zoom"):
        a: Axes = axs[f"e-as-i{k}"]
        a.axhline(exact, color=PLOT_COLORS[-1])
        a.fill_between(
            iterations, *fill_area, color=PLOT_COLORS[-2], alpha=0.3, hatch="\\"
        )
        if EnergyType.RAW & energy_type:
            ...
            # Test for it
        elif EnergyType.MITIGATED & energy_type:
            a.errorbar(
                iterations,
                energy,
                yerr=energy_error,
                color=PLOT_COLORS[-3],
                linewidth=2.5,
                capsize=4,
                markersize=8,
                marker=".",
            )
        elif EnergyType.IQ & energy_type:
            ...
            # Some voodoo
        a.set(xlabel="Iterations []", ylabel="Energy [Ha]")
    axs["e-as-i-zoom"].set_ylim(*zoom_area)

    artists: list[Sequence[Artist]] = []

    for i in iterations:
        pov = [
            axs[f"e-as-i{k}"].scatter(
                iterations[i], energy[i], color=PLOT_COLORS[-4], s=25, zorder=5
            )
            for k in ("", "-zoom")
        ]
        channels = convert_trajectory_to_channels(parameters_trajectory[i])
        ...

    ...
