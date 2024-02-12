import numpy as np
import numpy.typing as npt

from typing import Callable, Any, ClassVar, Optional
from collections.abc import Sequence, Iterable, Mapping, MutableMapping
from numpy.fft import rfft, rfftfreq

from dataclasses import dataclass, field
from pathlib import Path

from qiskit.pulse import library
from qiskit.circuit.parameter import Parameter

from ..runners.protocol import get_default_drag_parameters, BACKEND_TYPE

from functools import partial
from operator import add, sub, mul, truediv


def get_default_parameters(
    backend: BACKEND_TYPE, qubit: int
) -> MutableMapping[str, Any]:
    parameters = dict(get_default_drag_parameters(backend, qubit, "x"))
    # parameters["duration"] = 64
    # TODO Look at exact parameters
    return parameters


def extrema(values: Iterable[float]) -> tuple[float, float]:
    return min(values), max(values)


def transform_range(
    new_min: float, new_max: float, old_min: float, old_max: float
) -> Callable[[float], float]:
    old_range = old_max - old_min
    new_range = new_max - new_min
    m = new_range / old_range

    def _internal(value: float) -> float:
        return (value - old_min) * m + new_min

    return _internal


def lorenzian(
    x: npt.NDArray[float] | float, A: float, q_freq: float, B: float, C: float
) -> float:
    A /= np.pi
    shift_x = x - q_freq
    denom = shift_x**2 + B**2
    return A * B / denom + C


def lorenzian_fit(
    x: npt.ArrayLike | float,
    amplitude: float,
    center: float,
    width: float,
    vertical_shift: float,
) -> npt.ArrayLike | float:
    y = np.atleast_1d(x).copy()
    y -= center
    y /= width
    y = 1 + y**2
    y = amplitude / y
    y += vertical_shift
    return y


def lorenzian_guess(
    x: npt.NDArray[float], y: npt.NDArray[float]
) -> tuple[float, float, float, float]:
    ymin, ymax = extrema(y)
    amplitude = (ymax + ymin) / 2
    vertical_shift = ymin
    center = x[np.argmax(y)]

    width_inds = y < vertical_shift + amplitude / 2
    width_x = x[~width_inds]
    min_w_x, max_w_x = extrema(width_x)
    width = max_w_x - min_w_x

    return amplitude, center, width, vertical_shift


# def cos_fit(x: npt.NDArray[float]|float, A: float, B: float, omega: float, phi: float) -> float:
#     return A * np.cos(2* np.pi * omega * x + 2*np.pi * phi) + B


def cos_fit(
    x: npt.ArrayLike | float,
    amplitude: float,
    omega: float,
    phi: float,
    vertical_shift: float,
) -> npt.ArrayLike | float:
    y = 2 * np.pi * (omega * x + phi)
    y = np.cos(y)
    y *= amplitude
    y += vertical_shift
    return y


def cos_fit_guess(
    x: npt.NDArray[float], y: npt.NDArray[float]
) -> tuple[float, float, float, float]:
    ymin, ymax = extrema(y)
    amplitude = (ymax - ymin) / 2
    vertical_shift = (ymax + ymin) / 2

    fx = rfftfreq(n=len(x), d=np.diff(x).mean())
    fy = rfft(y - y.mean())

    omega = fx[np.argmax(np.abs(fy))]

    phi = np.arccos((ymax - vertical_shift) / amplitude) / 2 / np.pi
    xom = x[np.argmax(y)]
    if xom != 0:
        phi /= 2 * np.pi * omega * xom
    return amplitude, omega, phi, vertical_shift


@dataclass(frozen=True, kw_only=True)
class FreqHunt:
    start: float = -30
    stop: float = 30
    number: int = 60
    units: str = "MHz"

    @classmethod
    def from_dict(cls: type["FreqHunt"], d: Mapping[str, Any]) -> "FreqHunt":
        start = d.get("start", -30)
        stop = d.get("stop", 30)
        number = d.get("number", 60)
        units = d.get("units", "MHz")
        return cls(start=start, stop=stop, number=number, units=units)


@dataclass(frozen=True, kw_only=True)
class AmpHunt:
    start: float = 0.01
    stop: float = 1.0
    number: int = 60

    @classmethod
    def from_dict(cls: type["AmpHunt"], d: Mapping[str, Any]) -> "AmpHunt":
        start = d.get("start", 0.01)
        stop = d.get("stop", 1.0)
        number = d.get("number", 60)
        return cls(
            start=start,
            stop=stop,
            number=number,
        )


VALUE = int | float | complex | Parameter


@dataclass(frozen=True, kw_only=True)
class BaseDragPulseFactors:
    # TODO make this more generic
    parameters: Mapping[str, Mapping[str, Any]]

    actions: ClassVar[Mapping[str, Callable[[VALUE, VALUE], VALUE]]] = {
        "+": add,
        "-": sub,
        "*": mul,
        "/": truediv,
        "=": lambda _, b: b,
    }

    def apply(
        self: "BaseDragPulseFactors", d: Mapping[str, VALUE]
    ) -> MutableMapping[str, VALUE]:
        ret = dict(d)
        for key, value in self.parameters.items():
            if key not in ret:
                continue
            func_name = value["func"]
            func_value = value["value"]
            func = self.actions[func_name]
            ret[key] = func(ret[key], func_value)
        return ret

    def __call__(
        self: "BaseDragPulseFactors", d: Mapping[str, VALUE]
    ) -> MutableMapping[str, VALUE]:
        return self.apply(d)

    @classmethod
    def from_dict(
        cls: type["BaseDragPulseFactors"], d: Mapping[str, Any]
    ) -> Optional["BaseDragPulseFactors"]:
        if len(d) == 0:
            return None
        parameters: dict[str, Mapping[str, Any]] = {}
        for key, val in d.items():
            if not isinstance(val, Mapping) or "func" not in val or "value" not in val:
                continue
            if val["func"] not in cls.actions:
                raise RuntimeError(f"What function is this? {key=}, {val=}.")
            parameters[key] = val
        return cls(parameters=parameters)


@dataclass(frozen=False, kw_only=True)
class IQOptions:
    slice_obj: slice = field(default_factory=lambda: slice(-1, -16, -1))  # -1:-16:-1
    output_directory: Path = field(default_factory=lambda: Path.cwd() / "leak-test")
    measure_x: bool = True
    freq_hunt_01: FreqHunt = field(default_factory=FreqHunt)
    amp_hunt_01: AmpHunt = field(default_factory=AmpHunt)
    freq_hunt_12: FreqHunt = field(default_factory=FreqHunt)
    amp_hunt_12: AmpHunt = field(default_factory=AmpHunt)
    shots: int = -1
    movie_types: Sequence[str] = field(default_factory=list)
    fig_shots: int = 5_000
    base_factors: Optional[BaseDragPulseFactors] = None

    @classmethod
    def from_dict(cls: type["IQOptions"], d: Mapping[str, Any]) -> "IQOptions":
        slice_d = d.get("slice", {})
        slice_start = slice_d.get("start", -1)
        slice_stop = slice_d.get("stop", -16)
        slice_step = slice_d.get("step", -1)
        slice_obj = slice(slice_start, slice_stop, slice_step)

        output_directory = Path(d.get("output-directory", "./leak-test/"))

        measure_x = d.get("measure-x", True)

        freq_hunt_01 = FreqHunt.from_dict(d.get("01-freq-hunt", {}))
        amp_hunt_01 = AmpHunt.from_dict(d.get("01-amp-hunt", {}))

        freq_hunt_12 = FreqHunt.from_dict(d.get("12-freq-hunt", {}))
        amp_hunt_12 = AmpHunt.from_dict(d.get("12-amp-hunt", {}))

        shots = d.get("shots", -1)

        movie_types = d.get("movie-types", [])

        fig_shots = d.get("fig-shots", 5_000)

        base_factors = BaseDragPulseFactors.from_dict(d.get("base-factors", {}))

        return cls(
            slice_obj=slice_obj,
            output_directory=output_directory,
            measure_x=measure_x,
            shots=shots,
            movie_types=movie_types,
            fig_shots=fig_shots,
            freq_hunt_01=freq_hunt_01,
            amp_hunt_01=amp_hunt_01,
            freq_hunt_12=freq_hunt_12,
            amp_hunt_12=amp_hunt_12,
            base_factors=base_factors,
        )
