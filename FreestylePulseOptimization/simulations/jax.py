from collections.abc import Mapping, Sequence, MutableSequence
from typing import Callable, TypeVar, Optional
from collections import defaultdict

from qiskit_dynamics import DiscreteSignal, Signal

from ..jax_utils import JAXChannelProperties
from ..utils import TimingConstraints


def func_startswith(startswith: str) -> Callable[[str], bool]:
    def _internal(value: str) -> bool:
        return value.startswith(startswith)

    return _internal


T = TypeVar("T")


def remove_and_apply(remove: str, func: Callable[[str], T]) -> Callable[[str], T]:
    def _internal(value: str) -> T:
        return func(value.replace(remove, ""))

    return _internal


def create_schedule(
    dt: float,
    frequencies: Mapping[str, float],
    parameter_names: Sequence[str],
    channel_properties: Mapping[str, JAXChannelProperties],
    timing_constraints: TimingConstraints,
    drive_channels: Sequence[str],
    control_channels: Sequence[str],
    remove_channels: Optional[Sequence[str]],
) -> Callable[[Sequence[complex]], Sequence[Signal]]:
    def _internal(parameters: Sequence[complex]) -> Sequence[Signal]:
        # channel_map: dict[Channel, MutableSequence[complex]] = defaultdict(list)
        channel_map: dict[str, MutableSequence[complex]] = defaultdict(list)
        for name, value in zip(parameter_names, parameters):
            channel_name, index = name.split("_", maxsplit=1)
            index = int(index)
            # channel = get_channel(channel_name)
            channel = channel_name
            phys_to_logical = channel_properties[channel_name].phys_to_logical
            channel_map[channel].extend(
                [
                    value,
                ]
                * phys_to_logical
            )
        signals: list[Signal] = []

        drive_func = func_startswith("d")
        drop_and_int = remove_and_apply("d", int)
        drive_labels = sorted(filter(drive_func, channel_map.keys()), key=drop_and_int)

        # for d in drive_labels:
        for d in drive_channels:
            if d in drive_labels:
                padding = channel_properties[d].padding
                samples = padding.pad(channel_map[d], timing_constraints)
                signal = DiscreteSignal(
                    dt=dt,
                    samples=samples,
                    carrier_freq=frequencies[d],
                )
            elif remove_channels is not None and d in remove_channels:
                signal = Signal(0)
            else:
                raise RuntimeError(f"What?!?!?")
            signals.append(signal)

        control_func = func_startswith("u")
        drop_and_int = remove_and_apply("u", int)
        control_labels = sorted(
            filter(control_func, channel_map.keys()), key=drop_and_int
        )

        # for u in control_labels:
        for u in control_channels:
            if u in control_labels:
                padding = channel_properties[u].padding
                samples = padding.pad(channel_map[u], timing_constraints)
                signal = DiscreteSignal(
                    dt=dt,
                    samples=samples,
                    carrier_freq=frequencies[u],
                )
            elif remove_channels is not None and u in remove_channels:
                signal = Signal(0)
            else:
                raise RuntimeError("What gives man?!!?")
            signals.append(signal)

        return signals

    return _internal
