from typing import NamedTuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from ..utils import CI_MatrixSolution, QISKIT_STATE
from ..utils import PaddingType
from .. import pulses


def CA_region(center: float) -> tuple[float, float]:
    CA = 0.0016
    return center - CA, center + CA


class QubitChannels(NamedTuple):
    drive: str
    control: Mapping[int, str]


# TODO Consider allowing each channel to have different number of "tiny" pulses
class ChannelProperties(NamedTuple):
    phys_to_logical: int
    padding: PaddingType


def get_channel_list(
    qubit_specification: Sequence[pulses.QubitSpecification],
) -> Mapping[int, QubitChannels]:
    n_control_channels: int = 0
    all_channels: dict[int, QubitChannels] = {}

    qubit_indices = [qubit.index for qubit in qubit_specification]

    for i, qubit in enumerate(qubit_specification):
        control_channels: dict[int, str] = {}
        for j in sorted(qubit.coupling_map.keys()):
            if j not in qubit_indices:
                continue
            control_channels[qubit_indices.index(j)] = f"u{n_control_channels}"
            # control_channels.append(f"u{n_control_channels}")
            n_control_channels += 1
        all_channels[i] = QubitChannels(f"d{i}", control_channels)

    return all_channels


@dataclass(kw_only=True)
class SimulationCI_MatrixSolution(CI_MatrixSolution):
    channel_properties: Mapping[str, ChannelProperties]
    transmon_dim: int
    states: Sequence[QISKIT_STATE]


from qiskit.pulse.channels import DriveChannel, ControlChannel

Channel = DriveChannel | ControlChannel


def get_channel(channel_name: str) -> Channel:
    match channel_name[0]:
        case "d":
            return DriveChannel(int(channel_name[1:]))
        case "u":
            return ControlChannel(int(channel_name[1:]))
        case _:
            raise NotImplementedError(f"I do not know {channel_name}!")
