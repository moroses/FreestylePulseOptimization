"""Module to get the various features of the qubits from a Backend"""
import datetime
import qiskit
import qiskit.quantum_info
import qiskit_dynamics as qd
import qiskit.pulse
import qiskit.pulse.transforms
import numpy as np
from numpy import typing as npt
import dataclasses
from typing import Callable, Literal, Optional, TypeAlias
from collections.abc import Iterator, Sequence, Iterable, Mapping

from .protocols import DeviceParameterBackend, NoiseParameterBackend

FREQ_FUNCTIONS: Mapping[str, Callable[[float, float], float]] = {
    "DIFFERENCE": lambda x, y: x - y,
    "ONLY_SECOND": lambda x, y: y,
    "ONLY_FIRST": lambda x, y: x,
}
"""Various functions to calculate the frequency shifts of the control channels"""

DEFAULT_FREQ_OF_CONTROL: Callable[[float, float], float] = FREQ_FUNCTIONS["DIFFERENCE"]
"""The default function is 'DIFFERENCE'"""


# %%
@dataclasses.dataclass(frozen=True)
class QubitSpecification:
    """
    Qubit specification for a Transmon.

    Holds the various data that is relevant for the simulation of the qubit.
    """

    index: int
    """The qubit index"""
    freq: float
    """The base frequency"""
    delta: float
    """The anharmonicity of the Transmon"""
    rabi: float
    """The Rabi coupling"""
    coupling_map: Mapping[int, float] = dataclasses.field(default_factory=dict)
    """The coupling mapping of this qubit"""
    control_channels: Mapping[int, int] | None = None
    """The indices of the various control channels of this qubit"""

    def __iter__(self):
        """
        Returns an iterator version of this object

        Returns the features in the following order (index, frequency, anharmonicity, Rabi, coupling map)
        """
        return iter((self.index, self.freq, self.delta, self.rabi, self.coupling_map))


# %%
def get_si_mult(si_prefix: str) -> int:
    """
    Get the SI multiplier

    Returns the SI multiplier

    Parameters
    ----------
    si_prefix : str
        SI prefix

    Returns
    -------
    int
        The SI multiplier
    """
    mult_map: Mapping[str, int] = {
        "Y": 24,
        "Z": 21,
        "E": 18,
        "P": 15,
        "T": 12,
        "G": 9,
        "M": 6,
        "k": 3,
        "h": 2,
        "da": 1,
        "d": -1,
        "c": -2,
        "m": -3,
        "μ": -6,
        "n": -9,
        "p": -12,
        "f": -15,
        "a": -18,
        "z": -21,
        "y": -24,
    }
    return mult_map.get(si_prefix, 0)


# %%
def get_device_parameters(
    backend: DeviceParameterBackend,
    qubits: Sequence[int] | int,
    div_by_two_pi: bool = True,
    freq_units: str = "GHz",
    time_units: str = "ns",
) -> tuple[Sequence[QubitSpecification], float]:
    """
    Extracts specific qubit parameters from the backend, and the sampling rate of the device.

    Extracts specific qubit parameters from the backend and stores them in a `QubitSpecification` object.
    Also, returns the devices sampling rate.

    Parameters
    ----------
    backend : DeviceParameterBackend
        The backend to get the information from.
    qubits : Sequence[int] | int
        What qubit(s) to get data for
    div_by_two_pi : bool
        Divide units by 2π? Defaults True
    freq_units : str
        Frequency units. Defaults to "GHz"
    time_units : str
        Time units. Defaults to "ns"

    Returns
    -------
    tuple[Sequence[QubitSpecification], float]
        Returns a list of `QubitSpecification` and the sampling rate of the device.
    """
    assert freq_units.lower().endswith(
        "hz"
    ), f"I do not know these frequency units! {freq_units}"
    assert time_units.lower().endswith(
        "s"
    ), f"I do not know these time units! {time_units}"

    freq_mult = 10 ** get_si_mult(freq_units[:-2])

    freq_mult *= (2 * np.pi) if div_by_two_pi else 1.0

    time_mult = 10 ** get_si_mult(time_units[:-1])
    dt = backend.configuration().dt / time_mult

    if not isinstance(qubits, Iterable):
        qubits = [qubits]
    ret: list[QubitSpecification] = []

    hamil: Mapping[str, float] = backend.configuration().hamiltonian["vars"]
    control_channels = backend.configuration().control_channels

    coupling_map: dict[tuple[int, int], float] = {}
    for k, v in hamil.items():
        if not k.startswith("j"):
            continue
        i0, i1 = map(int, k[2:].split("q", maxsplit=2))
        coupling_map[(i0, i1)] = v / freq_mult

    for qubit in qubits:
        freq = hamil[f"wq{qubit}"] / freq_mult
        rabi = hamil[f"omegad{qubit}"] / freq_mult
        delta = hamil[f"delta{qubit}"] / freq_mult
        lcm: dict[int, float] = {}
        lcc: dict[int, int] = {}
        for k in coupling_map.keys():
            if qubit not in k:
                continue
            ind = k[1] if qubit == k[0] else k[0]
            lcm[ind] = coupling_map[k]
            # lcc[ind] = control_channels[(qubit, ind)][0].index
            channel = control_channels.get(k, control_channels.get(k[-1::-1], None))
            assert channel is not None
            lcc[ind] = channel[0].index
        spec = QubitSpecification(
            index=qubit,
            freq=freq,
            delta=delta,
            rabi=rabi,
            coupling_map=lcm,
            control_channels=lcc,
        )
        ret.append(spec)

    return ret, dt


# %%
SIGNAL_MAKER_TYPE: TypeAlias = Callable[
    [qiskit.pulse.Schedule | qiskit.pulse.ScheduleBlock],
    Sequence[qd.signals.signals.Signal],
]
"""Base protocol for a schedule to signal transformer function"""


def get_frequencies(
    qubit_specifications: Sequence[QubitSpecification],
    real_to_sim_map: Mapping[int, int],
    control_map: Mapping[tuple[int, int], int],
) -> Mapping[str, float]:
    """
    Get the frequencies of each channel from the qubit specification

    Return the relevant channel frequencies from the given qubit specification, real to simulation mapping and control channel mapping

    Parameters
    ----------
    qubit_specifications
        The qubits to consider
    real_to_sim_map
        A mapping of the "real" to simulation qubit indices
    control_map
        The control channel mapping

    Returns
    -------
    Mapping[str, float]
        The mapping of each relevant channel
    """
    car_freq = {f"d{qubit.index}": qubit.freq for qubit in qubit_specifications}
    control_channels: dict[str, str] = {}
    for qubit in qubit_specifications:
        if qubit.control_channels is None:
            continue
        for other, con in qubit.control_channels.items():
            if other not in real_to_sim_map.keys():
                continue
            qs = real_to_sim_map[qubit.index]
            os = real_to_sim_map[other]
            other_spec = qubit_specifications[os]
            control_channels[f"u{con}"] = f"u{control_map[(qs,os)]}"
            car_freq[f"u{con}"] = 0 * qubit.freq - (-other_spec.freq)
            car_freq[f"u{con}"] = DEFAULT_FREQ_OF_CONTROL(qubit.freq, other_spec.freq)
    return car_freq


def _convert_schedule_to_signal_maker(
    qubit_specifications: Sequence[QubitSpecification],
    real_to_sim_map: Mapping[int, int],
    control_map: Mapping[tuple[int, int], int],
    dt: float,
    cross_talk: bool,
) -> SIGNAL_MAKER_TYPE:
    """
    Generates a converter from `Schedule` or `ScheduleBlock` to a `Signal` object with the correct channel naming.

    This is an internal function and is not meant to be used outside the module!
    It uses the qubit specifications, qubit labels mapping, and the control channel naming scheme to convert `Schedule`(`ScheduleBlock`) to `Signal`.

    Parameters
    ----------
    qubit_specifications : Sequence[QubitSpecification]
        The qubit specifications of the device.
    real_to_sim_map : Mapping[int, int]
        The mapping from the real label to the simulation ones.
    control_map : Mapping[tuple[int, int], int]
        Mapping from the `new` qubit labels pair to a control channel.
    dt : float
        The sampling rate of the device.

    Returns
    -------
    SIGNAL_MAKER_TYPE
        A function that does the converting.
    """
    drive_channels = {f"d{r}": f"d{s}" for r, s in real_to_sim_map.items()}
    car_freq = {f"d{i}": qubit_specifications[i].freq for i in real_to_sim_map.values()}
    car_freq = {f"d{qubit.index}": qubit.freq for qubit in qubit_specifications}
    control_channels: dict[str, str] = {}
    if cross_talk:
        for qubit in qubit_specifications:
            if qubit.control_channels is None:
                continue
            for other, con in qubit.control_channels.items():
                if other not in real_to_sim_map.keys():
                    continue
                qs = real_to_sim_map[qubit.index]
                os = real_to_sim_map[other]
                other_spec = qubit_specifications[os]
                control_channels[f"u{con}"] = f"u{control_map[(qs,os)]}"
                car_freq[f"u{con}"] = 0 * qubit.freq - (-other_spec.freq)
        drive_channels.update(control_channels)

    converter = qd.pulse.InstructionToSignals(
        dt,
        carriers=car_freq,
        channels=list(drive_channels.keys()),
    )

    def _internal(
        schedule: qiskit.pulse.ScheduleBlock | qiskit.pulse.Schedule,
    ) -> Sequence[qd.signals.signals.Signal]:
        if isinstance(schedule, qiskit.pulse.ScheduleBlock):
            schedule = qiskit.pulse.transforms.block_to_schedule(schedule)
        signals = converter.get_signals(schedule)
        for signal in signals:
            signal._name = drive_channels[signal._name]  # TODO remove ugly fix!
        return signals

    return _internal


# TODO Continue from here
@dataclasses.dataclass(frozen=True)
class QubitNoiseParameters:
    """Dataclass of a qubit's noise properties"""

    index: int
    """The qubit's index"""
    T1: Optional[float]
    """T1 - Time for depolarization process"""
    T2: Optional[float]
    """T2 - Time for dephasing process"""
    # TODO consider more noise types ?

    @property
    def gamma1(self) -> float | None:
        """
        Get the coefficient for depolarization process in the Lindbladian, $\gamma_1$.

        If T1 is not None, return the coefficient for the depolarization process for the Lindbladian, $\gamma_1=1/T_1$.

        Returns
        -------
        float | None
            If T1 is not None, return $\gamma_1$
        """
        if self.T1 is None:
            return None
        return 1 / self.T1

    @property
    def gamma2(self) -> float | None:
        """
        Get the coefficient for dephasing process in the Lindbladian, $\gamma_2$.

        If T2 is not None, return the coefficient for the dephasing process for the Lindbladian, $\gamma_2=1/T_2$.

        Returns
        -------
        float | None
            If T2 is not None, return $\gamma_2$
        """
        if self.T2 is None:
            return None
        return 1 / self.T2


def get_qubit_noise_parameters(
    backend: NoiseParameterBackend,
    qubits: Sequence[int] | int,
    freq_units: str = "GHz",
    time_units: str = "ns",
    div_by_two_pi: bool = False,
) -> Sequence[QubitNoiseParameters]:
    """
    Extracts specific qubit noise parameters, namely, T1 and T2 times, from the backend.

    Extracts specific qubit noise parameters from the backend and stores them in a `QubitNoiseParameters` object.
    For each qubit, extract (1) the depolarization time, $T_1$; and (2) the dephasing time, $T_2$. Each missing property is given the value of `None`.

    Parameters
    ----------
    backend: NoiseParameterBackend
        The backend to get the information from.
    qubits: Sequence[int] | int
        What qubit(s) ti get data for.
    freq_units: str
        Frequency units. Defaults to "GHz".
    time_units: str
        Time units. Defaults to "ns".
    div_by_two_pi: bool
        Divide units by 2π? Defaults to True.

    Returns
    -------
    Sequence[QubitNoiseParameters]
        Returns a list of `QubitNoiseParameters`.
    """
    assert freq_units.lower().endswith(
        "hz"
    ), f"I do not know these frequency units! {freq_units}"
    assert time_units.lower().endswith(
        "s"
    ), f"I do not know these time units! {time_units}"

    freq_mult = 10 ** get_si_mult(freq_units[:-2])

    freq_mult *= (2 * np.pi) if div_by_two_pi else 1.0

    time_mult = 10 ** get_si_mult(time_units[:-1])

    if not isinstance(qubits, Iterable):
        qubits = [qubits]

    ret: list[QubitNoiseParameters] = []

    properties = backend.properties()
    assert properties is not None, f"This backend has no properties!"

    for qubit in qubits:
        qp = properties.qubit_property(qubit)
        T1 = qp.get("T1", None)
        T2 = qp.get("T2", None)
        if T1 is not None:
            T1 = T1[0]
            T1 /= time_mult
        if T2 is not None:
            T2 = T2[0]
            T2 /= time_mult
        qubit_noise = QubitNoiseParameters(
            index=qubit,
            T1=T1,
            T2=T2,
        )
        ret.append(qubit_noise)

    return ret


METHODS: TypeAlias = Literal["qubit","transmon","transmon-dicke"]
"""Currently supported simulation methods."""


# TODO Add overload typehints
def generate_solver(
    qubit_specification: Sequence[QubitSpecification],
    dt: float,
    cross_talk: bool = False,
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
    method: METHODS = "qubit",
    **kwargs,
) -> tuple[
    qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE
]:
    """
    Generate a solver for the given system parameters.

    For (1) a given qubit properties and topology, specified in `qubit_specification`; (2) noise parameters given in `qubit_noise_model`; (3) `dt` sampling time; (4) `cross_talk` control whether the qubits are connected; and (5) what `method` to use to generate the solver.

    Currently support the following methods:
        - 'qubit' - Simulate each qubit as a qubit;
        - 'transmon' - Simulate each qubit as a Tranmon;
        - 'transmon-dicke' - Simulate each qubit as a Transmon with Dicke model like interactions.

    Each method may use some additional parameters
     - method == 'qubit' - No additional parameters;
     - method == 'transmon' | 'transmon_dicke' - 'transmon_dim' the dimensionality of each Transmon. Defaults to 5.

    Parameters
    ----------
    qubit_specification: Sequence[QubitSpecification]
        The qubits specifications.
    dt: float
        The sampling rate.
    cross_talk: bool
        Connect the qubits
    qubit_noise_model: Sequence[QubitNoiseParameters]|None
        The qubits' noise parameters, can be None to simulate without noise.
    method: METHODS
        The simulation method. Defaults to 'qubit'

    Returns
    -------
    tuple[
     qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE
    ]
        Returns a tuple of: (1) solver, (2) mapping from real index to simulation index, (3) mapping from a pair of qubit indices to the appropriate control channel index, and (4) a function to convert a `Schedule`|`ScheduleBlock` to list[Signal].

    Raises
    ------
    NotImplementedError:
        If the method is not supported.
    """
    if method == "qubit":
        return generate_solver_qubit(
            qubit_specification, dt, cross_talk, qubit_noise_model
        )
    elif method == "transmon":
        transmon_dim = kwargs.get("transmon_dim", 5)
        return generate_solver_transmon(
            qubit_specification, dt, cross_talk, qubit_noise_model, transmon_dim
        )
    elif method == "transmon-dicke":
        transmon_dim = kwargs.get("transmon_dim", 5)
        return generate_solver_transmon_dicke(
            qubit_specification, dt, cross_talk, qubit_noise_model, transmon_dim
        )
    else:
        raise NotImplementedError(f"{method=} is unknown to me!")


def compose_space(
    N: int,
    dim: int,
    place_operators: Optional[Mapping[int, npt.NDArray[np.complex_]]] = None,
    I: Optional[npt.NDArray[np.complex_]] = None,
) -> npt.NDArray[np.complex_]:
    """
    Compose various single-qubit operators to a multi-qubit operator.

    Compose a given set of single-qubit operators to a multi-qubit operator.

    Parameters
    ----------
    N: int
        The total number of qubits in the system.
    dim: int
        The dimensionality of each qubit in the system.
    place_operators: Mapping[int, npt.NDArray[np.complex_]] | None
        A mapping of qubit index to operator, can be `None` to get the identity operator.
    I: npt.NDArray[np.complex_] | None
        The identity operator for a single-qubit system, can be `None` to generate from the `dim` parameters.

    Returns
    -------
    npt.NDArray[np.complex_]
        The multi-qubit operator.
    """
    if I is None:
        I = np.eye(dim).astype(complex)
    ret: npt.NDArray[np.complex_] = np.array(1, dtype=complex)

    for i in range(N):
        o = (
            place_operators[i]
            if place_operators is not None and i in place_operators
            else I
        )
        ret = np.kron(ret, o)

    return ret


def generate_solver_transmon(
    qubit_specifications: Sequence[QubitSpecification],
    dt: float,
    cross_talk: bool = False,
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
    transmon_dim: int = 5,
) -> tuple[
    qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE
]:
    """
    Generates the Solver object from the given `QubitSpecification` and `QubitNoiseParameters` lists.

    Generates the Hamiltonian from the given `QubitSpecification` list.
    The Hamiltonian is build from the following blocks
     - ```H_{static} = 2πωᵢNᵢ-παᵢNᵢ(Nᵢ-1)```
     - ```H_{drive} = 2πΩᵢ(aᵢ⁻+aᵢ⁺)dᵢ(t)```
     - ```H_{CR} = 2πJᵢⱼ(aᵢ⁺aⱼ⁻+h.c.)```
     - ```H_{control} = 2πΩᵢ(aᵢ⁻+aᵢ⁺)U⁽ⁱʲ⁾ₖ(t)```
     Where all the definitions are in the (Qiskit tutorial)[LINK].

    Parameters
    ----------
    qubit_specifications : Sequence[QubitSpecification]
        The qubits to generate.
    dt : float
        The time sampling of the device.
    cross_talk : bool
        Include ```H_{CR}``` and ```H_{control}``` in the Hamiltonian.

    Returns
    -------
    tuple[qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE]
        - The solver
        - Mapping from the real qubit index to the simulation index.
        - Mapping from the simulation indices to the control index.
        - A function to convert from `Schedule` or `ScheduleBlock` to `Signal` with correct signal names.
    """
    p = np.array([[0, 1], [0, 0]], dtype=complex)
    m = p.conj().T

    drift_hamil: list[npt.NDArray[np.complex_]] = []
    operators: list[npt.NDArray[np.complex_]] = []
    cross_operators: list[npt.NDArray[np.complex_]] = []
    cross_map: dict[tuple[int, int], int] = {}
    cross_channels: list[str] = []
    cross_freqs: dict[str, float] = {}

    Nq = len(qubit_specifications)
    rwa_freq: list[float] = []
    real_to_sim_map: dict[int, int] = {
        spec.index: i for i, spec in enumerate(qubit_specifications)
    }

    I = np.eye(transmon_dim, dtype=complex)
    nda = np.diag(np.sqrt(np.arange(1, transmon_dim)), 1).astype(complex)
    yda = np.diag(np.sqrt(np.arange(1, transmon_dim)), -1).astype(complex)
    N = np.diag(np.arange(transmon_dim), 0).astype(complex)
    II = compose_space(Nq, transmon_dim, I=I)

    for i, spec in enumerate(qubit_specifications):
        _, freq, alpha, rabi, mapping = spec
        rwa_freq.append(freq)
        NN = compose_space(Nq, transmon_dim, {i: N}, I=I)
        NDA = compose_space(Nq, transmon_dim, {i: nda}, I=I)
        YDA = compose_space(Nq, transmon_dim, {i: yda}, I=I)
        X = qiskit.quantum_info.Operator.from_label(f"{'I'*i}X{'I'*(Nq-i-1)}")
        Z = qiskit.quantum_info.Operator.from_label(f"{'I'*i}Z{'I'*(Nq-i-1)}")
        ld = 2 * np.pi * (freq * NN - NN * (NN - II) * alpha / 2)
        lo = 2 * np.pi * rabi * (NDA + YDA)
        if cross_talk:
            for oq in sorted(mapping.keys()):
                loc = list(filter(lambda q: q.index == oq, qubit_specifications))
                if len(loc) == 0:
                    continue
                loc = loc[0]

                oper: list[npt.NDArray[np.complex_]] = []
                for t, o in ((yda, nda), (nda, yda)):
                    j = real_to_sim_map[loc.index]
                    b = compose_space(Nq, transmon_dim, {i: t, j: o}, I=I)
                    # b = qiskit.quantum_info.Operator.from_label("I" * Nq)
                    # b = b.compose(t, qargs=[i])
                    # b = b.compose(o, qargs=[real_to_sim_map[loc.index]])
                    oper.append(b)
                # lo += 2 * np.pi * mapping[oq] * sum(oper)
                cur_ind = len(cross_channels)
                ch = f"u{cur_ind}"
                cross_channels.append(ch)
                cross_freqs[ch] = -(freq - loc.freq)  # TODO verify the order !!
                cross_map[(i, real_to_sim_map[loc.index])] = cur_ind
                cross_operators.append(2 * np.pi * mapping[oq] * sum(oper))

        drift_hamil.append(ld)
        operators.append(lo)

    static_hamil: npt.NDArray[np.complex_] = sum(drift_hamil)
    # operators_sum: qiskit.quantum_info.Operator = sum(operators)
    carrier_freqs = {f"d{i}": rwa_freq[i] for i in range(Nq)}
    carrier_freqs.update(cross_freqs)

    static_dissipators: list[npt.NDArray[np.complex_]] | None = None
    if qubit_noise_model is not None:
        assert len(qubit_noise_model) == len(
            qubit_specifications
        ), f"Why give me weird data? {qubit_specifications=}, {qubit_noise_model=}"
        static_dissipators = []
        b = qiskit.quantum_info.Operator.from_label("I" * Nq)
        for q_noise in qubit_noise_model:
            # TODO verify that noise is actually added to the system.
            # current test are not positive...
            assert (
                q_noise.index in real_to_sim_map
            ), f"Why send noise for qubit {q_noise.index}?!?"
            if q_noise.T1 is not None:
                gamma1 = q_noise.gamma1
                # lb = b.copy()
                # lb = lb.compose(
                #     p,
                #     qargs=[
                #         real_to_sim_map[q_noise.index],
                #     ],
                # )
                lb = compose_space(
                    Nq, transmon_dim, {real_to_sim_map[q_noise.index]: nda}, I=I
                )
                static_dissipators.append(np.sqrt(gamma1) * lb)
            if q_noise.T2 is not None:
                gamma2 = q_noise.gamma2
                # lb = b.copy()
                # lb = lb.compose(
                #     qiskit.quantum_info.Operator.from_label("Z"),
                #     qargs=[
                #         real_to_sim_map[q_noise.index],
                #     ],
                # )
                lb = compose_space(
                    Nq, transmon_dim, {real_to_sim_map[q_noise.index]: N}, I=I
                )
                static_dissipators.append(np.sqrt(gamma2) * lb)

    return (
        qd.Solver(
            static_hamiltonian=static_hamil,
            rotating_frame=np.diag(static_hamil),
            hamiltonian_operators=[*operators, *cross_operators],
            rwa_cutoff_freq=2 * max(rwa_freq),
            hamiltonian_channels=[*[f"d{i}" for i in range(Nq)], *cross_channels],
            channel_carrier_freqs=carrier_freqs,
            dt=dt,
            static_dissipators=static_dissipators,
        ),
        real_to_sim_map,
        cross_map,
        _convert_schedule_to_signal_maker(
            qubit_specifications, real_to_sim_map, cross_map, dt, cross_talk
        ),
    )


# TODO verify this is needed...
def generate_solver_transmon_dicke(
    qubit_specifications: Sequence[QubitSpecification],
    dt: float,
    cross_talk: bool = False,
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
    transmon_dim: int = 5,
) -> tuple[
    qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE
]:
    """
    Generates the Solver object from the given `QubitSpecification` and `QubitNoiseParameters` lists.

    Generates the Hamiltonian from the given `QubitSpecification` list.
    The Hamiltonian is build from the following blocks
     - ```H_{static} = 2πωᵢNᵢ-παᵢNᵢ(Nᵢ-1)```
     - ```H_{drive} = 2πΩᵢ(aᵢ⁻+aᵢ⁺)dᵢ(t)```
     - ```H_{CR} = 2πJᵢⱼ(aᵢ⁺aⱼ⁻+h.c.)```
     - ```H_{control} = 2πΩᵢ(aᵢ⁻+aᵢ⁺)U⁽ⁱʲ⁾ₖ(t)```
     Where all the definitions are in the (Qiskit tutorial)[LINK].

    Parameters
    ----------
    qubit_specifications : Sequence[QubitSpecification]
        The qubits to generate.
    dt : float
        The time sampling of the device.
    cross_talk : bool
        Include ```H_{CR}``` and ```H_{control}``` in the Hamiltonian.

    Returns
    -------
    tuple[qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE]
        - The solver
        - Mapping from the real qubit index to the simulation index.
        - Mapping from the simulation indices to the control index.
        - A function to convert from `Schedule` or `ScheduleBlock` to `Signal` with correct signal names.
    """
    p = np.array([[0, 1], [0, 0]], dtype=complex)
    m = p.conj().T

    drift_hamil: list[npt.NDArray[np.complex_]] = []
    operators: list[npt.NDArray[np.complex_]] = []
    cross_operators: list[npt.NDArray[np.complex_]] = []
    cross_map: dict[tuple[int, int], int] = {}
    cross_channels: list[str] = []
    cross_freqs: dict[str, float] = {}

    Nq = len(qubit_specifications)
    rwa_freq: list[float] = []
    real_to_sim_map: dict[int, int] = {
        spec.index: i for i, spec in enumerate(qubit_specifications)
    }

    I = np.eye(transmon_dim, dtype=complex)
    nda = np.diag(np.sqrt(np.arange(1, transmon_dim)), 1).astype(complex)
    yda = np.diag(np.sqrt(np.arange(1, transmon_dim)), -1).astype(complex)
    N = np.diag(np.arange(transmon_dim), 0).astype(complex)
    II = compose_space(Nq, transmon_dim, I=I)

    for i, spec in enumerate(qubit_specifications):
        _, freq, alpha, rabi, mapping = spec
        rwa_freq.append(freq)
        NN = compose_space(Nq, transmon_dim, {i: N}, I=I)
        NDA = compose_space(Nq, transmon_dim, {i: nda}, I=I)
        YDA = compose_space(Nq, transmon_dim, {i: yda}, I=I)
        X = qiskit.quantum_info.Operator.from_label(f"{'I'*i}X{'I'*(Nq-i-1)}")
        Z = qiskit.quantum_info.Operator.from_label(f"{'I'*i}Z{'I'*(Nq-i-1)}")
        ld = 2 * np.pi * (freq * NN + NN * (NN - II) * alpha / 2)
        lo = 2 * np.pi * rabi * (NDA + YDA)
        if cross_talk:
            for oq in sorted(mapping.keys()):
                loc = list(filter(lambda q: q.index == oq, qubit_specifications))
                if len(loc) == 0:
                    continue
                loc = loc[0]

                oper: list[npt.NDArray[np.complex_]] = []
                for t, o in ((yda, nda), (nda, yda)):
                    j = real_to_sim_map[loc.index]
                    b = compose_space(Nq, transmon_dim, {i: t, j: o}, I=I)
                    # b = qiskit.quantum_info.Operator.from_label("I" * Nq)
                    # b = b.compose(t, qargs=[i])
                    # b = b.compose(o, qargs=[real_to_sim_map[loc.index]])
                    oper.append(b)
                # lo += 2 * np.pi * mapping[oq] * sum(oper)
                cur_ind = len(cross_channels)
                ch = f"u{cur_ind}"
                cross_channels.append(ch)
                cross_freqs[ch] = -(freq - loc.freq)  # TODO verify the order !!
                cross_freqs[ch] = loc.freq  # TODO verify the order !!
                cross_map[(i, real_to_sim_map[loc.index])] = cur_ind
                drift_hamil.append(
                    2 * np.pi * mapping[oq] * sum(oper) / 2
                )  # / 2 due to double counting
                # cross_operators.append(2 * np.pi * mapping[oq] * sum(oper))
                cross_operators.append(lo)
                # cross_operators.append(loc.rabi * lo / rabi)

        drift_hamil.append(ld)
        operators.append(lo)

    static_hamil: npt.NDArray[np.complex_] = sum(drift_hamil)
    # operators_sum: qiskit.quantum_info.Operator = sum(operators)
    carrier_freqs = {f"d{i}": rwa_freq[i] for i in range(Nq)}
    carrier_freqs.update(cross_freqs)

    static_dissipators: list[npt.NDArray[np.complex_]] | None = None
    if qubit_noise_model is not None:
        assert len(qubit_noise_model) == len(
            qubit_specifications
        ), f"Why give me weird data? {qubit_specifications=}, {qubit_noise_model=}"
        static_dissipators = []
        b = qiskit.quantum_info.Operator.from_label("I" * Nq)
        for q_noise in qubit_noise_model:
            # TODO verify that noise is actually added to the system.
            # current test are not positive...
            assert (
                q_noise.index in real_to_sim_map
            ), f"Why send noise for qubit {q_noise.index}?!?"
            if q_noise.T1 is not None:
                gamma1 = q_noise.gamma1
                # lb = b.copy()
                # lb = lb.compose(
                #     p,
                #     qargs=[
                #         real_to_sim_map[q_noise.index],
                #     ],
                # )
                lb = compose_space(
                    Nq, transmon_dim, {real_to_sim_map[q_noise.index]: nda}, I=I
                )
                static_dissipators.append(np.sqrt(gamma1) * lb)
            if q_noise.T2 is not None:
                gamma2 = q_noise.gamma2
                # lb = b.copy()
                # lb = lb.compose(
                #     qiskit.quantum_info.Operator.from_label("Z"),
                #     qargs=[
                #         real_to_sim_map[q_noise.index],
                #     ],
                # )
                lb = compose_space(
                    Nq, transmon_dim, {real_to_sim_map[q_noise.index]: N}, I=I
                )
                static_dissipators.append(np.sqrt(gamma2) * lb)

    return (
        qd.Solver(
            static_hamiltonian=static_hamil,
            rotating_frame=np.diag(static_hamil),
            hamiltonian_operators=[*operators, *cross_operators],
            rwa_cutoff_freq=2 * max(rwa_freq),
            hamiltonian_channels=[*[f"d{i}" for i in range(Nq)], *cross_channels],
            channel_carrier_freqs=carrier_freqs,
            dt=dt,
            static_dissipators=static_dissipators,
        ),
        real_to_sim_map,
        cross_map,
        _convert_schedule_to_signal_maker(
            qubit_specifications, real_to_sim_map, cross_map, dt, cross_talk
        ),
    )


def generate_solver_qubit(
    qubit_specifications: Sequence[QubitSpecification],
    dt: float,
    cross_talk: bool = False,
    qubit_noise_model: Optional[Sequence[QubitNoiseParameters]] = None,
) -> tuple[
    qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE
]:
    """
    Generates the Solver object from the given `QubitSpecification` and `QubitNoiseParameters` lists.

    Generates the Hamiltonian from the given `QubitSpecification` list.
    The Hamiltonian is build from the following blocks
     - ```H_{static} = 2πωᵢσᵢ³/2```
     - ```H_{drive} = 2πΩᵢσᵢ¹dᵢ(t)```
     - ```H_{CR} = 2πJᵢⱼ(σᵢ⁺σⱼ⁻+h.c.)```
     - ```H_{control} = 2πΩᵢσᵢ¹U⁽ⁱʲ⁾ₖ(t)```
     Where all the definitions are in the (Qiskit tutorial)[LINK].

    Parameters
    ----------
    qubit_specifications : Sequence[QubitSpecification]
        The qubits to generate.
    dt : float
        The time sampling of the device.
    cross_talk : bool
        Include ```H_{CR}``` and ```H_{control}``` in the Hamiltonian.

    Returns
    -------
    tuple[qd.Solver, Mapping[int, int], Mapping[tuple[int, int], int], SIGNAL_MAKER_TYPE]
        - The solver
        - Mapping from the real qubit index to the simulation index.
        - Mapping from the simulation indices to the control index.
        - A function to convert from `Schedule` or `ScheduleBlock` to `Signal` with correct signal names.
    """
    p = np.array([[0, 1], [0, 0]], dtype=complex)
    m = p.conj().T

    drift_hamil: list[qiskit.quantum_info.Operator] = []
    operators: list[qiskit.quantum_info.Operator] = []
    cross_operators: list[qiskit.quantum_info.Operator] = []
    cross_map: dict[tuple[int, int], int] = {}
    cross_channels: list[str] = []
    cross_freqs: dict[str, float] = {}

    Nq = len(qubit_specifications)
    rwa_freq: list[float] = []
    real_to_sim_map: dict[int, int] = {
        spec.index: i for i, spec in enumerate(qubit_specifications)
    }

    for i, spec in enumerate(qubit_specifications):
        _, freq, _, rabi, mapping = spec
        rwa_freq.append(freq)
        X = qiskit.quantum_info.Operator.from_label(f"{'I'*i}X{'I'*(Nq-i-1)}")
        Z = qiskit.quantum_info.Operator.from_label(f"{'I'*i}Z{'I'*(Nq-i-1)}")
        ld = 2 * np.pi * freq * Z / 2
        lo = 2 * np.pi * rabi * X
        if cross_talk:
            for oq in sorted(mapping.keys()):
                loc = list(filter(lambda q: q.index == oq, qubit_specifications))
                if len(loc) == 0:
                    continue
                loc = loc[0]

                oper: Sequence[qiskit.quantum_info.Operator] = []
                for t, o in ((p, m), (m, p)):
                    b = qiskit.quantum_info.Operator.from_label("I" * Nq)
                    b = b.compose(t, qargs=[i])
                    b = b.compose(o, qargs=[real_to_sim_map[loc.index]])
                    oper.append(b)
                lo += 2 * np.pi * mapping[oq] * sum(oper)
                cur_ind = len(cross_channels)
                ch = f"u{cur_ind}"
                cross_channels.append(ch)
                cross_freqs[ch] = freq - loc.freq  # TODO verify the order !!
                cross_map[(i, real_to_sim_map[loc.index])] = cur_ind
                cross_operators.append(2 * np.pi * rabi * X)

        drift_hamil.append(ld)
        operators.append(lo)

    static_hamil: qiskit.quantum_info.Operator = sum(drift_hamil)
    # operators_sum: qiskit.quantum_info.Operator = sum(operators)
    carrier_freqs = {f"d{i}": rwa_freq[i] for i in range(Nq)}
    carrier_freqs.update(cross_freqs)

    static_dissipators: list[qiskit.quantum_info.Operator] | None = None
    if qubit_noise_model is not None:
        assert len(qubit_noise_model) == len(
            qubit_specifications
        ), f"Why give me weird data? {qubit_specifications=}, {qubit_noise_model=}"
        static_dissipators = []
        b = qiskit.quantum_info.Operator.from_label("I" * Nq)
        for q_noise in qubit_noise_model:
            # TODO verify that noise is actually added to the system.
            # current test are not positive...
            assert (
                q_noise.index in real_to_sim_map
            ), f"Why send noise for qubit {q_noise.index}?!?"
            if q_noise.T1 is not None:
                gamma1 = q_noise.gamma1
                lb = b.copy()
                lb = lb.compose(
                    p,
                    qargs=[
                        real_to_sim_map[q_noise.index],
                    ],
                )
                static_dissipators.append(np.sqrt(gamma1) * lb)
            if q_noise.T2 is not None:
                gamma2 = q_noise.gamma2
                lb = b.copy()
                lb = lb.compose(
                    qiskit.quantum_info.Operator.from_label("Z"),
                    qargs=[
                        real_to_sim_map[q_noise.index],
                    ],
                )
                static_dissipators.append(np.sqrt(gamma2) * lb)

    return (
        qd.Solver(
            static_hamiltonian=static_hamil,
            rotating_frame=np.diag(static_hamil),
            hamiltonian_operators=[*operators, *cross_operators],
            rwa_cutoff_freq=2 * max(rwa_freq),
            hamiltonian_channels=[*[f"d{i}" for i in range(Nq)], *cross_channels],
            channel_carrier_freqs=carrier_freqs,
            dt=dt,
            static_dissipators=static_dissipators,
        ),
        real_to_sim_map,
        cross_map,
        _convert_schedule_to_signal_maker(
            qubit_specifications, real_to_sim_map, cross_map, dt, cross_talk
        ),
    )


# %%
def pad_schedule(
    schedule: qiskit.pulse.Schedule | qiskit.pulse.ScheduleBlock,
    channels: Sequence[str],
) -> qiskit.pulse.Schedule | qiskit.pulse.ScheduleBlock:
    """
    Applies a command on each of the given channels.

    This function is to fix a bug in Qiskit-Dynamics where the "signal" being simulated must have all the channels active.

    Parameters
    ----------
    schedule : qiskit.pulse.Schedule | qiskit.pulse.ScheduleBlock
        The `Schedule`|`ScheduleBlock` to "pad".
    channels : Sequence[str]
        What channels to "activate".

    Returns
    -------
    qiskit.pulse.Schedule | qiskit.pulse.ScheduleBlock
        The "padded" `Schedule`|`ScheduleBlock`

    Raises
    ------
    RuntimeWarning:
        If the channel is not a DriveChannel or a ControlChannel.
    """
    ch: Sequence[qiskit.pulse.channels.Channel] = []
    for c in channels:
        match c[0]:
            case "d":
                cc = qiskit.pulse.DriveChannel(int(c[1:]))
            case "u":
                cc = qiskit.pulse.ControlChannel(int(c[1:]))
            case _:
                raise RuntimeWarning(f"Got a strange channel {c}!")
        ch.append(cc)
    with qiskit.pulse.build() as ret_schd:
        for c in ch:
            qiskit.pulse.delay(0, c)
        qiskit.pulse.call(schedule)

    return ret_schd


# %%
def make_simple_H(
    i: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0
) -> qiskit.quantum_info.Operator:
    """
    Helper function to build single qubit operators.

    Returns the following single qubit operator
    ```Ô = i∙I + x∙σ¹ + y∙σ² + z∙σ³```

    Parameters
    ----------
    i : float
        The I coefficient.
    x : float
        The X coefficient.
    y : float
        The Y coefficient.
    z : float
        The Z coefficient.

    Returns
    -------
    qiskit.quantum_info.Operator
        The operator.
    """
    Im = qiskit.quantum_info.Operator.from_label("I")
    Xm = qiskit.quantum_info.Operator.from_label("X")
    Ym = qiskit.quantum_info.Operator.from_label("Y")
    Zm = qiskit.quantum_info.Operator.from_label("Z")

    return i * Im + x * Xm + y * Ym + z * Zm
