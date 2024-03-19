"""Module for JAX utilities"""
from typing import Callable, NamedTuple, Self

from .simulations.utils import ChannelProperties
from .utils import TimingConstraints, STANDARD_TIMING, PaddingType
from jax import numpy as jnp
from numpy import typing as npt
from enum import auto, Enum


class JAXPaddingType(Enum):
    """JAX enabled padding type enum"""
    NO = auto()
    """No padding"""
    LEFT = auto()
    """Left padding"""
    RIGHT = auto()
    """Right padding"""
    MIDDLE = auto()
    """Middle padding"""

    def pad(
        self: Self,
        values,
        timing_const: TimingConstraints = STANDARD_TIMING,
    ):
        """
        Pads a JAX array.

        Pads a JAX array to have a specific padding with timing constraints.

        Parameters
        ----------
        values : JAX array
            The original array to be padded
        timing_const
            The timing constraints to be met
        """
        if self == self.NO:
            return jnp.asarray(values, dtype=complex)
        cur_len = len(values)
        new_len = timing_const.get_length(cur_len)
        new_len = timing_const.fix_min(new_len)
        ret = jnp.zeros(shape=(new_len,), dtype=complex)
        match self:
            case self.LEFT:
                ret = ret.at[:cur_len].set(values)
            case self.MIDDLE:
                s = new_len // 2 - cur_len // 2
                ret = ret.at[s : s + cur_len].set(values)
            case self.RIGHT:
                ret = ret.at[-cur_len:].set(values)
        return ret


def jaxify(padding_type: PaddingType) -> JAXPaddingType:
    """
    Convert "regular" padding to JAX padding

    Convert "regular" padding to JAX padding that can use JAX JIT

    Parameters
    ----------
    padding_type
        The original padding type

    Returns
    -------
    JAXPaddingType
        The JAX padding type
    """
    match padding_type:
        case PaddingType.NO:
            return JAXPaddingType.NO
        case PaddingType.LEFT:
            return JAXPaddingType.LEFT
        case PaddingType.RIGHT:
            return JAXPaddingType.RIGHT
        case PaddingType.MIDDLE:
            return JAXPaddingType.MIDDLE


class JAXChannelProperties(NamedTuple):
    """JAX channel properties"""
    phys_to_logical: int
    """How long to stretch each time window for the given channel type"""
    padding: JAXPaddingType
    """The padding of the given channel type"""


def convert_to_jax_properties(channel_props: ChannelProperties) -> JAXChannelProperties:
    """
    Convert channel properties to JAX format

    Convert the "regular" channel properties to JAX version

    Parameters
    ----------
    channel_props
        The original channel properties

    Returns
    -------
    JAXChannelProperties
        JAX format of `channel_props`
    """
    return JAXChannelProperties(
        channel_props.phys_to_logical, jaxify(channel_props.padding)
    )


class CanDoExpectationValue:
    _data: jnp.ndarray

    def __init__(self: Self, data: jnp.ndarray) -> None:
        """
        Exposes `expectation_value` for JAX representation of a quantum state

        Exposes `expectation_value` for JAX representation of a quantum state (JAX array).

        Parameters
        ----------
        data
            JAX NDArray of the quantum state

        Returns
        -------
        None
        ⟩
        """
        self._data = data

    def expectation_value(self: Self, operator: npt.NDArray) -> jnp.ndarray:
        """
        Expectation value of a quantum state with respect to some operator

        Returns ⟨O⟩ with respect to the given state.
        If ψ => ⟨ψ|O|ψ⟩; Else => Tr[ρ O]

        Parameters
        ----------
        operator
            The operator to measure

        Returns
        -------
        jnp.ndarray
            The expectation value

        Raises
        ------
        NotImplementedError:
            If the state is not one of state vector, or density matrix
        """
        data = self._data
        shape = data.shape
        jo = jnp.array(operator)
        match len(shape):
            case 2:
                return jnp.trace(data @ jo)
            case 1:
                return jnp.conjugate(data.T) @ jo @ data
            case _:
                raise NotImplementedError(f"What is this? {shape=}")


def project01(
    n_qubits: int, transmon_dim: int
) -> Callable[[jnp.ndarray], CanDoExpectationValue]:
    """
    Convert a Transmon quantum state to a two-level quantum state

    Reduces the Hilbert space from a multi Transmon system to multiple qubits

    Parameters
    ----------
    n_qubits
        Total number of qubits
    transmon_dim
        The Transmon dimension

    Returns
    -------
    Callable[[jnp.ndarray], CanDoExpectationValue]
        A function to convert the Transmon system to a qubit system

    Raises
    ------
    NotImplementedError:
        In case of a weird quantum state fail
    """
    proj_01 = jnp.array(
        [
            jnp.concatenate((jnp.array([1]), jnp.repeat(0, transmon_dim - 1))),
            jnp.concatenate((jnp.array([0, 1]), jnp.repeat(0, transmon_dim - 2))),
        ]
    ).astype(complex)
    P01 = jnp.array([1])
    for _ in range(n_qubits):
        P01 = jnp.kron(P01, proj_01)

    def _internal(data: jnp.ndarray) -> CanDoExpectationValue:
        shape = data.shape
        if len(shape) == 2:
            data = P01 @ data @ P01.T
            data /= data.trace()
            return CanDoExpectationValue(
                data,
            )
        elif len(shape) == 1:
            data = data @ P01.T
            data /= jnp.sqrt(jnp.sum(jnp.abs(data) ** 2))
            return CanDoExpectationValue(
                data,
            )
        else:
            raise NotImplementedError(f"What am I to do? {type(data)=}, {data=}.")

    return _internal
