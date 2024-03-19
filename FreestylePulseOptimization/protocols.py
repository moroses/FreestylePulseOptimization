"""Module for base protocols to interact with a Backend"""
from typing import (
    Self,
    Any,
    Protocol,
    Union,
    overload,
    Optional,
    TypeAlias,
)
from collections.abc import Mapping
from datetime import datetime as py_datetime


class DeviceParameterControlChannel(Protocol):
    """Protocol for a `ControlChannel` like object"""
    @property
    def index(self: Self) -> int:
        """
        Get the index of the channel

        Returns
        -------
        int
            The channel index
        """
        ...


class DeviceParameterBackendConfiguration(Protocol):
    """Protocol to get basic properties from the Backend configuration object"""
    @property
    def dt(self: Self) -> float:
        """
        The device `dt`

        Returns
        -------
        float
        """
        ...

    @property
    def hamiltonian(self: Self) -> Mapping[str, Any]:
        """
        The device's Hamiltonian

        Returns
        -------
        Mapping[str, Any]
        """
        ...

    @property
    def control_channels(
        self: Self,
    ) -> Mapping[tuple[int, int], list[DeviceParameterControlChannel]]:
        """
        The device's control channels dictionary

        Returns
        -------
        Mapping[tuple[int, int], list[DeviceParameterControlChannel]]
        """
        ...


class DeviceParameterBackend(Protocol):
    """Protocol to get the Backend's configuration object"""
    def configuration(self: Self) -> DeviceParameterBackendConfiguration:
        """
        Get the configuration object

        Returns
        -------
        DeviceParameterBackendConfiguration
            [TODO:description]
        """
        ...


VALUEwDATETIME: TypeAlias = tuple[float, py_datetime]
"""A tuple of a value and a datetime object"""


class NoiseParameterProperties(Protocol):
    """Protocol to obtain qubit's noise parameters"""
    @overload
    def qubit_property(
        self: Self, qubit: int, name: Optional[str] = None
    ) -> Mapping[str, VALUEwDATETIME]:
        """
        Get all qubit's properties

        Return a dictionary of the qubit properties

        Parameters
        ----------
        qubit
            The qubit's index
        name
            None

        Returns
        -------
        Mapping[str, VALUEwDATETIME]
            All qubit properties
        """
        ...

    @overload
    def qubit_property(self: Self, qubit: int, name: str) -> VALUEwDATETIME:
        """
        Get a given property for a specific qubit

        Return the specific property for a given qubit

        Parameters
        ----------
        qubit
            The qubit index
        name
            The property name

        Returns
        -------
        VALUEwDATETIME
            The given property of the qubit
        """
        ...

    def qubit_property(
        self: Self, qubit: int, name: Optional[str] = None
    ) -> Union[Mapping[str, VALUEwDATETIME], VALUEwDATETIME]:
        """
        Get either all or a specific property for a given qubit

        If property name is given, return that property. Else return all the properties

        Parameters
        ----------
        qubit
            The qubit index
        name
            The property name

        Returns
        -------
        Union[Mapping[str, VALUEwDATETIME], VALUEwDATETIME]
            Either a specific property, or all of the properties
        """
        ...


class NoiseParameterBackend(Protocol):
    """Protocol to get the properties object from a Backend"""
    def properties(
        self: Self, refresh: bool = False, datetime: Optional[py_datetime] = None
    ) -> NoiseParameterProperties | None:
        """
        Get the properties object from a Backend

        Return the properties object from a Backend, can be from a specific datetime or forced to be refreshed.

        Parameters
        ----------
        refresh
            Refresh? Defaults to False
        datetime
            The datetime of the properties object to return

        Returns
        -------
        NoiseParameterProperties | None
            None if there is no properties of the Backend, else the properties object of the Backend
        """
        ...
