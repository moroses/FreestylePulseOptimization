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
    @property
    def index(self: Self) -> int: ...


class DeviceParameterBackendConfiguration(Protocol):
    @property
    def dt(self: Self) -> float: ...
    @property
    def hamiltonian(self: Self) -> Mapping[str, Any]: ...
    @property
    def control_channels(
        self: Self,
    ) -> Mapping[tuple[int, int], list[DeviceParameterControlChannel]]: ...


class DeviceParameterBackend(Protocol):
    def configuration(self: Self) -> DeviceParameterBackendConfiguration: ...


VALUEwDATETIME: TypeAlias = tuple[float, py_datetime]


class NoiseParameterProperties(Protocol):
    @overload
    def qubit_property(
        self: Self, qubit: int, name: Optional[str] = None
    ) -> Mapping[str, VALUEwDATETIME]: ...
    @overload
    def qubit_property(self: Self, qubit: int, name: str) -> VALUEwDATETIME: ...
    def qubit_property(
        self: Self, qubit: int, name: Optional[str] = None
    ) -> Union[Mapping[str, VALUEwDATETIME], VALUEwDATETIME]: ...


class NoiseParameterBackend(Protocol):
    def properties(
        self: Self, refresh: bool = False, datetime: Optional[py_datetime] = None
    ) -> NoiseParameterProperties | None: ...
