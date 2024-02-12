from typing import Optional, Any
from typing_extensions import Self
from collections.abc import Sequence, Mapping

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    RuntimeJob,
    Options,
    RuntimeOptions,
)
from qiskit_ibm_runtime.ibm_backend import IBMBackend


class RuntimeRunner:
    __backend_name: str
    __instance: str
    __channel: str
    __token: str | None
    __service: QiskitRuntimeService | None
    __session: Session | None
    __backend: IBMBackend | None
    __additional_kwargs: Mapping[str, Any]

    def __init__(
        self: Self,
        channel: str,
        instance: str,
        backend_name: str,
        token: Optional[str] = None,
    ) -> None:
        self.__channel = channel
        self.__instance = instance
        self.__token = token
        self.__backend_name = backend_name
        self.__service = self.__session = self.__backend = None
        self.__additional_kwargs = {}

    def connect(self: Self) -> None:
        if self.__service is not None:
            return
        service = QiskitRuntimeService(
            channel=self.__channel, instance=self.__instance, token=self.__token
        )
        self.__service = service
        self.__backend = service.backend(self.__backend_name)

    def open_session(self: Self, **kwargs) -> None:
        if self.__session is not None:
            return
        self.__session = Session(
            service=self.__service, backend=self.__backend_name, **kwargs
        )

    def close_session(self: Self) -> None:
        if self.__session is None:
            return
        self.__session.close()
        self.__session = None

    @property
    def backend_name(self: Self) -> str:
        return self.__backend_name

    @property
    def backend(self: Self) -> IBMBackend:
        if self.__service is None:
            self.connect()
        assert self.__backend is not None, "What is this?!?!"
        return self.__backend

    @property
    def session(self: Self) -> Session:
        if self.__session is None:
            self.open_session()
        assert self.__session is not None, "What the hell?!?!?"
        return self.__session

    @property
    def service(self: Self) -> QiskitRuntimeService:
        if self.__service is None:
            self.connect()
        assert self.__service is not None, "What happened?"
        return self.__service

    def load(self: Self, job_id: str) -> RuntimeJob:
        if self.__service is None:
            self.connect()
        assert self.__service is not None, "What?!?!"
        return self.__service.job(job_id)

    @property
    def additional_options(self: Self) -> Mapping[str, Any]:
        return self.__additional_kwargs

    @additional_options.setter
    def additional_options(self: Self, value: Mapping[str, Any]) -> None:
        self.__additional_kwargs = {k: v for k, v in value.items()}

    def set_additional_options(self: Self, **additional_options: Any) -> None:
        self.additional_options = additional_options
        # self.__additional_kwargs = {
        #     k: v
        #     for k, v in additional_options.items()
        # }

    def run(
        self: Self, circuits: Sequence[QuantumCircuit], **options: Any
    ) -> RuntimeJob:
        self.open_session()
        session = self.session
        inputs = {
            "circuits": circuits,
            "skip_transpilation": True,
            **self.__additional_kwargs,
            **options,
        }
        return session.run(
            program_id="circuit-runner",
            inputs=inputs,
        )

    @classmethod
    def from_service_and_session(
        cls: type[Self],
        service: QiskitRuntimeService,
        backend_name: str,
        session: Optional[Session] = None,
        **kwargs,
    ) -> Self:
        active_account = service.active_account()
        runner = cls(
            channel=active_account.get("channel", None),
            token=active_account.get("token", None),
            instance=active_account.get("instance", None),
            backend_name=backend_name,
            **kwargs,
        )
        runner.__service = service
        runner.__session = session
        runner.__backend = service.backend(backend_name)
        return runner

    @classmethod
    def least_busy_backend(
        cls: type["RuntimeRunner"],
        channel: str,
        instance: str,
        token: Optional[str] = None,
    ) -> "RuntimeRunner":
        service = QiskitRuntimeService(channel=channel, instance=instance, token=token)
        backend_name = service.least_busy(simulator=False).name
        return cls.from_service_and_session(service, backend_name)
