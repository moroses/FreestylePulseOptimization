from typing import Callable, Optional, Any, TypeAlias
from typing_extensions import Self
from collections.abc import Mapping, Sequence
from qiskit.circuit import QuantumCircuit

from copy import deepcopy

from qiskit.providers.jobstatus import JobStatus

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    RuntimeJob,
    Options,
    RuntimeOptions,
)

from qiskit_ibm_runtime.ibm_backend import IBMBackend

from .runtime import RuntimeRunner

"A function to test if the job is ok."
TEST_FUNCTION: TypeAlias = Callable[[Optional[RuntimeJob]], bool]
"A function to handle errors in a job and try to recover prior to running again."
REACTION_FUNCTION: TypeAlias = Callable[
    ["RetryRuntimeRunner", Optional[RuntimeJob]], None
]


def good_status(job: Optional[RuntimeJob]) -> bool:
    if job is None:
        return False
    return job.status() == JobStatus.DONE


def recover_from_session(
    runner: "RetryRuntimeRunner", job: Optional[RuntimeJob]
) -> None:
    if job is None:
        return
    if job.status() == JobStatus.ERROR:
        reason = job._reason
        if reason is not None:
            if "session" in reason.lower():
                runner.close_session()
                runner.open_session()
        else:
            print(f"Unknown error for {job.job_id()}!")


class RetryRuntimeRunner(RuntimeRunner):
    __N: int
    __session_kwargs: Mapping[str, Any]
    __test_function: TEST_FUNCTION
    __reaction_function: REACTION_FUNCTION

    def __init__(
        self: Self,
        channel: str,
        instance: str,
        backend_name: str,
        test_function: TEST_FUNCTION = good_status,
        reaction_function: REACTION_FUNCTION = recover_from_session,
        token: Optional[str] = None,
        N: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(channel, instance, backend_name, token)
        self.__N = N
        self.__session_kwargs = deepcopy(dict(kwargs))
        self.__test_function = test_function
        self.__reaction_function = reaction_function

    @property
    def session_kwargs(self: Self) -> Mapping[str, Any]:
        return self.__session_kwargs

    @session_kwargs.setter
    def session_kwargs(self: Self, values: Mapping[str, Any]) -> None:
        self.__session_kwargs = {k: v for k, v in values.items()}

    def set_session_kwargs(self: Self, values: Mapping[str, Any]) -> None:
        self.session_kwargs = values

    def open_session(self, **kwargs) -> None:
        return super().open_session(**self.__session_kwargs, **kwargs)

    def run(self, circuits: Sequence[QuantumCircuit], **options: Any) -> RuntimeJob:
        i = 0
        errors: list[str] = []
        while i < self.__N:
            print(f"Running {i}...")
            job = super().run(circuits, **options)
            job.wait_for_final_state()

            if self.__test_function(job):
                return job
            errors.append(f"{job.session_id} - {job.job_id()}: {job.error_message()}")

            self.__reaction_function(self, job)

            i += 1
        err_str = "\n".join(errors)
        raise RuntimeError(
            f"Job failed {i} times !!\nI am lost for words...\n{err_str}"
        )
