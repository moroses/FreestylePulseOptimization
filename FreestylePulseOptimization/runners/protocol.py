from typing import Protocol, Union
from collections.abc import Sequence
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameter import Parameter
from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse

from qiskit_ibm_provider.ibm_backend import IBMBackend as NewBackend
from qiskit_ibm_runtime.ibm_backend import IBMBackend as RuntimeBackend

BACKEND_TYPE = Union[NewBackend, RuntimeBackend]

from qiskit_ibm_provider.job import IBMJob as NewJob
from qiskit_ibm_runtime.runtime_job import RuntimeJob

from ..protocols import NoiseParameterBackend

JOB_TYPE = Union[NewJob, RuntimeJob]

# TODO Add protocols for return jobs and backend features


class RunOnBackend(Protocol):
    def run(self, circuits: Sequence[QuantumCircuit], **options) -> JOB_TYPE: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def backend(self) -> BACKEND_TYPE: ...

    def load(self, job_id: str) -> JOB_TYPE: ...


def get_default_anharmonicity(backend: NoiseParameterBackend, qubit: int) -> float:
    if backend.properties() is None:
        raise AttributeError(f"Backend has no properties! {backend=}")
    properties = backend.properties()
    return properties.qubit_property(qubit, "anharmonicity")[0]


def get_default_drag_parameters(
    backend: BACKEND_TYPE, qubit: int, instruction: str
) -> dict[str, float | complex | int | Parameter]:
    defaults = backend.defaults()
    if defaults is None:
        raise AttributeError(f"Backend has no defaults! {backend=}")
    ins = defaults.instruction_schedule_map.get(instruction, qubit)
    if not hasattr(ins.instructions[0][1], "pulse") or not isinstance(
        ins.instructions[0][1].pulse, ScalableSymbolicPulse
    ):
        raise AttributeError(f"What type of instruction is this? {ins=}")
    pulse_operation: ScalableSymbolicPulse = ins.instructions[0][1].pulse
    pulse_parameters = pulse_operation.parameters
    return dict(pulse_parameters)


def get_n_qubits(backend: BACKEND_TYPE) -> int:
    if hasattr(backend, "num_qubits"):
        return backend.num_qubits
    return backend.configuration().n_qubits
