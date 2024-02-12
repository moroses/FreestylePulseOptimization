from dataclasses import dataclass, asdict
from typing import Any, Optional, Callable, Protocol
from collections.abc import Sequence, Mapping
from matplotlib.cbook import file_requires_unicode
import numpy.typing as npt
import qiskit
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from pathlib import Path
from qiskit.providers.jobstatus import JobStatus
import json
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit import pulse
import qiskit
from qiskit_ibm_provider import IBMProvider, IBMJob
from qiskit_ibm_provider.ibm_backend import IBMBackend
from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse, Drag
from qiskit_ibm_runtime import QiskitRuntimeService, Session, RuntimeJob
from qiskit import transpile

from dataclasses import is_dataclass, asdict

from lmfit.model import save_modelresult
from lmfit.models import LorentzianModel, ConstantModel, SineModel

from ..runners.protocol import (
    BACKEND_TYPE,
    JOB_TYPE,
    get_default_anharmonicity,
    get_n_qubits,
    RunOnBackend,
)
from .utils import (
    cos_fit_guess,
    extrema,
    lorenzian_guess,
    transform_range,
    lorenzian_fit,
    cos_fit,
    get_default_parameters,
    BaseDragPulseFactors,
)

from ..runtime.utils import (
    HWCI_MatrixSolutionRuntimeLogical,
    HWCI_MatrixSolutionRuntime,
    convert_channels_to_scheduleblock,
    convert_parameters_to_channels,
)
from ..hw import HWCI_MatrixSolution
from ..runtime.utils import compose, run_on_index, partial
from ..runtime.utils import Parameter as MP


# TODO Fix the general measurement protocol
# Current schema is working for a single qubit *only!*


def freq_offset_drag_schedule(
    backend: BACKEND_TYPE,
    qubit: int,
    freq: float | Parameter,
    name: str = "freq_offset_drag",
    **parameters,
) -> ScheduleBlock:
    drag_pulse = Drag(**parameters)
    with pulse.build(backend=backend, name=name) as schd:
        dq = pulse.drive_channel(qubit)
        with pulse.frequency_offset(freq, dq):
            pulse.play(drag_pulse, dq)
    return schd


def build_01_freq_hunt(
    backend: BACKEND_TYPE,
    qubit: int,
    freq_shifts: npt.ArrayLike,
    absolute_freq: bool = False,
    base_factors: Optional[BaseDragPulseFactors] = None,
    **kwargs,
) -> Sequence[QuantumCircuit]:
    num_qubits = get_n_qubits(backend)
    orig_qc = QuantumCircuit(num_qubits, num_qubits)
    def_freq = backend.defaults().qubit_freq_est[qubit]
    if not absolute_freq:
        def_freq = 0.0
    # pulse_parameters = get_default_drag_parameters(backend, qubit, "x")
    pulse_parameters = get_default_parameters(backend, qubit)
    pulse_parameters.update(**kwargs)

    if base_factors is not None:
        pulse_parameters = base_factors(pulse_parameters)

    freq_param = Parameter("freq")

    fh01 = freq_offset_drag_schedule(
        backend, qubit, freq_param, name="01 freq hunt", **pulse_parameters
    )

    gate = Gate("01_freq_hunt", 1, [freq_param])
    orig_qc.add_calibration(gate, [qubit], fh01, [freq_param])
    orig_qc.metadata = {
        "name": "01 freq hunt",
        "absolute_freq": absolute_freq,
        "parameter": kwargs,
    }

    orig_qc.append(gate, [qubit])
    orig_qc.measure(qubit, qubit)

    ret: list[QuantumCircuit] = []
    for f in freq_shifts:
        ff = f + def_freq
        qc = orig_qc.assign_parameters({freq_param: ff})
        qc.metadata["f"] = ff
        ret.append(qc)

    return transpile(ret, backend)


def build_12_freq_hunt(
    backend: BACKEND_TYPE,
    qubit: int,
    freq_shifts: npt.ArrayLike,
    x01_schedule: ScheduleBlock,
    absolute_freq: bool = True,
    base_factors: Optional[BaseDragPulseFactors] = None,
    **kwargs,
) -> Sequence[QuantumCircuit]:
    num_qubits = get_n_qubits(backend)
    orig_qc = QuantumCircuit(num_qubits, num_qubits)
    def_anharmonic = get_default_anharmonicity(backend, qubit)
    if not absolute_freq:
        def_anharmonic = 0.0
    # pulse_parameters = get_default_drag_parameters(backend, qubit, "x")
    pulse_parameters = get_default_parameters(backend, qubit)
    pulse_parameters.update(kwargs)

    if base_factors is not None:
        pulse_parameters = base_factors(pulse_parameters)

    freq_param = Parameter("freq")

    fh12 = freq_offset_drag_schedule(
        backend, qubit, freq_param, name="12 freq hunt", **pulse_parameters
    )
    gate = Gate("12_freq_hunt", 1, [freq_param])
    orig_qc.add_calibration(gate, [qubit], fh12, [freq_param])
    orig_qc.metadata = {
        "name": "12 hunt",
        "absolute_freq": absolute_freq,
        "parameter": kwargs,
    }

    x01_gate = Gate("x01", 1, [])
    orig_qc.add_calibration(x01_gate, [qubit], x01_schedule, [])
    orig_qc.append(x01_gate, [qubit])

    orig_qc.append(gate, [qubit])
    orig_qc.measure(qubit, qubit)

    ret: list[QuantumCircuit] = []
    for f in freq_shifts:
        ff = f + def_anharmonic
        qc = orig_qc.assign_parameters({freq_param: ff})
        qc.metadata["f"] = ff
        ret.append(qc)

    return transpile(ret, backend)


def build_01_amp_hunt(
    backend: BACKEND_TYPE,
    qubit: int,
    amps: npt.ArrayLike,
    freq: float,
    base_factors: Optional[BaseDragPulseFactors] = None,
    **kwargs,
) -> Sequence[QuantumCircuit]:
    num_qubits = get_n_qubits(backend)
    orig_qc = QuantumCircuit(num_qubits, num_qubits)

    # pulse_parameters = get_default_drag_parameters(backend, qubit, "x")
    pulse_parameters = get_default_parameters(backend, qubit)
    pulse_parameters.update(**kwargs)

    if base_factors is not None:
        pulse_parameters = base_factors(pulse_parameters)

    amp_param = Parameter("amp")
    pulse_parameters["amp"] = amp_param

    amp_schd = freq_offset_drag_schedule(
        backend, qubit, freq, name="01 amp hunt", **pulse_parameters
    )
    amp_gate = Gate("01_amp", 1, [amp_param])

    orig_qc.add_calibration(amp_gate, [qubit], amp_schd, [amp_param])
    orig_qc.metadata = {"name": "01 amp hunt", "parameter": kwargs}
    orig_qc.append(amp_gate, [qubit])
    orig_qc.measure(qubit, qubit)

    ret: list[QuantumCircuit] = []
    for amp in amps:
        qc = orig_qc.assign_parameters({amp_param: amp})
        qc.metadata["amp"] = amp
        ret.append(qc)

    return transpile(ret, backend)


def build_12_amp_hunt(
    backend: BACKEND_TYPE,
    qubit: int,
    amps: npt.ArrayLike,
    freq: float,
    x01_schedule: ScheduleBlock,
    base_factors: Optional[BaseDragPulseFactors] = None,
    **kwargs,
) -> Sequence[QuantumCircuit]:
    num_qubits = get_n_qubits(backend)
    orig_qc = QuantumCircuit(num_qubits, num_qubits)

    # pulse_parameters = get_default_drag_parameters(backend, qubit, "x")
    pulse_parameters = get_default_parameters(backend, qubit)
    pulse_parameters.update(**kwargs)

    if base_factors is not None:
        pulse_parameters = base_factors(pulse_parameters)

    amp_param = Parameter("amp")
    pulse_parameters["amp"] = amp_param

    a12s = freq_offset_drag_schedule(
        backend, qubit, freq, name="12 amp hunt", **pulse_parameters
    )

    gate = Gate("12_amp_hunt", 1, [amp_param])
    orig_qc.add_calibration(gate, [qubit], a12s, [amp_param])

    x01_gate = Gate("x01", 1, [])
    orig_qc.add_calibration(x01_gate, [qubit], x01_schedule, [])
    orig_qc.append(x01_gate, [qubit])

    orig_qc.append(gate, [qubit])
    orig_qc.measure(qubit, qubit)

    orig_qc.metadata = {
        "name": "12 amp hunt",
        "freq": freq,
        "parameter": kwargs,
    }

    ret: list[QuantumCircuit] = []
    for amp in amps:
        qc = orig_qc.assign_parameters({amp_param: amp})
        qc.metadata["amp"] = amp
        ret.append(qc)

    return transpile(ret, backend)


# TODO URGENT - Make the frequency and amplitude functions more general


def build_012_measurements(
    backend: BACKEND_TYPE,
    qubit: int,
    x01_schedule: ScheduleBlock,
    x12_schedule: ScheduleBlock,
) -> Sequence[QuantumCircuit]:
    num_qubits = get_n_qubits(backend)

    qc0 = QuantumCircuit(num_qubits, num_qubits)
    qc0.metadata = {
        "name": "discriminator",
        "state": 0,
    }

    qc1 = qc0.copy()
    x01_gate = Gate("x01", 1, [])
    qc1.add_calibration(x01_gate, [qubit], x01_schedule, [])
    qc1.append(x01_gate, [qubit])
    qc1.metadata["state"] = 1

    qc2 = qc1.copy()
    x12_gate = Gate("x12", 1, [])
    qc2.add_calibration(x12_gate, [qubit], x12_schedule, [])
    qc2.append(x12_gate, [qubit])
    qc2.metadata["state"] = 2

    for qc in (qc0, qc1, qc2):
        qc.measure(qubit, qubit)

    return transpile([qc0, qc1, qc2], backend)


def find_01_freq(
    runner: RunOnBackend,
    qubit: int,
    base_dir: Path | str,
    n_points: int,
    min_f: float,
    max_f: float,
    shots: int = 1024,
    job_id: Optional[str] = None,
    **kwargs,
) -> tuple[float, str]:
    default_args = {"amp": 0.2}
    default_args.update(kwargs)
    base_dir = Path(base_dir)
    base_dir /= "01_freq"
    base_dir.mkdir(parents=True, exist_ok=True)
    backend = runner.backend

    freqs = np.linspace(min_f, max_f, n_points, endpoint=True)

    if job_id is None:
        circuits = build_01_freq_hunt(backend, qubit, freqs, **default_args)
        job = runner.run(
            circuits,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            shots=shots,
        )
        job_id = job.job_id()
    else:
        job = runner.load(job_id)

    # Quick hack to fix base_factors issues
    kwargs2 = dict(kwargs)
    for key in kwargs2.keys():
        if is_dataclass(value := kwargs2[key]):
            kwargs2[key] = asdict(value)

    (base_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "min_f": min_f,
                "max_f": max_f,
                "n_points": n_points,
                "backend_name": runner.backend_name,
                "qubit": qubit,
                "kwargs": kwargs2,
            }
        ),
        encoding="utf-8",
    )
    job.wait_for_final_state()
    if job.status() != JobStatus.DONE:
        raise RuntimeError(f"Job failed ! {job_id=}")

    result = job.result()
    IQ_raw = np.array([r.data.memory[qubit] for r in result.results])
    IQ_raw = np.array([r.data.memory[qubit] for r in result.results])
    (base_dir / "raw_data.json").write_text(
        json.dumps(
            {
                "IQ": IQ_raw.tolist(),
            }
        ),
        encoding="utf-8",
    )
    I_raw = IQ_raw[:, 0]
    # I_raw = np.abs(I_raw)  # Quick fix --- This was not working the way it should!
    I_norm = transform_range(0.0, 1.0, *extrema(I_raw))(I_raw)
    fig, axs = plt.subplots(figsize=plt.figaspect(1))
    GHz = 1e9
    freqs_GHz = freqs / GHz
    axs.scatter(freqs_GHz, I_norm, label="Raw data")
    # TODO Fix this part to use lmfit protocol
    if False:
        p0 = lorenzian_guess(freqs_GHz, I_norm)
        params, _ = curve_fit(lorenzian_fit, freqs_GHz, I_norm, p0=p0)
        amplitude, center, width, vertical_shift = params
        (base_dir / "fit.json").write_text(
            json.dumps(
                {
                    "amplitude": amplitude,
                    "center": center,
                    "width": width,
                    "vertical_shift": vertical_shift,
                }
            ),
            encoding="utf-8",
        )
        # label = f"$\\frac{{{amplitude:e}}}{{\\pi}}\\frac{{{B:e}}}{{(x-{anharmonicity_calc:e})^2+({B:e})^2}}+({C:e})$"
        label = f"$\\frac{{{amplitude:e}}}{{1+\\left(\\frac{{x-{center:e}}}{{{width:e}}}\\right)^2}}+{vertical_shift:e}$"
        label = r"$f\left(x\right)=\frac{A}{1+\left(\frac{x-B}{C}\right)^2}+D$"
        axs.plot(
            freqs_GHz, lorenzian_fit(freqs_GHz, *params), ls=":", label=f"Fit {label}"
        )
    else:
        model = (loren_model := LorentzianModel(prefix="loren_")) + (
            const_model := ConstantModel(prefix="const_")
        )
        p0 = loren_model.guess(I_norm, x=freqs_GHz) + const_model.make_params(
            c=I_norm.min()
        )
        fit = model.fit(I_norm, x=freqs_GHz, params=p0)
        label = r"$f\left(x\right)=\frac{A}{1+\left(\frac{x-B}{C}\right)^2}+D$"
        label = r"$\frac{A}{\pi}\frac{\sigma}{\left(x-\mu\right)^2+\sigma^2}+D$"
        axs.plot(freqs_GHz, fit.best_fit, ls=":", label=f"Fit {label}")
        save_modelresult(fit, base_dir / "fit.dat")
        p1 = fit.params
        amplitude = p1["loren_amplitude"].value
        center = p1["loren_center"].value
        width = p1["loren_sigma"].value
        vertical_shift = p1["const_c"].value
    axs.set(xlabel="Frequency offset [GHz]", ylabel="I []")
    axs.legend()
    fig.tight_layout()
    fig.savefig(base_dir / "figure.png", dpi=300)
    fig.savefig(base_dir / "figure.svg")
    return center * GHz, job_id


def find_01_amp(
    runner: RunOnBackend,
    qubit: int,
    base_dir: Path | str,
    freq: float,
    n_points: int,
    min_amp: float,
    max_amp: float,
    shots: int = 1024,
    job_id: Optional[str] = None,
    **kwargs,
) -> tuple[float, str]:
    base_dir = Path(base_dir)
    base_dir /= "01_amp"
    base_dir.mkdir(parents=True, exist_ok=True)

    default_kwargs = {}
    default_kwargs.update(**kwargs)

    amps = np.linspace(min_amp, max_amp, num=n_points, endpoint=True)

    backend = runner.backend
    backend_name = runner.backend_name
    # job = backend.run(circuits, meas_level=MeasLevel.KERNELED, meas_return=MeasReturnType.AVERAGE, shots=shots)
    if job_id is None:
        circuits = build_01_amp_hunt(backend, qubit, amps, freq)
        job = runner.run(
            circuits,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            shots=shots,
        )
        job_id = job.job_id()
    else:
        job = runner.load(job_id)

    # Quick hack to fix base_factors issues
    kwargs2 = dict(kwargs)
    for key in kwargs2.keys():
        if is_dataclass(value := kwargs2[key]):
            kwargs2[key] = asdict(value)

    (base_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "min_amp": min_amp,
                "max_amp": max_amp,
                "n_points": n_points,
                "backend_name": backend_name,
                "qubit": qubit,
                "freq": freq,
                "kwargs": kwargs2,
            }
        ),
        encoding="utf-8",
    )
    job.wait_for_final_state()
    if job.status() != JobStatus.DONE:
        raise RuntimeError(f"Job failed! {job.job_id()=}")
    result = job.result()
    IQ_raw = np.array([r.data.memory[qubit] for r in result.results])
    (base_dir / "raw_data.json").write_text(
        json.dumps({"IQ": IQ_raw.tolist()}), encoding="utf-8"
    )
    I_raw = IQ_raw[:, 0]
    I_norm = transform_range(0.0, 1.0, *extrema(I_raw))(I_raw)
    fig, axs = plt.subplots(figsize=plt.figaspect(1))
    axs.set(xlabel="Amplitude []", ylabel="I []")

    axs.scatter(amps, I_norm, label="Raw data")

    if False:
        p0 = cos_fit_guess(amps, I_norm)

        params, _ = curve_fit(cos_fit, amps, I_norm, p0=p0)
        amplitude, omega, phi, vertical_shift = params

        (base_dir / "fit.json").write_text(
            json.dumps(
                {
                    "amplitude": amplitude,
                    "omega": omega,
                    "phi": phi,
                    "vertical_shift": vertical_shift,
                }
            ),
            encoding="utf-8",
        )

        label = f"${amplitude:e}\\cos\\left(2\\pi {omega:e}x+2\\pi{phi:e}\\right)+{vertical_shift:e}$"
        label = r"$f\left(x\right)=A\cos\left(2\pi\left(B x+C\right)\right)+D$"

        axs.plot(amps, cos_fit(amps, *params), label=f"Fit {label}")
    else:
        model = (sine_model := SineModel(prefix="sine_")) + (
            const_model := ConstantModel(prefix="const_")
        )
        p0 = sine_model.guess(I_norm, x=amps) + const_model.make_params(c=I_norm.min())
        fit = model.fit(I_norm, x=amps, params=p0)
        save_modelresult(fit, base_dir / "fit.dat")
        p1 = fit.params
        label = r"$A\operatorname{sin}\left(fx+\phi\right)+D$"
        amplitude = p1["sine_amplitude"].value
        f = p1["sine_frequency"].value
        phi = p1["sine_shift"].value
        vertical_shift = p1["const_c"].value
        omega = f / 2 / np.pi
        axs.plot(amps, fit.best_fit, ls=":", label=f"Fit {label}")

    axs.legend()
    fig.tight_layout()

    fig.savefig(base_dir / "figure.png", dpi=300)
    fig.savefig(base_dir / "figure.svg")

    return 1 / omega / 2, job_id


def find_12_freq(
    runner: RunOnBackend,
    qubit: int,
    base_dir: Path | str,
    n_points: int,
    min_f: float,
    max_f: float,
    x01_schedule: ScheduleBlock,
    shots: int = 1024,
    job_id: Optional[str] = None,
    **kwargs,
) -> tuple[float, str]:
    default_kwargs = {"amp": 0.2}
    default_kwargs.update(**kwargs)
    base_dir = Path(base_dir)
    base_dir /= "12_freq"
    base_dir.mkdir(parents=True, exist_ok=True)
    backend = runner.backend
    def_anharmonicity = get_default_anharmonicity(backend, qubit)
    freqs = def_anharmonicity + np.linspace(min_f, max_f, num=n_points, endpoint=True)
    if job_id is None:
        circuits = build_12_freq_hunt(
            backend, qubit, freqs, x01_schedule, absolute_freq=False, **default_kwargs
        )
        job = runner.run(
            circuits,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            shots=shots,
        )
        job_id = job.job_id()
    else:
        job = runner.load(job_id)

    # Quick hack to fix base_factors issues
    kwargs2 = dict(kwargs)
    for key in kwargs2.keys():
        if is_dataclass(value := kwargs2[key]):
            kwargs2[key] = asdict(value)

    (base_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "min_f": min_f,
                "max_f": max_f,
                "n_points": n_points,
                "backend_name": runner.backend_name,
                "qubit": qubit,
                "kwargs": kwargs2,
            }
        ),
        encoding="utf-8",
    )
    job.wait_for_final_state()
    if job.status() != JobStatus.DONE:
        raise RuntimeError(f"Job failed! {job.job_id()=}")
    result = job.result()
    IQ_raw = np.array([r.data.memory[qubit] for r in result.results])
    (base_dir / "raw_data.json").write_text(
        json.dumps(
            {
                "IQ": IQ_raw.tolist(),
            }
        ),
        encoding="utf-8",
    )
    I_raw = IQ_raw[:, 0]
    I_raw = np.abs(I_raw)  # Quick fix
    I_norm = transform_range(0.0, 1.0, *extrema(I_raw))(I_raw)
    fig, axs = plt.subplots(figsize=plt.figaspect(1))
    GHz = 1e9
    freqs_GHz = freqs / GHz
    axs.scatter(freqs_GHz, I_norm, label="Raw data")
    if False:
        p0 = lorenzian_guess(freqs_GHz, I_norm)
        params, _ = curve_fit(lorenzian_fit, freqs_GHz, I_norm, p0=p0)
        amplitude, center, width, vertical_shift = params
        (base_dir / "fit.json").write_text(
            json.dumps(
                {
                    "amplitude": amplitude,
                    "center": center,
                    "width": width,
                    "vertical_shift": vertical_shift,
                }
            ),
            encoding="utf-8",
        )
        # label = f"$\\frac{{{amplitude:e}}}{{\\pi}}\\frac{{{B:e}}}{{(x-{anharmonicity_calc:e})^2+({B:e})^2}}+({C:e})$"
        label = f"$\\frac{{{amplitude:e}}}{{1+\\left(\\frac{{x-{center:e}}}{{{width:e}}}\\right)^2}}+{vertical_shift:e}$"
        label = r"$f\left(x\right)=\frac{A}{1+\left(\frac{x-B}{C}\right)^2}+D$"
        axs.plot(
            freqs_GHz, lorenzian_fit(freqs_GHz, *params), ls=":", label=f"Fit {label}"
        )
    else:
        model = (loren_model := LorentzianModel(prefix="loren_")) + (
            const_model := ConstantModel(prefix="const_")
        )
        p0 = loren_model.guess(I_norm, x=freqs_GHz) + const_model.make_params(
            c=I_norm.min()
        )
        fit = model.fit(I_norm, x=freqs_GHz, params=p0)
        label = r"$f\left(x\right)=\frac{A}{1+\left(\frac{x-B}{C}\right)^2}+D$"
        label = r"$\frac{A}{\pi}\frac{\sigma}{\left(x-\mu\right)^2+\sigma^2}+D$"
        axs.plot(freqs_GHz, fit.best_fit, ls=":", label=f"Fit {label}")
        save_modelresult(fit, base_dir / "fit.dat")
        p1 = fit.params
        amplitude = p1["loren_amplitude"].value
        center = p1["loren_center"].value
        width = p1["loren_sigma"].value
        vertical_shift = p1["const_c"].value
    axs.set(xlabel="Frequency offset [GHz]", ylabel="I []")
    axs.legend()
    fig.tight_layout()
    fig.savefig(base_dir / "figure.png", dpi=300)
    fig.savefig(base_dir / "figure.svg")
    return center * GHz, job_id


def find_12_amp(
    runner: RunOnBackend,
    qubit: int,
    base_dir: Path | str,
    freq: float,
    n_points: int,
    min_amp: float,
    max_amp: float,
    x01_schedule: ScheduleBlock,
    shots: int = 1024,
    job_id: Optional[str] = None,
    **kwargs,
) -> tuple[float, str]:
    base_dir = Path(base_dir)
    base_dir /= "12_amp"
    base_dir.mkdir(parents=True, exist_ok=True)

    default_kwargs = {}
    default_kwargs.update(**kwargs)

    amps = np.linspace(min_amp, max_amp, num=n_points, endpoint=True)

    backend = runner.backend
    backend_name = runner.backend_name
    # job = backend.run(circuits, meas_level=MeasLevel.KERNELED, meas_return=MeasReturnType.AVERAGE, shots=shots)
    if job_id is None:
        circuits = build_12_amp_hunt(
            backend, qubit, amps, freq, x01_schedule, **default_kwargs
        )
        job = runner.run(
            circuits,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            shots=shots,
        )
        job_id = job.job_id()
    else:
        job = runner.load(job_id)

    # Quick hack to fix base_factors issues
    kwargs2 = dict(kwargs)
    for key in kwargs2.keys():
        if is_dataclass(value := kwargs2[key]):
            kwargs2[key] = asdict(value)

    (base_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "min_amp": min_amp,
                "max_amp": max_amp,
                "n_points": n_points,
                "backend_name": backend_name,
                "qubit": qubit,
                "freq": freq,
                "kwargs": kwargs2,
            }
        ),
        encoding="utf-8",
    )
    job.wait_for_final_state()
    if job.status() != JobStatus.DONE:
        raise RuntimeError(f"Job failed! {job.job_id()=}")
    result = job.result()
    IQ_raw = np.array([r.data.memory[qubit] for r in result.results])
    (base_dir / "raw_data.json").write_text(
        json.dumps({"IQ": IQ_raw.tolist()}), encoding="utf-8"
    )
    I_raw = IQ_raw[:, 0]
    I_norm = transform_range(0.0, 1.0, *extrema(I_raw))(I_raw)
    fig, axs = plt.subplots(figsize=plt.figaspect(1))
    axs.set(xlabel="Amplitude []", ylabel="I []")

    axs.scatter(amps, I_norm, label="Raw data")

    if False:
        p0 = cos_fit_guess(amps, I_norm)

        params, _ = curve_fit(cos_fit, amps, I_norm, p0=p0)
        amplitude, omega, phi, vertical_shift = params

        (base_dir / "fit.json").write_text(
            json.dumps(
                {
                    "amplitude": amplitude,
                    "omega": omega,
                    "phi": phi,
                    "vertical_shift": vertical_shift,
                }
            ),
            encoding="utf-8",
        )

        label = f"${amplitude:e}\\cos\\left(2\\pi {omega:e}x+2\\pi{phi:e}\\right)+{vertical_shift:e}$"
        label = r"$f\left(x\right)=A\cos\left(2\pi\left(B x+C\right)\right)+D$"

        axs.plot(amps, cos_fit(amps, *params), label=f"Fit {label}")
    else:
        model = (sine_model := SineModel(prefix="sine_")) + (
            const_model := ConstantModel(prefix="const_")
        )
        p0 = sine_model.guess(I_norm, x=amps) + const_model.make_params(c=I_norm.min())
        fit = model.fit(I_norm, x=amps, params=p0)
        save_modelresult(fit, base_dir / "fit.dat")
        p1 = fit.params
        label = r"$A\operatorname{sin}\left(fx+\phi\right)+D$"
        amplitude = p1["sine_amplitude"].value
        f = p1["sine_frequency"].value
        phi = p1["sine_shift"].value
        vertical_shift = p1["const_c"].value
        omega = f / 2 / np.pi
        axs.plot(amps, fit.best_fit, ls=":", label=f"Fit {label}")

    axs.legend()
    fig.tight_layout()

    fig.savefig(base_dir / "figure.png", dpi=300)
    fig.savefig(base_dir / "figure.svg")

    return 1 / omega / 2, job_id


def experiment_result_to_ndarray(
    experiment_results: Sequence[ExperimentResult], experiment_index: int, qubit: int
) -> npt.NDArray[float]:
    return np.array(experiment_results[experiment_index].data.memory)[:, qubit, :]


@dataclass(frozen=True, slots=True, kw_only=True)
class QuadRange:
    minimal: float
    maximal: float
    base: float = 10.0
    exp: float = 0.0

    @property
    def mult(self: "QuadRange") -> float:
        return self.base**self.exp

    @classmethod
    def create(
        cls: type["QuadRange"],
        extrema: tuple[float, float],
        exp: float = 0.0,
        base: float = 10.0,
    ) -> "QuadRange":
        return cls(
            minimal=extrema[0],
            maximal=extrema[1],
            exp=exp,
            base=base,
        )


def find_012_LDA(
    runner: RunOnBackend,
    qubit: int,
    base_dir: str | Path,
    x01_schedule: ScheduleBlock,
    x12_schedule: ScheduleBlock,
    shots: int = 1024,
    job_id: Optional[str] = None,
    create_figure: Optional[int] = None,
) -> tuple[LinearDiscriminantAnalysis, float, tuple[QuadRange, QuadRange], str]:
    base_dir = Path(base_dir)
    base_dir /= "012_LDA"
    base_dir.mkdir(parents=True, exist_ok=True)

    backend = runner.backend

    if job_id is None:
        circuits = build_012_measurements(backend, qubit, x01_schedule, x12_schedule)
        job = runner.run(
            circuits,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
            shots=shots,
        )
        job_id = job.job_id()
    else:
        job = runner.load(job_id)
    (base_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "backend_name": runner.backend_name,
                "qubit": qubit,
                "shots": shots,
            }
        ),
        encoding="utf-8",
    )
    job.wait_for_final_state()
    if job.status() != JobStatus.DONE:
        raise RuntimeError(f"Job failed! {job_id=}")
    result = job.result()

    meas0 = experiment_result_to_ndarray(result.results, 0, qubit)
    meas1 = experiment_result_to_ndarray(result.results, 1, qubit)
    meas2 = experiment_result_to_ndarray(result.results, 2, qubit)

    (base_dir / "raw_data.json").write_text(
        json.dumps(
            {
                "meas0": meas0.tolist(),
                "meas1": meas1.tolist(),
                "meas2": meas2.tolist(),
            }
        ),
        encoding="utf-8",
    )

    meas_all = np.concatenate((meas0, meas1, meas2))

    meas_all_expI = np.floor(np.log10(np.abs(np.mean(extrema(meas_all[:, 0])))))
    meas_all_expQ = np.floor(np.log10(np.abs(np.mean(extrema(meas_all[:, 1])))))

    meas_all_expI = np.floor(np.log10(np.max(np.abs(meas_all[:, 0]))))
    meas_all_expQ = np.floor(np.log10(np.max(np.abs(meas_all[:, 1]))))

    meas_all /= (10**meas_all_expI, 10**meas_all_expQ)
    meas_all = meas_all.astype(d_type := np.min_scalar_type(np.max(meas_all)))

    meas0 /= (10**meas_all_expI, 10**meas_all_expQ)
    meas0 = meas0.astype(d_type := np.min_scalar_type(np.max(meas0)))
    meas1 /= (10**meas_all_expI, 10**meas_all_expQ)
    meas1 = meas1.astype(d_type := np.min_scalar_type(np.max(meas0)))
    meas2 /= (10**meas_all_expI, 10**meas_all_expQ)
    meas2 = meas2.astype(d_type := np.min_scalar_type(np.max(meas0)))

    states_all = np.concatenate([i * np.ones(shots, dtype=np.uint8) for i in range(3)])

    train012, test012, train012_states, test012_states = train_test_split(
        meas_all, states_all, test_size=0.5
    )

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(train012, train012_states)

    LDA_score = LDA.score(test012, test012_states)

    Irange = QuadRange.create(extrema(meas_all[:, 0]), meas_all_expI)
    Qrange = QuadRange.create(extrema(meas_all[:, 1]), meas_all_expQ)

    if create_figure is not None:
        Il = np.linspace(Irange.minimal, Irange.maximal, create_figure, endpoint=True)
        Ql = np.linspace(Qrange.minimal, Qrange.maximal, create_figure, endpoint=True)

        I, Q = np.meshgrid(Il, Ql)
        states = LDA.predict(np.c_[I.ravel(), Q.ravel()]).reshape(I.shape)
        fig, axs = plt.subplot_mosaic(
            mosaic=[
                ["0", "c"],
                ["1", "c"],
                ["2", "c"],
            ],
            figsize=(1.2, 3) * plt.figaspect(1),
            width_ratios=[0.9, 0.1],
        )
        colors = plt.get_cmap("Set2", 3)
        state_labels = ["0", "1", "2"]
        contour = None
        for state in state_labels:
            axs[state].set_title(rf"$\left|{state}\right\rangle$")
            contour = axs[state].contourf(I, Q, states, cmap=colors, alpha=0.2)
            axs[state].set(
                xlabel=f"I [$10^{{{Irange.mult}}}$]",
                ylabel=f"Q [$10^{{{Qrange.mult}}}$]",
            )
        colorbar = fig.colorbar(contour, cax=axs["c"])
        colorbar.set_ticks([0, 1, 2])
        colorbar.set_label("State")
        colorbar.set_ticklabels(
            [rf"$\left|{state}\right\rangle$" for state in state_labels]
        )

        colors = plt.get_cmap("tab20", 6)(np.linspace(0.0, 1.0, 6))
        for i, (state, meas) in enumerate(zip(state_labels, (meas0, meas1, meas2))):
            axs[state].scatter(meas[:, 0], meas[:, 1], alpha=0.5, s=2, c=colors[2 * i])
            # TODO Maybe add center of mass
        fig.tight_layout()

        fig.savefig(base_dir / "figure.png", dpi=300)
        fig.savefig(base_dir / "figure.svg")
        plt.close(fig)

    (base_dir / "LDA.pkl").write_bytes(
        pickle.dumps(
            {
                "LDA": LDA,
                "LDA_score": LDA_score,
                "Irange": Irange,
                "Qrange": Qrange,
                "meas_all": meas_all,
            }
        )
    )

    return (
        LDA,
        LDA_score,
        (Irange, Qrange),
        job_id,
    )
    # return LDA, LDA_score, extrema(meas_all[:, 0]), extrema(meas_all[:, 1]), meas_all_expI, meas_all_expQ, job_id
    # return LDA, LDA_score, IQS(I=X, Q=Y, S=classes), job_id
