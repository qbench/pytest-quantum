"""Real quantum hardware fixtures for pytest-quantum."""
from __future__ import annotations

import os
from typing import Any

import pytest

from pytest_quantum.plugin import _require


@pytest.fixture(scope="session")
def ibm_backend(request: pytest.FixtureRequest) -> Any:
    """Real IBM Quantum backend via QiskitRuntimeService.

    Requires:
        - --quantum-real CLI flag
        - IBM_QUANTUM_TOKEN environment variable (or ~/.qiskit/qiskit-ibm.json)
        - Optional: IBM_QUANTUM_INSTANCE env var (e.g. "ibm-q/open/main")
        - Optional: IBM_QUANTUM_BACKEND env var (e.g. "ibm_brisbane") -- defaults to least_busy

    Example::

        @pytest.mark.quantum_real
        def test_on_real_hardware(ibm_backend):
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from pytest_quantum import assert_measurement_distribution

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)

            sampler = Sampler(ibm_backend)
            job = sampler.run([qc], shots=1024)
            result = job.result()[0]
            counts = dict(result.data.c.get_counts())
            assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5})
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
    backend_name = os.environ.get("IBM_QUANTUM_BACKEND", "")
    # Valid channels in qiskit-ibm-runtime >= 0.20:
    #   ibm_quantum_platform  — token from quantum.ibm.com (recommended)
    #   ibm_cloud             — IBM Cloud API key + CRN instance
    channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
    min_qubits = int(os.environ.get("IBM_QUANTUM_MIN_QUBITS", "5"))

    if not token:
        pytest.skip(
            "IBM_QUANTUM_TOKEN not set. Export your token:\n"
            "  export IBM_QUANTUM_TOKEN=<your-token>\n"
            "Get one at https://quantum.ibm.com"
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        pytest.skip("qiskit-ibm-runtime not installed: pip install qiskit-ibm-runtime")

    # Try all known valid channel names in order
    channels_to_try = list(
        dict.fromkeys([channel, "ibm_quantum_platform", "ibm_cloud"])
    )
    service = None
    last_error = ""
    for ch in channels_to_try:
        try:
            service = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            break
        except Exception as exc:
            last_error = str(exc)
            continue

    if service is None:
        pytest.skip(
            f"IBM Quantum connection failed: {last_error}\n"
            "  Check IBM_QUANTUM_TOKEN is valid (get it from quantum.ibm.com)."
        )

    try:
        if backend_name:
            return service.backend(backend_name)
        return service.least_busy(
            min_num_qubits=min_qubits, simulator=False, operational=True
        )
    except Exception as exc:
        pytest.skip(f"IBM Quantum backend selection failed: {exc}")


@pytest.fixture(scope="session")
def ionq_backend(request: pytest.FixtureRequest) -> Any:
    """Real IonQ quantum backend via Azure Quantum or IonQ cloud.

    Requires:
        - --quantum-real CLI flag
        - IONQ_API_KEY environment variable
        - Optional: IONQ_BACKEND env var ("simulator" or "qpu.aria-1", "qpu.forte-1", default "simulator")

    Example::

        def test_on_ionq(ionq_backend):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            counts = assert_backend_executes(qc, ionq_backend, shots=1024)
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    api_key = os.environ.get("IONQ_API_KEY", "")
    if not api_key:
        pytest.skip(
            "IONQ_API_KEY not set. Export your IonQ API key:\n"
            "  export IONQ_API_KEY=<your-api-key>\n"
            "Get one at https://cloud.ionq.com"
        )

    ionq_backend_name = os.environ.get("IONQ_BACKEND", "simulator")

    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        pytest.skip("qiskit-ionq not installed: pip install qiskit-ionq")

    try:
        provider = IonQProvider(api_key)
        return provider.get_backend(ionq_backend_name)
    except Exception as exc:
        pytest.skip(f"IonQ connection failed: {exc}")


@pytest.fixture(scope="session")
def quantinuum_backend(request: pytest.FixtureRequest) -> Any:
    """Real Quantinuum quantum backend via pytket-quantinuum.

    Requires:
        - --quantum-real CLI flag
        - QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD environment variables
        - Optional: QUANTINUUM_DEVICE env var (default "H1-1E" emulator)

    Example::

        def test_on_quantinuum(quantinuum_backend): ...
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    username = os.environ.get("QUANTINUUM_USERNAME", "")
    password = os.environ.get("QUANTINUUM_PASSWORD", "")
    if not username or not password:
        pytest.skip(
            "QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD must both be set.\n"
            "  export QUANTINUUM_USERNAME=<your-email>\n"
            "  export QUANTINUUM_PASSWORD=<your-password>\n"
            "Register at https://um.qapi.quantinuum.com"
        )

    device_name = os.environ.get("QUANTINUUM_DEVICE", "H1-1E")

    try:
        from pytket.extensions.quantinuum import QuantinuumBackend
    except ImportError:
        pytest.skip("pytket-quantinuum not installed: pip install pytket-quantinuum")

    try:
        backend = QuantinuumBackend(
            device_name=device_name,
            username=username,
            password=password,
        )
        return backend
    except Exception as exc:
        pytest.skip(f"Quantinuum connection failed: {exc}")


@pytest.fixture(scope="session")
def braket_cloud_device(request: pytest.FixtureRequest) -> Any:
    """AWS Braket cloud quantum device (session-scoped).

    Auto-skips unless ``--quantum-real`` is passed.
    Requires:

    - AWS credentials configured (``aws configure`` or env vars)
    - ``BRAKET_DEVICE_ARN`` environment variable set to the device ARN,
      e.g. ``arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice``

    Usage::

        @pytest.mark.quantum_real
        def test_bell_on_ionq(braket_cloud_device):
            from braket.circuits import Circuit

            circuit = Circuit().h(0).cnot(0, 1).measure_all()
            task = braket_cloud_device.run(circuit, shots=100)
            counts = {str(k): v for k, v in task.result().measurement_counts.items()}
            assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip(
            "AWS Braket cloud test skipped. Pass --quantum-real to enable.\n"
            "Also requires AWS credentials and BRAKET_DEVICE_ARN env var."
        )

    device_arn = os.environ.get("BRAKET_DEVICE_ARN")
    if not device_arn:
        pytest.skip(
            "BRAKET_DEVICE_ARN not set. Set it to the device ARN, e.g.:\n"
            "  export BRAKET_DEVICE_ARN=arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice"
        )

    _require("braket", "pip install pytest-quantum[braket]")

    try:
        from braket.aws import AwsDevice

        return AwsDevice(device_arn)
    except Exception as exc:
        pytest.skip(f"AWS Braket device unavailable: {exc}")


@pytest.fixture(scope="session")
def quantum_hardware_info(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Returns info about configured real hardware backends.

    Checks environment variables to determine which cloud backends have
    credentials configured.  Does **not** attempt a live connection.

    Returns a dict with keys:
        - ``"ibm_available"``        — True if IBM_QUANTUM_TOKEN is set
        - ``"ionq_available"``       — True if IONQ_API_KEY is set
        - ``"quantinuum_available"`` — True if both QUANTINUUM_USERNAME and
          QUANTINUUM_PASSWORD are set
        - ``"aws_available"``        — True if BRAKET_DEVICE_ARN is set

    Example::

        def test_info(quantum_hardware_info):
            info = quantum_hardware_info
            assert isinstance(info["ibm_available"], bool)
    """
    return {
        "ibm_available": bool(os.environ.get("IBM_QUANTUM_TOKEN", "")),
        "ionq_available": bool(os.environ.get("IONQ_API_KEY", "")),
        "quantinuum_available": (
            bool(os.environ.get("QUANTINUUM_USERNAME", ""))
            and bool(os.environ.get("QUANTINUUM_PASSWORD", ""))
        ),
        "aws_available": bool(os.environ.get("BRAKET_DEVICE_ARN", "")),
    }
