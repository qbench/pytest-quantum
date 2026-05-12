"""Integration tests for CUDA Quantum adapter.

All tests are skipped if cudaq is not installed.
"""

import pytest

cudaq = pytest.importorskip("cudaq")


class TestCudaQuantumAdapter:
    def test_detect(self):
        from pytest_quantum.adapters.cuda_quantum import CudaQuantumAdapter

        adapter = CudaQuantumAdapter()
        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.h(q)
        assert adapter.detect(kernel)

    def test_detect_rejects_non_cudaq(self):
        from pytest_quantum.adapters.cuda_quantum import CudaQuantumAdapter

        adapter = CudaQuantumAdapter()
        assert not adapter.detect("not a kernel")

    def test_framework_name(self):
        from pytest_quantum.adapters.cuda_quantum import CudaQuantumAdapter

        assert CudaQuantumAdapter.framework_name == "cuda_quantum"
