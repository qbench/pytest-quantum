import pytest
import numpy as np
from pytest_quantum.assertions.qec import (
    assert_code_distance,
    assert_syndrome_decoding_correct,
    _pauli_str_to_symplectic,
)


class TestAssertCodeDistance:
    def test_three_qubit_bit_flip(self):
        # 3-qubit bit-flip code: stabilizers ZZI, IZZ
        # Distance should be 1 (single X error is a logical operator...
        # actually for repetition code the distance is the code length
        # but with these stabilizers the minimum weight logical is X on any single qubit
        # Wait - for [[3,1,1]] the distance is actually 3 for the repetition code
        # ZZI and IZZ stabilize the code space. Logical X = XXX, Logical Z = ZII (or IZI or IIZ)
        # Minimum weight logical is ZII which has weight 1...
        # Actually ZII commutes with both stabilizers and is not in the stabilizer group
        # So distance = 1. But that's the [[3,1,1]] code which isn't very useful.
        # The repetition code has distance 3 because logical operators are XXX and ZZZ
        # But ZII commutes with ZZI and IZZ and is not in {III, ZZI, IZZ, ZIZ}
        # So distance = 1.
        assert_code_distance(["ZZI", "IZZ"], expected_distance=1)

    def test_single_stabilizer(self):
        # Single stabilizer ZZ on 2 qubits
        # Stabilizer group: {II, ZZ}
        # Normalizer elements that commute with ZZ: anything with even X count
        # Minimum weight: XI or IX (weight 1) - but do these commute with ZZ?
        # XI and ZZ: symplectic product = 1*0 + 0*1 + 0*1 + 1*0 = 0? Let me check
        # XI = (1,0, 0,0), ZZ = (0,0, 1,1)
        # product = 1*1 + 0*1 + 0*0 + 0*0 = 1 (mod 2) = 1, so they anti-commute
        # XX = (1,1, 0,0), ZZ = (0,0, 1,1)
        # product = 1*1 + 1*1 + 0*0 + 0*0 = 2 mod 2 = 0, commutes
        # XX has weight 2, not in {II, ZZ}
        # ZI = (0,0, 1,0), ZZ = (0,0, 1,1)
        # product = 0*1 + 0*1 + 1*0 + 0*0 = 0, commutes
        # ZI has weight 1, not in {II, ZZ}
        assert_code_distance(["ZZ"], expected_distance=1)

    def test_assertion_fails_wrong_distance(self):
        with pytest.raises(AssertionError, match="Code distance is"):
            assert_code_distance(["ZZI", "IZZ"], expected_distance=3)


class TestAssertSyndromeDecodingCorrect:
    def test_correct_decoding(self):
        stabilizers = ["ZZI", "IZZ"]
        # X error on qubit 0: symplectic (1,0,0, 0,0,0)
        error = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)
        # Decoder corrects with same error
        correction = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)
        assert_syndrome_decoding_correct(stabilizers, error, correction)

    def test_correct_decoding_with_stabilizer_diff(self):
        stabilizers = ["ZZ"]
        # Error: XI = (1,0, 0,0)
        # Correction: XI = (1,0, 0,0)
        error = np.array([1, 0, 0, 0], dtype=np.int8)
        correction = np.array([1, 0, 0, 0], dtype=np.int8)
        # Residual = II, which is in stabilizer group
        assert_syndrome_decoding_correct(stabilizers, error, correction)

    def test_string_inputs(self):
        stabilizers = ["ZZI", "IZZ"]
        assert_syndrome_decoding_correct(stabilizers, "III", "III")

    def test_incorrect_decoding_fails(self):
        stabilizers = ["ZZI", "IZZ"]
        # Error: X on qubit 0
        error = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)
        # Wrong correction: X on qubit 1
        correction = np.array([0, 1, 0, 0, 0, 0], dtype=np.int8)
        # Residual: XX0 = (1,1,0, 0,0,0) - check if this is in stabilizer group
        # Stabilizer group: {III, ZZI, IZZ, ZIZ}
        # (1,1,0,0,0,0) is not in the group, and it commutes with stabilizers
        # So it's a logical operator -> should fail
        with pytest.raises(AssertionError):
            assert_syndrome_decoding_correct(stabilizers, error, correction)


class TestPauliSymplectic:
    def test_identity(self):
        vec = _pauli_str_to_symplectic("I")
        np.testing.assert_array_equal(vec, [0, 0])

    def test_x(self):
        vec = _pauli_str_to_symplectic("X")
        np.testing.assert_array_equal(vec, [1, 0])

    def test_z(self):
        vec = _pauli_str_to_symplectic("Z")
        np.testing.assert_array_equal(vec, [0, 1])

    def test_y(self):
        vec = _pauli_str_to_symplectic("Y")
        np.testing.assert_array_equal(vec, [1, 1])

    def test_multi_qubit(self):
        vec = _pauli_str_to_symplectic("XZ")
        np.testing.assert_array_equal(vec, [1, 0, 0, 1])
