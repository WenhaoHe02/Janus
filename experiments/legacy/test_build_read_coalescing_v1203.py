# SPDX-License-Identifier: Apache-2.0
"""CPU-only safeguards for constructing the isolated READ extension."""

# Standard
from pathlib import Path
import unittest

# Local
from build_read_coalescing_v1203 import extension_sources


class BuildContractTests(unittest.TestCase):
    """Preserve original code and reject ambiguous source transformation."""

    def test_original_cuda_remains_byte_for_byte_prefix(self) -> None:
        """The additional entry point must not rewrite baseline kernels."""
        root = Path(__file__).resolve().parent
        source = (root.parent / 'csrc/tutti_kv_ops.cu').read_text(encoding='utf-8')
        suffix = (root / 'read_range_coalescing_v1203.cu').read_text(encoding='utf-8')
        binding = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n}\n'
        generated, bindings = extension_sources(source, binding, suffix)
        self.assertEqual(generated[:len(source)], source)
        self.assertTrue(generated.endswith(suffix))
        self.assertEqual(bindings.count('m.def("read_coalesced"'), 1)
        self.assertIn('py::call_guard<py::gil_scoped_release>()', bindings)
        self.assertNotIn('launch_coalesced_v1203', generated)

    def test_ambiguous_binding_is_rejected(self) -> None:
        """A missing or duplicate module anchor cannot silently omit the API."""
        source = 'k_poll_batch( k_validate_counted_read( make_qd('
        for binding in ('', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {' * 2):
            with self.assertRaises(ValueError):
                extension_sources(source, binding, '')

    def test_unreviewed_or_unfinished_source_is_rejected(self) -> None:
        """Required queue routines and complete launcher are explicit gates."""
        binding = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {'
        with self.assertRaises(ValueError):
            extension_sources('', binding, '')
        with self.assertRaises(ValueError):
            extension_sources('k_poll_batch( k_validate_counted_read( make_qd(',
                              binding, 'launch_coalesced_v1203(')


if __name__ == '__main__':
    unittest.main()
