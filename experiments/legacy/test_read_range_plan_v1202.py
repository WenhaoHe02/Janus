# SPDX-License-Identifier: Apache-2.0
"""Read-command coalescing invariants, independent of the CUDA implementation."""

# Standard
import unittest

# Local
from read_range_plan_v1202 import ReadSpan, coalesce_reads


class RangePlanTests(unittest.TestCase):
    """Merging cannot change requested bytes or scatter destinations."""

    def plan(self, spans: list[ReadSpan], **options: object) -> list:
        """Run the public reference contract with explicit test-device limits."""
        limits = dict(sector_bytes=512, max_command_bytes=16384, max_sgl_entries=8)
        limits.update(options)
        return coalesce_reads(spans, **limits)

    def test_empty_and_single(self) -> None:
        """An empty request emits no command; a single read remains unchanged."""
        self.assertEqual(self.plan([]), [])
        self.assertEqual(self.plan([ReadSpan(1, 'v1', 8, 4096, 65536)])[0].byte_length, 4096)

    def test_disjoint_gpu_destinations_stay_disjoint(self) -> None:
        """SSD adjacency does not imply contiguous GPU IOVAs."""
        spans = [ReadSpan(1, 'v1', i * 8, 4096, (10 - i) * 65536) for i in range(4)]
        result = self.plan(spans)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].destinations, tuple((s.gpu_iova, 4096) for s in spans))
        self.assertEqual(result[0].member_indices, (0, 1, 2, 3))
        self.assertEqual(result[0].byte_length, 16384)
        self.assertEqual(len(self.plan(spans, enabled=False)), 4)

    def test_gaps_overlap_namespace_and_version_do_not_merge(self) -> None:
        """Do not fill holes, deduplicate overlaps, or cross storage identities."""
        first = ReadSpan(1, 'v1', 0, 4096, 65536)
        for second in [ReadSpan(1, 'v1', 9, 4096, 131072),
                       ReadSpan(1, 'v1', 0, 4096, 131072),
                       ReadSpan(2, 'v1', 8, 4096, 131072),
                       ReadSpan(1, 'v2', 8, 4096, 131072)]:
            self.assertEqual(len(self.plan([first, second])), 2)

    def test_transfer_and_descriptor_limits(self) -> None:
        """A contiguous run splits at either independently enforced limit."""
        spans = [ReadSpan(1, 'v1', i * 8, 4096, (i + 1) * 65536) for i in range(5)]
        self.assertEqual([len(c.member_indices) for c in self.plan(spans)], [4, 1])
        self.assertEqual([len(c.member_indices) for c in self.plan(spans, max_sgl_entries=2)], [2, 2, 1])
        self.assertEqual(sum(c.byte_length for c in self.plan(spans)), 5 * 4096)

    def test_invalid_ranges_fail_before_submission(self) -> None:
        """Reject invalid namespaces, sector lengths and address overflow."""
        for span in [ReadSpan(0, 'v1', 0, 4096, 65536),
                     ReadSpan(1, 'v1', 0, 4095, 65536),
                     ReadSpan(1, 'v1', (1 << 64) - 1, 4096, 65536)]:
            with self.assertRaises(ValueError):
                self.plan([span])


if __name__ == '__main__':
    unittest.main()
