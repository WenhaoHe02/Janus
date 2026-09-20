# SPDX-License-Identifier: Apache-2.0
"""Check matched serving flags and forbidden prediction/query combinations."""

# Standard
import unittest

# Local
from serving_glm_configuration_v766 import environment


class ConfigurationTests(unittest.TestCase):
    """Every arm disables measurement instrumentation and real-indexer CP."""

    def test_arms_differ_only_in_prediction_identity(self) -> None:
        left, right = environment('k_local'), environment('no_prefetch')
        changed = {key for key in left if left[key] != right[key]}
        self.assertEqual(changed, {'LMCACHE_SERVING_ARM', 'LMCACHE_PROXY_INDEXER_K_CP',
            'LMCACHE_GLM_DSA_OWNER_PARTITION', 'LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH',
            'LMCACHE_GLM_DSA_LAYER_MAJOR', 'LMCACHE_GLM_DEVICE_SELECTION_MISS_READ'})
        for flags in (left, right):
            self.assertEqual(flags['LMCACHE_PROXY_INDEXER_CP'], '0')
            self.assertEqual(flags['LMCACHE_TRUE_INDEXER_CP'], '0')
            self.assertEqual(flags['LMCACHE_TUTTI_CPU_STAGE_GIB'], '2')
            self.assertEqual(flags['LMCACHE_TUTTI_CPU_STAGE_ENABLE'], '1')
            self.assertEqual(flags['LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE'], '1')
            self.assertEqual(flags['LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE'], '0')
            self.assertEqual(flags['LMCACHE_TUTTI_DECODE_WRITE_MIBPS'], '0')
            for key, value in flags.items():
                if any(word in key for word in ('PROFILE', 'TIMING', 'TIMESTAMPS', 'NVTX')):
                    self.assertEqual(value, '0', key)

    def test_unknown_arm_rejected(self) -> None:
        with self.assertRaises(ValueError):
            environment('q_sync')

    def test_host_stage_diagnostic_changes_only_timestamp_flag(self) -> None:
        for arm in ('k_local', 'no_prefetch', 'ssd_only', 'dram'):
            original = environment(arm)
            diagnostic = environment(arm, stage_timing_diagnostic=True)
            changed = {key for key in original if original[key] != diagnostic[key]}
            self.assertEqual(changed, {'LMCACHE_TTFT_STAGE_PROFILE'})
            self.assertEqual(diagnostic['LMCACHE_TTFT_STAGE_PROFILE'], '1')
            self.assertEqual(environment(arm), original)

    def test_ssd_baseline_does_not_enable_tutti_sparse_correction(self) -> None:
        flags = environment('ssd_only')
        self.assertEqual(flags['LMCACHE_GLM_DSA_LAYER_MAJOR'], '0')
        self.assertEqual(flags['LMCACHE_GLM_DEVICE_SELECTION_MISS_READ'], '0')
        self.assertEqual(flags['LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH'], '0')
        self.assertEqual(flags['LMCACHE_PROXY_INDEXER_CP'], '0')

    def test_dram_matches_conventional_retrieval_without_prediction(self) -> None:
        flags = environment('dram')
        expected = environment('ssd_only')
        expected['LMCACHE_SERVING_ARM'] = 'dram'
        self.assertEqual(flags, expected)

    def test_tutti_no_prefetch_does_not_attach_sparse_prediction_manager(self) -> None:
        flags = environment('no_prefetch')
        self.assertEqual(flags['LMCACHE_GLM_DSA_LAYER_MAJOR'], '0')
        self.assertEqual(flags['LMCACHE_GLM_DEVICE_SELECTION_MISS_READ'], '0')
        self.assertEqual(flags['LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH'], '0')
        self.assertEqual(environment('k_local')['LMCACHE_GLM_DSA_LAYER_MAJOR'], '1')


if __name__ == '__main__':
    unittest.main()
