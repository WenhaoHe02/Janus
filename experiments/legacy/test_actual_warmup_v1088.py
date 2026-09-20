# SPDX-License-Identifier: Apache-2.0
"""Tests against the native warmup identity and ordering contract."""

# Standard
import unittest

# Local
from audit_actual_warmup_v1088 import validate_actual_warmup


def fixture() -> tuple[dict,dict,list[dict]]:
    """Return a minimal valid one-request preparation record."""
    row = dict(source_request_id='a',input_token_sha256='hash',prompt_tokens=110,
               reusable_input_prefix_tokens=100,new_input_tokens=10)
    manifest = dict(native_content_warmup=True,chains=[dict(first_timed_index=1,rounds=[{},row])])
    prepared = dict(warmup_input_mode='actual_native_content')
    events = [dict(phase='actual_content_warmup_excluded',source_request_id='a',
                   input_token_sha256='hash',H=100,A=10,completion=dict(prompt_tokens=110)),
              dict(phase='actual_warmup_cache_cleared'),
              dict(phase='original_predecessor_setup_excluded')]
    return manifest, prepared, events


class WarmupTests(unittest.TestCase):
    """A shape match alone cannot satisfy the content warmup contract."""

    def test_accepts_actual_content(self) -> None:
        """Matching content, length and ordering are accepted."""
        self.assertEqual(validate_actual_warmup(*fixture())['count'],1)

    def test_rejects_different_content_same_shape(self) -> None:
        """Synthetic tokens with identical H/A are not treated as actual warmup."""
        m,p,e = fixture()
        e[0]['input_token_sha256']='different'
        with self.assertRaises(ValueError):
            validate_actual_warmup(m,p,e)

    def test_rejects_seed_before_cleanup(self) -> None:
        """Formal seed preceding the cleanup invalidates this evidence."""
        m,p,e = fixture()
        e[1],e[2]=e[2],e[1]
        with self.assertRaises(ValueError):
            validate_actual_warmup(m,p,e)


if __name__=='__main__':
    unittest.main()
