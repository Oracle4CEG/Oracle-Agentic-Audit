import tempfile
import unittest
from pathlib import Path
from oracle_audit.core import checked_file, evidence_is_admissible, metrics
from oracle_audit.__main__ import release_check, template_check


class ContractTests(unittest.TestCase):
    def test_bad_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.csv'
            path.write_text('modified input')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                checked_file(path, '0'*64)

    def test_missing_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Missing input'):
                checked_file(Path(temp)/'missing', '0'*64)

    def test_future_and_outcome_evidence_excluded(self):
        self.assertFalse(evidence_is_admissible(11, 10))
        self.assertFalse(evidence_is_admissible(9, 10, is_outcome=True))
        self.assertTrue(evidence_is_admissible(10, 10))

    def test_zero_coverage_selective_error_undefined(self):
        result = metrics([{'case_id': 'SYN-1', 'run_id': '0', 'label': '1',
                           'p_reject': '0.5', 'action': 'investigate'}])
        self.assertEqual(result['coverage'], 0)
        self.assertIsNone(result['selective_error'])
        self.assertAlmostEqual(result['mean_action_loss'], 0.1)
        self.assertAlmostEqual(result['brier'], 0.25)

    def test_duplicate_run_rejected(self):
        row = {'case_id': 'SYN-1', 'run_id': '0', 'label': '1',
               'p_reject': '0.5', 'action': 'investigate'}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            metrics([row, row])

    def test_incomplete_research_release_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Research release BLOCKED'):
            release_check()

    def test_template_checksum_and_index(self):
        template_check()
