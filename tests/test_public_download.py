import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from experiments.download import verify_archive, extract


class PublicArchiveContract(unittest.TestCase):
    def archive(self, root, name='data/local/example.json', payload=b'{}', expected=None):
        path = root/'bundle.tar.gz'
        manifest = dict(format='oracle-public-artifact-v1', artifact='research', files=[
            dict(path=name, bytes=len(payload), sha256=hashlib.sha256(expected or payload).hexdigest())])
        with tarfile.open(path, 'w:gz') as archive:
            for filename, data in [(name, payload), ('artifact_manifest.json', json.dumps(manifest).encode())]:
                info = tarfile.TarInfo(filename); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path

    def test_tampering_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.archive(Path(temp), expected=b'original')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_archive(path, 'research')

    def test_path_traversal_and_code_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ['data/local/../../code.py', '/tmp/injected', 'analysis/replay.py']:
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    verify_archive(self.archive(Path(temp), name=name), 'research')

    def test_existing_different_scientific_records_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = self.archive(root)
            manifest = verify_archive(path, 'research')
            extract(path, manifest, root/'checkout')
            target = root/'checkout/data/local/example.json'
            target.write_bytes(b'my existing experiment')
            with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                extract(path, manifest, root/'checkout')
            self.assertEqual(target.read_bytes(), b'my existing experiment')
