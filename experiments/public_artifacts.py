"""Build separate, deterministic Atlas-input and research-record archives.

This packages existing records; it never contacts an inference service or fabricates
missing records. Code and manifests are delivered in Git, not duplicated as a
replacement repository in the dataset. Historical source snapshots remain evidence.
"""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from oracle_audit.io import ROOT, save_json, sha256

ATLAS_INPUTS = {
    'polygon_oov2_events.parquet', 'polygon_uma_request_rounds.parquet',
    'polygon_uma_request_flow_qc.parquet', 'uma_polygon_ethereum_grade_a_links.parquet',
}
RECORD_ROOTS = [
    'data/local', 'outputs/qwen_supplement', 'outputs/guard_experiment',
    'outputs/infrastructure_attempts', 'outputs/scope_amendment', 'outputs/final',
]


def paths():
    selected = {'atlas': [], 'research': []}
    for name in RECORD_ROOTS:
        for path in sorted((ROOT/name).rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if '__pycache__' in rel.parts or path.suffix in {'.pyc', '.log', '.joblib'}:
                continue
            if path.name.startswith('.env') or path.is_symlink():
                raise ValueError('Unexpected secret/symlink candidate: '+str(rel))
            # Refit baseline estimators during replay, never deserialize a public pickle.
            kind = 'atlas' if rel.parent == Path('data/local') and path.name in ATLAS_INPUTS else 'research'
            selected[kind].append(path)
    return selected


def build(out):
    receipt = json.loads((ROOT/'outputs/final/replay_receipt.json').read_text())
    if receipt['status'] != 'COMPLETE_OFFLINE_REPLAY' or receipt['bootstrap_resamples'] != 2000:
        raise ValueError('Complete saved-record replay required')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for kind, files in paths().items():
        manifest = dict(format='oracle-public-artifact-v1', artifact=kind,
                        files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                    sha256=sha256(p)) for p in files])
        target = out/(kind+'-20261010.tar.gz')
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as archive:
                for path in files:
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    info.mode = 0o644
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
                data = (json.dumps(manifest, indent=2)+'\n').encode()
                info = tarfile.TarInfo('artifact_manifest.json'); info.size = len(data); info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
        save_json(out/(kind+'-manifest.json'), manifest)
        artifacts.append(dict(artifact=kind, filename=target.name, sha256=sha256(target),
                              bytes=target.stat().st_size, files=len(files)))
    save_json(out/'artifacts.json', artifacts)
    print(json.dumps(artifacts, indent=2))
    return artifacts


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'outputs/public_release')
    build(p.parse_args().out)
