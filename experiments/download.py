"""Download immutable public inputs without credentials; verify before extraction."""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_archive(path, expected_kind):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Archive contains duplicate paths or non-regular members')
        manifest = json.load(archive.extractfile('artifact_manifest.json'))
        if manifest['format'] != 'oracle-public-artifact-v1' or manifest['artifact'] != expected_kind:
            raise ValueError('Wrong artifact identity')
        records = {r['path']: r for r in manifest['files']}
        if len(records) != len(manifest['files']) or set(names) != set(records)|{'artifact_manifest.json'}:
            raise ValueError('Missing or unindexed archive files')
        for member in members:
            if member.name == 'artifact_manifest.json':
                continue
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or p.as_posix() != member.name or p.parts[:2] not in (
                ('data', 'local'), ('outputs', 'qwen_supplement'), ('outputs', 'guard_experiment'),
                ('outputs', 'infrastructure_attempts'), ('outputs', 'scope_amendment'), ('outputs', 'final')):
                raise ValueError('Unsafe/non-data archive path: '+member.name)
            record = records[member.name]
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    h.update(block)
            if member.size != record['bytes'] or h.hexdigest() != record['sha256']:
                raise ValueError('Archive content mismatch: '+member.name)
    return manifest


def extract(path, manifest, root):
    root = Path(root).resolve()
    # Refuse different pre-existing data rather than overwrite someone's experiments.
    for record in manifest['files']:
        target = root/record['path']
        if not target.resolve().is_relative_to(root):
            raise ValueError('Destination symlink escapes root')
        if target.exists() and (not target.is_file() or digest(target) != record['sha256']):
            raise ValueError('Existing file differs; use a fresh checkout: '+record['path'])
    with tarfile.open(path, 'r:gz') as archive:
        for record in manifest['files']:
            target = root/record['path']
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(record['path']) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)


def download(root=ROOT):
    root = Path(root).resolve()
    release = json.loads((ROOT/'manifests/public-release.json').read_text())
    revision = release['dataset_revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('An immutable 40-character dataset revision is required')
    cache = root/'outputs/downloads'; cache.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='PASS', dataset=release['dataset_id'], revision=revision,
                   authentication='anonymous HTTPS', artifacts=[])
    for spec in release['artifacts']:
        url = f"https://huggingface.co/datasets/{release['dataset_id']}/resolve/{revision}/{spec['path']}"
        path = cache/Path(spec['path']).name
        if not path.exists() or digest(path) != spec['sha256']:
            temporary = path.with_suffix('.part')
            try:
                with urlopen(url, timeout=120) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if temporary.stat().st_size != spec['bytes'] or digest(temporary) != spec['sha256']:
                    raise ValueError('Downloaded archive checksum/length mismatch')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        manifest = verify_archive(path, spec['artifact'])
        extract(path, manifest, root)
        receipt['artifacts'].append(dict(**spec, verified_files=len(manifest['files'])))
        print(f"Verified and extracted {spec['artifact']}: {len(manifest['files'])} files", flush=True)
    (cache/'download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    download(p.parse_args().root)
