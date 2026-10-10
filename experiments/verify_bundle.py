"""Verify an extracted research bundle before executing its replay."""
import json
import hashlib
import tarfile
from oracle_audit.io import ROOT,sha256


def verify():
    manifest=json.loads((ROOT/'bundle_manifest.json').read_text())
    for record in manifest['files']:
        path=(ROOT/record['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise ValueError('Bundle path escapes its root')
        if not path.is_file() or sha256(path)!=record['sha256']:raise ValueError('Missing/modified bundle file: '+record['path'])
    print(f'Verified {len(manifest["files"])} bundle files; status {manifest["scientific_status"]}')


def verify_archive(path):
    prefix='Oracle-Agentic-Audit/'
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        manifest=json.load(archive.extractfile(prefix+'bundle_manifest.json'))
        expected={prefix+r['path']:r for r in manifest['files']}
        if set(names)!=set(expected)|{prefix+'bundle_manifest.json'}:
            raise ValueError('Unindexed or missing archive member')
        for member in members:
            if not member.isfile():raise ValueError('Bundle must contain regular files only')
            from pathlib import PurePosixPath
            if not member.name.startswith(prefix) or '..' in PurePosixPath(member.name).parts:
                raise ValueError('Unsafe archive path')
            if member.name not in expected:continue
            entry=expected[member.name];digest=hashlib.sha256()
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
            if member.size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
                raise ValueError('Archive content mismatch: '+member.name)
    print(f'Verified archive integrity for {len(expected)} files without extracting.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--archive')
    args=p.parse_args()
    verify_archive(args.archive) if args.archive else verify()
